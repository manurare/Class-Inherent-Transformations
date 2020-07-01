import argparse
import os
import torch
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import models, transforms
import numpy as np
from data_utils import *
from loss import GeneratorLoss
from model import Generator
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler
from pytorchtools import EarlyStopping


parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--data_size', default=64, type=int, help='imgs resized to (data_size,data_size)')
parser.add_argument('--batch_size', default=32, type=int, help='train batch_size number')
parser.add_argument('--num_epochs', default=50, type=int, help='train epoch number')
parser.add_argument('--lambda_class', default=1, type=float)
parser.add_argument('--all_lambdas', default=False, action='store_true')
parser.add_argument('--classifier_name', default="resnet18", type=str)
parser.add_argument('--dataset', type=str, default="cat_vs_dog")

opt = parser.parse_args()

NUM_EPOCHS = opt.num_epochs
BATCH_SIZE = opt.batch_size
LAMBDA_CLASS = opt.lambda_class
CLASSIFIER_NAME = opt.classifier_name.lower()
DATASET = opt.dataset.lower()
DATA_SIZE = opt.data_size


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))


def validate(G_dict, classifier, val_loader, criterion_classifier, class_names):
    valid_losses = []  # Store losses for each val img

    # Set models for predicting
    for class_name in class_names:
        G_dict[class_name].eval()
    classifier.eval()

    val_bar = tqdm(val_loader)
    val_results = {'mse': 0, 'batch_sizes': 0, 'd_corrects': 0}
    y_true = []
    y_pred = []

    # Begin inference
    for input_img, _, label in val_bar:
        label = label.to(device)
        batch_size = input_img.size(0)
        val_results['batch_sizes'] += batch_size
        input_img.requires_grad = False
        input_img = input_img.to(device)

        classifier_outputs = []
        for idx, class_name in enumerate(class_names):
            sr = G_dict[class_name](input_img)
            out = classifier(sr)
            classifier_outputs.append(out)

        total_outputs = torch.cat(classifier_outputs, 1).squeeze()
        max_val, idx_max = torch.max(total_outputs, 0)
        preds = idx_max % len(class_names)
        val_loss = criterion_classifier(torch.cat(classifier_outputs, 0).squeeze(),
                                        label.repeat(len(class_names))).item()
        valid_losses.append(val_loss)
        val_results['d_corrects'] += torch.sum(preds == label).item()
        y_true.append(label.item())
        y_pred.append(preds.item())

        val_bar.set_description(
            desc='[Validating]: Acc_D: %.4f' % (
                val_results['d_corrects'] / val_results['batch_sizes']))

    val_loss = np.average(np.asarray(valid_losses))
    print("Valid Loss = {}".format(val_loss))

    return val_loss, y_true, y_pred, val_results


def train(train_set):
    class_names = train_set.classes
    print(train_set.classes)

    # Choose classifier indicated by opt.classifier_name
    switcher = {
        'resnet50': models.resnet50,
        'resnet18': models.resnet18
    }
    # Get the function from switcher dictionary
    func = switcher[CLASSIFIER_NAME]

    # In case classes are unbalanced, balance them with different weights
    unique, counts = np.unique(np.asarray(train_set.targets), return_counts=True)
    class_weights = np.max(counts)/counts
    class_weights /= np.max(class_weights)

    criterion_classifier = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).float().to(device))
    print("weights = {}".format(class_weights))

    print(CLASSIFIER_NAME)
    print("Data size %d" % DATA_SIZE)

    if opt.all_lambdas:
        lambda_values = [1, 0.5, 0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.00075, 5e-4]
    else:
        lambda_values = [LAMBDA_CLASS]

    for lambda_class in lambda_values:
        if lambda_class >= 1.0:
            lambda_class = int(lambda_class)
        print("CHOSEN LAMBDA {}".format(lambda_class))

        # 3-FOLD
        epoch_when_EearlyStopping = NUM_EPOCHS  # Monitor epoch when Early Stopping happens
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1337)
        for fold_idx, (train_index, val_index) in enumerate(kf.split(X=np.zeros(len(train_set.samples)), y=train_set.targets)):
            best_acc = 0.0  # Save model with best val acc
            G_dict = {}     # Array of Generators (one per class)
            G_criterion_dict = {}
            optimizers_dict = {}

            results = {'d_loss': [], 'd_acc_train': [], 'd_acc_val': []}
            for class_name in class_names:
                results['g_loss_' + class_name] = []

            for class_name in class_names:
                netG = Generator()
                G_dict[class_name] = netG.cuda()
                generator_criterion = GeneratorLoss()
                G_criterion_dict[class_name] = generator_criterion.cuda()
                optimizers_dict[class_name] = optim.Adam(G_dict[class_name].parameters(),
                                                         lr=0.0001, weight_decay=1e-4)

            # Pretrained classifier
            classifier = func(pretrained=True)

            # Freeze everything except fully connected
            ct = 0
            for child in classifier.children():
                print("FREEZING")
                ct += 1
                if ct < 7:
                    for param in child.parameters():
                        param.requires_grad = False

            # Change last layer to output N classes
            num_ftrs = classifier.fc.in_features
            classifier.fc = nn.Linear(num_ftrs, len(class_names))
            classifier.name = CLASSIFIER_NAME
            classifier.cuda()

            print("FOLD {}".format(fold_idx))
            train_sampler = SubsetRandomSampler(train_index)
            val_sampler = SubsetRandomSampler(val_index)
            train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, num_workers=4, sampler=train_sampler)
            val_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=1, sampler=val_sampler)

            optimizerD = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
            exp_lr_scheduler = lr_scheduler.StepLR(optimizerD, step_size=5, gamma=0.1)
            early_stopping = EarlyStopping(patience=10, verbose=True)

            # Begin training
            for epoch in range(1, NUM_EPOCHS + 1):
                train_bar = tqdm(train_loader)
                running_results = {'batch_sizes': 0, 'd_loss': 0, 'd_corrects': 0, 'd_score': 0, 'g_score': 0}
                for class_name in class_names:
                    running_results['g_loss_'+class_name] = 0

                for class_name in class_names:
                    G_dict[class_name].train()

                classifier.train()
                exp_lr_scheduler.step()
                for data, target, label in train_bar:
                    batch_size = data.size(0)
                    running_results['batch_sizes'] += batch_size

                    input_img = Variable(target)
                    if torch.cuda.is_available():
                        input_img = input_img.cuda()
                        label = label.to(device)
                    z = Variable(data)
                    if torch.cuda.is_available():
                        z = z.cuda()

                    transformed_imgs = []
                    unfold_labels = []
                    # For each image in the batch, transform it N times (one per generator). Unfold GT label as many
                    # times as transformed versions of an image (N times).
                    for idx, ind_label in enumerate(label):
                        for class_name in class_names:
                            tr_image = G_dict[class_name](z[idx].unsqueeze(0))[0]
                            transformed_imgs.append(tr_image)
                            unfold_labels.append(ind_label.item())

                    transformed_imgs = torch.stack(transformed_imgs)
                    unfold_labels = torch.LongTensor(unfold_labels).to(device)

                    # Predict transformed images
                    classifier_outputs = classifier(transformed_imgs)
                    classifier_outputs = classifier_outputs.to(device)
                    loss_classifier = criterion_classifier(classifier_outputs, unfold_labels)

                    # Optimize classifier
                    optimizerD.zero_grad()
                    loss_classifier.backward(retain_graph=True)
                    optimizerD.step()

                    # Optimize array of generators
                    ce_toGenerator = {}  # Cross entropy loss for each generator
                    for class_name in class_names:
                        current_label = class_names.index(class_name)
                        _, preds = torch.max(classifier_outputs, 1)
                        class_n_labels = unfold_labels[np.where(unfold_labels.cpu() == current_label)]
                        indexes = torch.from_numpy(np.where(unfold_labels.cpu() == current_label)[0]).to(device)
                        # Choose predictions for the current class
                        class_n_outputs = torch.index_select(classifier_outputs, 0, indexes)
                        if class_n_labels.shape[0] != 0:    # Maybe there are not samples of the current class in the batch
                            ce_toGenerator[class_names[current_label]] = criterion_classifier(class_n_outputs, class_n_labels)
                        else:
                            ce_toGenerator[class_name] = 0.0

                    # Backprop one time per generator
                    for idx, class_name in enumerate(class_names):
                        G_dict[class_name].zero_grad()
                        g_loss = G_criterion_dict[class_name](transformed_imgs[idx::len(class_names)], input_img,
                                                              ce_toGenerator[class_name], float(lambda_class))
                        if idx < len(class_names)-1:
                            g_loss.backward(retain_graph=True)
                        else:
                            g_loss.backward()
                        optimizers_dict[class_name].step()

                    # Re-do computations for obtaining loss after weight updates
                    transformed_imgs = []
                    for idx, ind_label in enumerate(label):
                        for class_name in class_names:
                            tr_image = G_dict[class_name](z[idx].unsqueeze(0))[0]
                            transformed_imgs.append(tr_image)
                    transformed_imgs = torch.stack(transformed_imgs)
                    for idx, class_name in enumerate(class_names):
                        g_loss = G_criterion_dict[class_name](transformed_imgs[idx::len(class_names)], input_img,
                                                              ce_toGenerator[class_name], float(lambda_class))
                        running_results['g_loss_'+class_name] += g_loss.item() * batch_size

                    running_results['d_loss'] += loss_classifier.item() * batch_size
                    _, preds = torch.max(classifier_outputs, 1)
                    running_results['d_corrects'] += torch.sum(preds == unfold_labels.data).item()

                    train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Acc_D: %.4f Loss_G_class1: %.4f Loss_G_class2: %.4f' % (
                        epoch, NUM_EPOCHS, running_results['d_loss'] / (len(class_names)*running_results['batch_sizes']),
                        running_results['d_corrects'] / (len(class_names)*running_results['batch_sizes']),
                        running_results['g_loss_'+class_names[0]] / (len(class_names)*running_results['batch_sizes']),
                        running_results['g_loss_'+class_names[1]] / (len(class_names)*running_results['batch_sizes'])))

                # Validate model
                valid_loss, y_true, y_pred, val_results = \
                    validate(G_dict, classifier, val_loader, criterion_classifier, class_names)

                out_folder = 'weights/{}_{}_lambda{}_{}/'.format(DATASET, classifier.name, lambda_class, DATA_SIZE)

                if not os.path.exists(out_folder):
                    os.makedirs(out_folder)

                # Save model if there is an increase in val acc
                curr_acc = metrics.accuracy_score(y_true, y_pred)
                print("Valid Acc = {}".format(curr_acc))
                if curr_acc >= best_acc:
                    best_acc = curr_acc
                    if not os.path.exists(out_folder):
                        os.makedirs(out_folder)
                    for class_name in class_names:
                        torch.save(G_dict[class_name].state_dict(),
                                   "{}best_{}_netG_{}.pth".format(out_folder, fold_idx, class_name))
                    torch.save(classifier.state_dict(), "{}best_{}_netD.pth".format(out_folder, fold_idx))

                results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
                results['d_acc_train'].append(running_results['d_corrects'] / running_results['batch_sizes'])
                results['d_acc_val'].append(val_results['d_corrects'] / val_results['batch_sizes'])
                for class_name in class_names:
                    results['g_loss_'+class_name].append(running_results['g_loss_'+class_name] / running_results['batch_sizes'])

                # Save model if val loss decreases
                early_stopping(valid_loss, classifier, G_dict, out_folder, fold_idx)

                if early_stopping.early_stop:
                    print("Early stopping")
                    epoch_when_EearlyStopping = epoch
                    break
                else:
                    epoch_when_EearlyStopping = epoch

            # Save statistics
            out_path = 'statistics_' + DATASET + os.sep
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            data_dict = {'Loss_D': results['d_loss'], 'Acc_D_train': results['d_acc_train'],
                         'Acc_D_val': results['d_acc_val']}
            for class_name in class_names:
                data_dict['Loss_G_'+class_name] = results['g_loss_'+class_name]
            data_frame = pd.DataFrame(data_dict, index=range(1, epoch_when_EearlyStopping + 1))
            if fold_idx > 0:
                data_frame.to_csv(out_path + 'srf_' + DATASET + '_' + str(classifier.name) + '_lambda' +
                                  str(lambda_class) + '_' + str(DATA_SIZE) + '_train_results.csv',
                                  index_label='Epoch', mode='a', header=False)
            else:
                data_frame.to_csv(out_path + 'srf_' + DATASET + '_' + str(classifier.name) + '_lambda' +
                                  str(lambda_class) + '_' + str(DATA_SIZE) + '_train_results.csv', index_label='Epoch')


if __name__ == "__main__":

    data_dir = os.path.join("data", DATASET, 'train')
    print(data_dir)
    train_set = ImageFolderWithPaths_noUps(data_dir, DATA_SIZE,
                                           img_transforms=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                                              transforms.RandomAffine(5),
                                                                              transforms.RandomRotation(5)]))
    train(train_set)

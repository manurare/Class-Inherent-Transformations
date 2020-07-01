import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import models, transforms
import os
from sklearn import metrics

from data_utils import ImageFolderWithPaths
import pandas as pd
import torchnet.meter.confusionmeter as cm
from PIL import Image
from model import Generator
import sys

parser = argparse.ArgumentParser(description='multiclass classification')
parser.add_argument('--lambda_class', default=-1.0, type=float)
parser.add_argument('--classifier_name', default="resnet18", type=str)
parser.add_argument('--dataset', type=str, default="cat_vs_dog")
parser.add_argument('--weight_type', type=str, default="best")
parser.add_argument('--all_lambdas', default=False, action='store_true')
parser.add_argument('--kfold', type=int, default=1)
parser.add_argument('--data_size', required=True, type=int)
parser.add_argument('--test_split', type=str, default="")   # Inference in another independent test split

opt = parser.parse_args()
if opt.kfold != parser.get_default('kfold') and (opt.lambda_class == -1.0) and not opt.all_lambdas:
    parser.error("--kfold bigger than 0 requires --lambda_class != -1.0 or opt.all_lambdas = True")
    sys.exit()

LAMBDA_CLASS = opt.lambda_class
CLASSIFIER_NAME = opt.classifier_name
DATASET = opt.dataset.lower()
ALL_LAMBDAS = opt.all_lambdas
DATA_SIZE = opt.data_size
KFOLD = opt.kfold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=(DATA_SIZE, DATA_SIZE), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=(DATA_SIZE, DATA_SIZE), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=(DATA_SIZE, DATA_SIZE), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])
}


def predict():
    # CLASSIFIER given by opt.classifier_name
    switcher = {
        'resnet50': models.resnet50,
        'resnet18': models.resnet18
    }
    # Get the function from switcher dictionary
    func = switcher[CLASSIFIER_NAME]

    classifier = func(pretrained=True)
    classifier.name = CLASSIFIER_NAME
    print(classifier.name)
    print("KFOLD = {}".format(KFOLD))

    # Generate a csv where metrics are to be saved
    if ALL_LAMBDAS:
        lambda_values = [1, 0.5, 0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.00075, 5e-4]
        lambda_values = lambda_values[::-1]
        csv_file = "metrics/{}_allLambdas_{}_{}_{}".format(DATASET, LAMBDA_CLASS, CLASSIFIER_NAME, DATA_SIZE)
    else:
        csv_file = "metrics/{}_lambda_{}_{}_{}".format(DATASET, LAMBDA_CLASS, CLASSIFIER_NAME, DATA_SIZE)
        lambda_values = [LAMBDA_CLASS]

    if not os.path.exists("metrics/"):
        os.makedirs("metrics/")

    model_phases = ['train', 'test']
    total_results = {'lambda': [], 'acc_train': [], 'acc_test': []}

    if os.path.isfile(csv_file):
        os.remove(csv_file)

    data_dir = os.path.join("data", DATASET)

    if os.path.exists(os.path.join(data_dir, "test")):
        test_dataset = ImageFolderWithPaths(os.path.join(data_dir, 'test'), transform=data_transforms["test"])
    if opt.test_split != "":
        test_dataset = ImageFolderWithPaths(opt.test_split, transform=data_transforms["test"])

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    class_names = test_dataset.classes

    for m_phase in model_phases:
        for class_name in class_names:
            total_results['precision_' + m_phase + "_" + class_name] = []
            total_results['recall_' + m_phase + "_" + class_name] = []
            total_results['f1_' + m_phase + "_" + class_name] = []

    # Evaluate model per lambda
    for lambda_idx, current_lambda in enumerate(lambda_values):
        if current_lambda >= 1:
            current_lambda = int(current_lambda)
        if current_lambda >= 1000:
            print("CURRENT LAMBDA = {:.1E}".format(current_lambda))
        else:
            print("CURRENT LAMBDA = {}".format(current_lambda))

        print(data_dir)
        print(class_names)

        partial_results = {}    # Save metrics per fold
        for m_phase in model_phases:
            partial_results['acc_' + m_phase] = []
            for class_name in class_names:
                partial_results['precision_' + m_phase + "_" + class_name] = []
                partial_results['recall_' + m_phase + "_" + class_name] = []
                partial_results['f1_' + m_phase + "_" + class_name] = []

        # Change last layer of classifier to N classes
        num_ftrs = classifier.fc.in_features
        classifier.fc = nn.Linear(num_ftrs, len(class_names))

        classifier = classifier.to(device)

        weights_path = "weights/{}_{}_lambda{}_{}/".format(DATASET, CLASSIFIER_NAME, current_lambda, DATA_SIZE)
        for fold_idx in range(0, KFOLD):
            print("FOLD {}".format(fold_idx))
            print(weights_path)
            # Load weights for classifier (best = highest val acc model, checkpoint = lowest loss acc model)
            if KFOLD == parser.get_default('kfold'):
                classifier.load_state_dict(torch.load("{}{}_netD.pth".format(weights_path,opt.weight_type.lower())))
            else:
                classifier.load_state_dict(torch.load("{}{}_{}_netD.pth".format(weights_path,opt.weight_type.lower(), fold_idx)))

            # Load weights for array of generators (best = highest val acc model, checkpoint = lowest loss acc model)
            G_dict = {}
            for class_name in class_names:
                netG = Generator()
                G_dict[class_name] = netG.cuda()
                if KFOLD == parser.get_default('kfold'):
                    G_dict[class_name].load_state_dict(torch.load("{}{}_netG_{}.pth".format(weights_path,opt.weight_type.lower(), class_name)))
                else:
                    G_dict[class_name].load_state_dict(torch.load("{}{}_{}_netG_{}.pth".format(weights_path,opt.weight_type.lower(), fold_idx, class_name)))
                G_dict[class_name].eval()

            classifier.eval()

            print("TEST")
            phase = "test"
            y_true = []
            y_pred = []
            conf_m = cm.ConfusionMeter(len(class_names))    # Confusion matrix

            with torch.no_grad():
                for i, (inputs, labels, name, _) in enumerate(test_dataloader):
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    classifier_outputs = []
                    for idx, class_name in enumerate(class_names):
                        tr_input = G_dict[class_name](inputs)
                        out = classifier(tr_input)
                        classifier_outputs.append(out)

                    total_outputs = torch.cat(classifier_outputs, 1).squeeze()
                    max_val, idx_max = torch.max(total_outputs, 0)
                    predicted = idx_max % len(class_names)

                    y_true.append(labels.item())
                    y_pred.append(predicted.item())
                    conf_m.add(predicted.unsqueeze(0), labels)
            print(conf_m.conf)
            print("")
            print(metrics.classification_report(y_true, y_pred, digits=4, target_names=class_names))
            df = pd.DataFrame(metrics.classification_report(y_true, y_pred, digits=4, target_names=class_names
                                                            , output_dict=True)).transpose()

            partial_results['acc_' + phase].append(metrics.accuracy_score(y_true, y_pred))
            for idx, class_name in enumerate(class_names):
                partial_results['precision_' + phase + "_" + class_name].append(df['precision'].to_numpy()[idx])
                partial_results['recall_' + phase + "_" + class_name].append(df['recall'].to_numpy()[idx])
                partial_results['f1_' + phase + "_" + class_name].append(df['f1-score'].to_numpy()[idx])

        for m_phase in model_phases:
            partial_results['acc_' + m_phase] = [np.mean(np.asarray(partial_results['acc_' + m_phase]))]
            for class_name in class_names:
                partial_results['precision_' + m_phase + "_" + class_name] = [np.mean(np.asarray(partial_results['precision_' + m_phase + "_" + class_name]))]
                partial_results['recall_' + m_phase + "_" + class_name] = [np.mean(np.asarray(partial_results['recall_' + m_phase + "_" + class_name]))]
                partial_results['f1_' + m_phase + "_" + class_name] = [np.mean(np.asarray(partial_results['f1_' + m_phase + "_" + class_name]))]

        total_results['lambda'].append(current_lambda)
        for m_phase in model_phases:
            total_results['acc_' + m_phase].append(partial_results['acc_' + m_phase][0])
            for class_name in class_names:
                total_results['precision_' + m_phase + "_" + class_name].append(partial_results['precision_' + m_phase + "_" + class_name][0])
                total_results['recall_' + m_phase + "_" + class_name].append(partial_results['recall_' + m_phase + "_" + class_name][0])
                total_results['f1_' + m_phase + "_" + class_name].append(partial_results['f1_' + m_phase + "_" + class_name][0])

        print("Test_acc = {}".format(partial_results['acc_test'][0]))

    df_results = pd.DataFrame(data=total_results)
    print(df_results.to_string())

    df_results.to_csv(csv_file, index=False)


if __name__ == "__main__":
    predict()
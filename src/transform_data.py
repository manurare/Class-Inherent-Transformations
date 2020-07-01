import argparse
import os

import torch.utils.data
from torchvision import datasets, models, transforms
from data_utils import *
from torch.utils.data import DataLoader
import shutil
from model import Generator
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--data_size', required=True, type=int)
parser.add_argument('--dataset', type=str, default="cat_vs_dog")
parser.add_argument('--lambda_class', default=-1.0, type=float)
parser.add_argument('--all_lambdas', default=False, action='store_true')
parser.add_argument('--classifier_name', default="resnet50", type=str)
parser.add_argument('--weight_type', type=str, default="checkpoint")
parser.add_argument('--weight_idx', type=int, default=-1)
parser.add_argument('--test_dir', type=str, default="")

opt = parser.parse_args()

LAMBDA_CLASS = opt.lambda_class
CLASSIFIER_NAME = opt.classifier_name.lower()
ALL_LAMBDAS = opt.all_lambdas
DATASET = opt.dataset.lower()
DATA_SIZE = opt.data_size

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(device.index))
classifier_name = CLASSIFIER_NAME
print(classifier_name)

data_dir = os.path.join('data', DATASET)
if opt.test_dir != "":
    data_dir = opt.test_dir

image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=False, num_workers=4)
               for x in ['train', 'test']}
class_names = image_datasets['train'].classes

if ALL_LAMBDAS:
    lambda_values = [1, 0.5, 0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.00075, 5e-4]
else:
    lambda_values = [LAMBDA_CLASS]

G_dict = {}
for current_lambda in lambda_values:
    if current_lambda >= 1:
        current_lambda = int(current_lambda)
    load_weights_path = 'weights/{}_{}_lambda{}_{}/'.format(DATASET, CLASSIFIER_NAME, current_lambda, DATA_SIZE)
    print(load_weights_path)

    for class_name in class_names:
        G_dict[class_name] = Generator()
        if opt.weight_idx == parser.get_default('weight_idx'):
            G_dict[class_name].load_state_dict(torch.load("{}{}_netG_{}.pth".format(load_weights_path,
                                                                                    opt.weight_type.lower(),
                                                                                    class_name)))
        else:
            G_dict[class_name].load_state_dict(torch.load("{}{}_{}_netG_{}.pth".format(load_weights_path,
                                                                                       opt.weight_type.lower(),
                                                                                       opt.weight_idx, class_name)))
        if torch.cuda.is_available():
            G_dict[class_name].to(device)
        G_dict[class_name].eval()

    for idx, name_dataloader in enumerate(dataloaders):
        path = 'data/transformed_versions/{}_{}_lambda{}_{}'.format(DATASET, CLASSIFIER_NAME, current_lambda, DATA_SIZE)

        print(path)
        print("")

        if os.path.exists(path) and idx == 0:
            shutil.rmtree(path)
            os.makedirs(path)
        elif not os.path.exists(path):
            os.makedirs(path)

        for lr_image, label, tuple_name, tuple_folder in dataloaders[name_dataloader]:
            folder = tuple_folder[0]
            name = tuple_name[0]
            if torch.cuda.is_available():
                lr_image = lr_image.to(device)
            save_images(G_dict, lr_image, label, class_names, path, name, CLASSIFIER_NAME)
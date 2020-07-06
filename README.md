# Class-Inherent Transformations
Paper: [FuCiTNet: Improving the generalization of deep learning networks by the fusion of learned class-inherent transformations](https://arxiv.org/abs/2005.08235)

## Requirements
This code was tested using python 3.6.9, cuda 10 and PyTorch 1.1.0

Run the following line ` pip3 install -r ./requirements.txt` to install the requirements.

## Data
Data must present the following layout

```
dataset/
    train/
        class1/
            img1.jpg
            ...
        class2/
            imga.jpg
            ...
    test/
        class1/
            img3.jpg
            ...
        class2/
            imgc.jpg
            ...
```
Datasets used in the paper are available within the folder `data`
## Training
In order to train the model:
`python3 src/train.py --all_lambdas --classifier_name resnet18 --dataset church_vs_palace --data_size 64`

## Testing
Trained models on "church vs palace", "cat vs dog" and "cat vs dog vs goldfish" are provided in `weights` (only the best lambdas reported in the paper):
- 0.05 for cat_vs_dog
- 0.005 for cat_vs_dog_vs_goldfish
- 0.0025 for church_vs_palace

To run the test script:
`python3 src/test.py --lambda_class 0.05 --weight_type checkpoint --classifier_name resnet18 --kfold 3 --dataset cat_vs_dog --data_size 64`

## Citation
If you use our code, please cite our work
```
@article{REYAREA2020188,
title = {{FuCiTNet}: Improving the generalization of deep learning networks by the fusion of learned class-inherent transformations},
journal = {Information Fusion},
volume = {63},
pages = {188 - 195},
year = {2020},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2020.06.015},
url = {http://www.sciencedirect.com/science/article/pii/S1566253520303122},
author = {Manuel Rey-Area and Emilio Guirado and Siham Tabik and Javier Ruiz-Hidalgo}
}
```

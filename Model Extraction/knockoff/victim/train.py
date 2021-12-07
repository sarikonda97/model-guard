#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
from datetime import datetime
import json
from collections import defaultdict as dd
import numpy as np

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as torch_data
from torch.utils.data import Dataset, DataLoader, Subset

import knockoff.config as cfg
from knockoff import datasets
import knockoff.utils.transforms as transform_utils
import knockoff.utils.model_victim as model_utils
import knockoff.utils.utils as knockoff_utils
import knockoff.models.zoo as zoo
from knockoff.datasets.cifarlike import CIFAR10


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('dataset', metavar='DS_NAME', type=str, help='Dataset name')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    parser.add_argument('image_obfs', metavar='OBFS_TCH', type=str, help='Image obfuscation technique')
    parser.add_argument('image_obfs_magnitude', metavar='OBFS_MAG', type=float, help='obfuscation magnitude')
    # Optional arguments
    parser.add_argument('-o', '--out_path', metavar='PATH', type=str, help='Output path for model',
                        default=cfg.MODEL_DIR)
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr-step', type=int, default=30, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--train_subset', type=int, help='Use a subset of train set', default=None)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=None)
    args = parser.parse_args()
    params = vars(args)

    # torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Set up dataset
    dataset_name = params['dataset']
    valid_datasets = datasets.__dict__.keys()
    if dataset_name not in valid_datasets: 
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]

    print("The dataset Name is:", dataset)

    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    image_obfs_tcq = params['image_obfs']
    image_obfs_mag = params['image_obfs_magnitude']
    trainset = dataset(train=True, transform=train_transform, img_tcq=image_obfs_tcq, img_mag=image_obfs_mag)
    testset = dataset(train=False, transform=test_transform, img_tcq=image_obfs_tcq, img_mag=image_obfs_mag)

    print(f"The obfuscation technique is {image_obfs_tcq} and magnitude {image_obfs_mag}")

    print("The trainset size is:", len(trainset))
    print("The trainset size is:", len(testset))

    # for i in range(len(trainset)):
    #     sample = trainset[i]

    #     ax = plt.subplot(1, 4, i + 1)
    #     plt.tight_layout()
    #     ax.set_title('Sample #{}'.format(i))
    #     ax.axis('off')
    #     plt.imshow(sample[0])

    #     if i == 3:
    #         plt.savefig(f'/home/mohan235/projects/def-guzdial/mohan235/CMPUT622_project/knockoffnets/trainset-{image_obfs_tcq}_{image_obfs_mag}.jpg')
    #         print("Trainset sample saved!")
    #         break
    

    # for i in range(len(testset)):
    #     sample = testset[i]

    #     ax = plt.subplot(1, 4, i + 1)
    #     plt.tight_layout()
    #     ax.set_title('Sample #{}'.format(i))
    #     ax.axis('off')
    #     plt.imshow(sample[0])

    #     if i == 3:
    #         plt.savefig(f'/home/mohan235/projects/def-guzdial/mohan235/CMPUT622_project/knockoffnets/testset-{image_obfs_tcq}_{image_obfs_mag}.jpg')
    #         print("Testset Sample saved!")
    #         break
    

    num_classes = len(trainset.classes)
    params['num_classes'] = num_classes

    if params['train_subset'] is not None:
        idxs = np.arange(len(trainset))
        ntrainsubset = params['train_subset']
        idxs = np.random.choice(idxs, size=ntrainsubset, replace=False)
        trainset = Subset(trainset, idxs)

    # ----------- Set up model
    model_name = params['model_arch']
    pretrained = params['pretrained']
    # model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained)
    model = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes)
    model = model.to(device)

    # ----------- Train
    out_path = params['out_path']
    
    print(f"The image obfuscation techniques is {image_obfs_tcq} with magnitude {image_obfs_mag}")

    
    # print("The train set is ",trainset[1][0], trainset[1][1])

    # for index in range(len(trainset)):
    #     trainset[index]= list(trainset[index])
    #     converted_image = torch.from_numpy(cv2.GaussianBlur(trainset[index][0].numpy(),(image_obfs_mag,image_obfs_mag),cv2.BORDER_DEFAULT))
    #     trainset[index][0] = converted_image
        # setattr(CIFAR10, trainset[index], tuple(trainset[index]))
        # print("The data is:",data, "and the label is:", label)
    
    # print("The input tensor is:", trainset)
    # print("The converted image is:", converted_image)
    # print("The obfuscation is over!")
    model_utils.train_model(model, trainset, testset=testset, device=device, **params)

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, 'params.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()

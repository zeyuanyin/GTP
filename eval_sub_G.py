'''
Evaluation for 10-targets subsource setting as discussed in our paper.
For each target, we have 450 samples of the other classes.
'''

import argparse
import os

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torchvision.models as models
from generators import GeneratorResnet
from gaussian_smoothing import *

# Purifier
from NRP import *

import logging

from utils import SubImageFolder
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Targeted Transferable Perturbations')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size for evaluation')
parser.add_argument('--eps', type=int, default=16, help='Perturbation Budget')
parser.add_argument('--target_model', type=str, default='vgg19_bn', help='Black-Box(unknown) model: SIN, Augmix etc')
parser.add_argument('--source_model', type=str, default='res50', help='TTP Discriminator: \
{res18, res50, res101, res152, dense121, dense161, dense169, dense201,\
 vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn,\
 ens_vgg16_vgg19_vgg11_vgg13_all_bn,\
 ens_res18_res50_res101_res152\
 ens_dense121_161_169_201}')
parser.add_argument('--source_domain', type=str, default='IN', help='Source Domain (TTP): Natural Images (IN) or painting')
# For purification (https://github.com/Muzammal-Naseer/NRP)
parser.add_argument('--NRP', action='store_true', help='Apply Neural Purification to reduce adversarial effect')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
parser.add_argument('--testset_path', type=str, default='/home/zeyuan.yin/imagenet/val', help='Path to testset')
parser.add_argument('--entire_round', type=bool, default=False, help='eval 10 generators, default is False to eval one generator w.r.t. one target class')
parser.add_argument('--target_class', type=int, default=3, help='Target data class')

args = parser.parse_args()
print(args)

def load_source_set(targets, target_class):
    class_values = targets
    del class_values[target_class]
    assert len(class_values) == 9

    scale_size = 256
    img_size = 224
    data_transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    source_set =  SubImageFolder(args.testset_path, class_values=class_values, transform=data_transform)
    return source_set

def normalize(t):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

    return t

def main():
    # GPU
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Perturbation Budget
    args.eps = args.eps/255.0

    # Set-up Kernel
    kernel_size = 3
    pad = 2
    sigma = 1
    kernel = get_gaussian_kernel(kernel_size=kernel_size, pad=pad, sigma=sigma).cuda()


    # Load Discriminator/Attacked Model
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    if args.target_model in model_names:
        model = models.__dict__[args.target_model](pretrained=True)
    elif args.target_model == 'SIN':
        model = torchvision.models.resnet50(pretrained=False)
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load('pretrained_models/resnet50_train_60_epochs-c8e5653e.pth.tar')
        model.load_state_dict(checkpoint["state_dict"])
    elif args.target_model == 'Augmix':
        model = torchvision.models.resnet50(pretrained=False)
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load('pretrained_models/checkpoint.pth.tar')
        model.load_state_dict(checkpoint["state_dict"])
    else:
        assert (args.target_model in model_names), 'Please provide correct target model names: {}'.format(model_names)

    model = model.to(args.device)
    model.eval()

    if args.NRP:
        purifier = NRP(3, 3, 64, 23)
        purifier.load_state_dict(torch.load('pretrained_purifiers/NRP.pth'))
        purifier = purifier.to(args.device)

    all_targets = [24,99,245,344,471,555,661,701,802,919]

    total_acc = 0
    total_distance = 0

    if args.entire_round:
        eval_targets = all_targets
    else:
        eval_targets = [args.target_class]

    for target in eval_targets:
        # Data
        test_set = load_source_set(all_targets, target_class=target)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        assert len(test_set) == 450
        print('Test data size:', len(test_set))

        # Load Generator
        netG = GeneratorResnet()
        netG.load_state_dict(torch.load('pretrained_generators/netG_{}_{}_19_{}.pth'.format(args.source_model,args.source_domain, target)))
        netG = netG.to(args.device)
        netG.eval()


        acc, distance = eval(netG, model, test_loader, target, kernel, purifier)
        total_acc += acc
        total_distance += distance

    print('*'*100)
    print('Average Target Transferability')
    print('*'*100)
    print(' %d \t\t %.4f\t \t %.4f', int(args.eps * 255), total_acc / len(eval_targets), total_distance / len(eval_targets))

def eval(netG, model, test_loader, target, kernel, purifier):
    print('Epsilon \t Target \t Acc. \t Distance')

    # Reset Metrics
    acc = 0
    distance = 0
    for i, (img, label) in enumerate(test_loader):
        img, label = img.to(args.device), label.to(args.device)

        target_label = torch.LongTensor(img.size(0))
        target_label.fill_(target)
        target_label = target_label.to(args.device)

        adv = kernel(netG(img)).detach()
        # Projection
        adv = torch.min(torch.max(adv, img - args.eps), img + args.eps)
        adv = torch.clamp(adv, 0.0, 1.0)

        if args.NRP:
            # Purify Adversary
            adv = purifier(adv).detach()

        out = model(normalize(adv.clone().detach()))
        acc += torch.sum(out.argmax(dim=-1) == target_label).item()

        distance +=(img - adv).max() *255

        avg_acc = acc / len(test_loader.dataset)
        avg_distance = distance / (i + 1)
    print(' %d \t\t %d\t  %.4f\t \t %.4f',int(args.eps * 255), target, avg_acc, avg_distance)
    return avg_acc, avg_distance

if __name__ == '__main__':
    main()
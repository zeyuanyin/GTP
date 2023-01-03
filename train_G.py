# This code is modified from https://github.com/Muzammal-Naseer/TTP

import argparse
import os
import numpy as np

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

from generators import *
from gaussian_smoothing import *

from utils import SubImageFolder, TwoCropTransform, rotation

from tqdm import tqdm


parser = argparse.ArgumentParser(description='Transferable Targeted Perturbations')
parser.add_argument('--batch_size', type=int, default=20, help='Number of trainig samples/batch')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate for adam')
parser.add_argument('--eps', type=int, default=16, help='Perturbation Budget during training, eps')
parser.add_argument('--model_type', type=str, default='resnet50',
                    help='Model under attack (discrimnator)')
parser.add_argument('--gs', action='store_true', help='Apply gaussian smoothing')
parser.add_argument('--save_dir', type=str, default='pretrained_generators_3', help='Directory to save generators')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
parser.add_argument('--target_class', type=int, default=3, help='Target data class')
parser.add_argument('--entire_round', type=bool, default=False, help='Train 10 generators in 10 rounds, default is False to train one generator w.r.t. one target class')
parser.add_argument('--trainset_path', type=str, default='/home/zeyuan.yin/imagenet/train_50', help='path to imagenet train_50')
parser.add_argument('--targetset_path', type=str, default='/home/zeyuan.yin/imagenet/train', help='path to imagenet train')


def main():
    args = parser.parse_args()
    print(args)

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    args.eps = args.eps / 255

    # GPU
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Discriminator
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    if args.model_type in model_names:
        netD = models.__dict__[args.model_type](pretrained=True)
    else:
        assert (args.model_type in model_names), 'Please provide correct discriminator model names: {}'.format(model_names)

    netD = netD.to(args.device)
    netD.eval()

    target_dict = {
        3: 'n01491361',
        16: 'n01560419',
        24: 'n01622779',
        36: 'n01667778',
        48: 'n01695060',
        52: 'n01728572',
        69: 'n01768244',
        71: 'n01770393',
        85: 'n01806567',
        99: 'n01855672'
    }
    targets = [24,99,245,344,471,555,661,701,802,919]

    if args.entire_round: # train 10 generators in 10 rounds
        train_targets = targets
    else:
        train_targets = [args.target_class]

    for target_class in train_targets:
        main_worker(args, netD, target_class)


def load_source_and_target_set(args, target_class):

    class_values = [i for i in range(1000)]
    del class_values[target_class]
    assert len(class_values) == 999

    # Input dimensions
    if args.model_type == 'inception_v3':
        scale_size = 300
        img_size = 299
    else:
        scale_size = 256
        img_size = 224

    # Data
    train_transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()])

    source_set =  SubImageFolder(args.trainset_path, class_values=class_values, transform=TwoCropTransform(train_transform, img_size))
    target_set = SubImageFolder(args.targetset_path, class_values=[target_class,], transform=train_transform)
    return source_set, target_set


def main_worker(args, netD, scale_size, img_size, target_class):
    # Data
    source_set, target_set = load_source_and_target_set(args, target_class)
    train_loader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print('Training (Source) data size:', len(source_set))

    if len(target_set) < 1300:
        target_set.samples = target_set.samples[0:1000]
    target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    assert len(target_set) == 1000 or len(target_set) == 1300
    print('Training (Match) data size:', len(target_set))


    # Generator
    if args.model_type == 'inception_v3':
        netG = GeneratorResnet(inception=True)
    else:
        netG = GeneratorResnet()
    netG.to(args.device)

    # Optimizer
    optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Gaussian Smoothing
    if args.gs:
        kernel_size = 3
        pad = 2
        sigma = 1
        kernel = get_gaussian_kernel(kernel_size=kernel_size, pad=pad, sigma=sigma).cuda()

    # Loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    for epoch in tqdm(range(args.epochs)):

        train(train_loader, netD, criterion_kl, optimG, epoch, args, kernel, target_loader)

        torch.save(netG.state_dict(),args.save_dir + '/netG_{}_{}_{}.pth'.format(args.model_type, epoch, target_class))


def normalize(t):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t


def train(train_loader, netD, criterion_kl, optimG, epoch, args, kernel, target_loader):
    running_loss = 0
    for i, (imgs, _) in enumerate(train_loader):
        img = imgs[0].to(args.device)
        img_rot = rotation(img)[0]
        img_aug = imgs[1].to(args.device)

        try:
            img_match = next(dataiter)[0]
        except StopIteration:
            dataiter = iter(target_loader)
            img_match = next(dataiter)[0]
        img_match = img_match.to(args.device)

        netG.train()
        optimG.zero_grad()

        # Unconstrained Adversaries
        adv = netG(img)
        adv_rot = netG(img_rot)
        adv_aug = netG(img_aug)

        # Smoothing
        if args.gs:
            adv = kernel(adv)
            adv_rot = kernel(adv_rot)
            adv_aug = kernel(adv_aug)


        # Projection
        adv = torch.min(torch.max(adv, img - args.eps), img + args.eps)
        adv = torch.clamp(adv, 0.0, 1.0)
        adv_rot = torch.min(torch.max(adv_rot, img_rot - args.eps), img_rot + args.eps)
        adv_rot = torch.clamp(adv_rot, 0.0, 1.0)
        adv_aug = torch.min(torch.max(adv_aug, img_aug - args.eps), img_aug + args.eps)
        adv_aug = torch.clamp(adv_aug, 0.0, 1.0)

        adv_out = netD(normalize(adv))
        adv_rot_out = netD(normalize(adv_rot))
        adv_aug_out = netD(normalize(adv_aug))
        img_match_out = netD(normalize(img_match))


        # Loss
        loss_kl = 0.0
        loss_sim  = 0.0

        for out in [adv_out, adv_rot_out, adv_aug_out]:
            # print(F.log_softmax(out, dim=1).shape,F.softmax(img_match_out, dim=1).shape)
            loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(out, dim=1),
                                                            F.softmax(img_match_out, dim=1))
            loss_kl += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(img_match_out, dim=1),
                                                            F.softmax(out, dim=1))

        # Neighbourhood similarity
        St = torch.matmul(img_match_out,  img_match_out.t())
        norm = torch.matmul(torch.linalg.norm(img_match_out, dim=1, ord=2), torch.linalg.norm(img_match_out, dim=1, ord=2).t())
        St = St/norm
        for out in [adv_rot_out, adv_aug_out]:
            Ss = torch.matmul(adv_out,  out.t())
            norm = torch.matmul(torch.linalg.norm(adv_out, dim=1, ord=2), torch.linalg.norm(out, dim=1, ord=2).t())
            Ss = Ss/norm
            loss_sim += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(Ss, dim=1),
                                                            F.softmax(St, dim=1))
            loss_sim += (1.0 / args.batch_size) * criterion_kl(F.log_softmax(St, dim=1),
                                                            F.softmax(Ss, dim=1))

        loss = loss_kl + loss_sim
        loss.backward()
        optimG.step()
        running_loss += loss.item()

        if i % 10 == 9:
            print('Epoch: {0} \t Batch: {1} \t loss: {2:.5f}'.format(epoch, i, running_loss / 10))
            running_loss = 0


if __name__ == '__main__':
    main()
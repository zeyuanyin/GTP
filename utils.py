# This code is modified from https://github.com/Muzammal-Naseer/TTP

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

# Transformations
class TwoCropTransform:
    def __init__(self, transform, img_size):
        self.transform = transform
        self.img_size = img_size
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.img_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.ToTensor()])

    def __call__(self, x):
        return [self.transform(x), self.data_transforms(x)]

def rotation(input):
    batch = input.shape[0]
    target = torch.tensor(np.random.permutation([0,1,2,3] * (int(batch / 4) + 1)), device = input.device)[:batch]
    target = target.long()
    image = torch.zeros_like(input)
    image.copy_(input)
    for i in range(batch):
        image[i, :, :, :] = torch.rot90(input[i, :, :, :], target[i], [1, 2])

    return image, target


# def get_source_set_and_target_set(root:str,transform:Optional[Callable]=None, target_idx:int=0):
#     source_set = ImageFolder_TargetAttack(root=root, transform=transform, target_idx=target_idx, target=False)
#     target_set = ImageFolder_TargetAttack(root='/home/zeyuan.yin/imagenet/train', transform=transform, target_idx=target_idx, target=True)
#     return source_set, target_set




# class ImageFolder_TargetAttack(ImageFolder):
#     def __init__(
#         self,
#         root: str,
#         transform: Optional[Callable] = None,
#         target_idx:int = 0,
#         target: bool = False
#     ):
#         super().__init__(
#             root,
#             transform)
#         self.target_idx = target_idx
#         IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
#         classes, class_to_idx = self.updata_classes(target)
#         samples = self.make_dataset(self.root, class_to_idx, IMG_EXTENSIONS)

#         self.classes = classes
#         self.class_to_idx = class_to_idx
#         self.samples = samples
#         self.targets = [s[1] for s in samples]

#     def updata_classes(self,target:bool):
#         target_key_value=list(self.class_to_idx.items())[self.target_idx]
#         if target: # return the target class (only one class)
#             return [target_key_value[0]], {target_key_value[0]:target_key_value[1]}

#         else : # return the source class (all classes except the target class)
#             del(self.class_to_idx[target_key_value[0]])
#             del(self.classes[self.target_idx])
#             return self.classes, self.class_to_idx

class ImageNet_TargetAttack(ImageFolder):
    def __init__(
        self,
        # root: str,
        transform: Optional[Callable] = None,
        target_class:int = 0,
        target: bool = False,
        all_train: bool = False
    ):
        self.target_class = target_class
        self.target = target
        self.all_train = all_train
        self.f2l = self.load_labels('/home/zeyuan.yin/adv/TTP/imagenet_lsvrc_2015_synsets.txt')
        if self.target == True:
            root = '/home/zeyuan.yin/imagenet/train'
        else:
            root = '/home/zeyuan.yin/adv/TTP/subset_source/train'

        super().__init__(
            root,
            transform)
        print(len(self.classes))
        print(len(self.class_to_idx))
        print(len(self.samples))

    def load_labels(self, path):
        """
        Load ImageNet labels from file
        """
        with open(path) as f:
            lines = f.readlines()
        return [line.split('\n')[0] for line in lines]

    def find_classes(self, root):
        """
        Override the find_classes function in DatasetFolder
        """
        if self.target == True: # return the target class (only one class)
            return [self.f2l[self.target_class]], {self.f2l[self.target_class]:self.target_class}
        else: # return the source class (all classes except the target class)
            if self.all_train:
                classes = [self.f2l[i] for i in range(len(self.f2l)) if i != self.target_class]
                class_to_idx = {self.f2l[i]:i for i in range(len(self.f2l))}
                del(class_to_idx[self.f2l[self.target_class]])
                return classes, class_to_idx
            else: # 10-class to train, self.target_class should be in dir_names, left 9 classes

                # get all dir name in self.root
                dir_names = os.listdir(self.root)
                classes = [dir_name for dir_name in dir_names if dir_name != self.f2l[self.target_class]]
                class_to_idx = {class_: self.f2l.index(class_) for class_ in classes}
                assert len(classes) == 9
                return classes, class_to_idx



class SubImageFolder(ImageFolder):
    """
    A generic Imagenet Folder data loader which can load images from a list containing the dir names or class values.
    dir names is the first priority when both dir names and class values are provided.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in an PIL image.
        dir_names (list, optional): A list of dir names to be included in the dataset. E.g, ['n01440764', 'n01443537']
        class_values (list, optional): A list of class values to be included in the dataset. E.g, [0, 1]

    """
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        dir_names: List[str] = None,
        class_values: List[str] = None,
        lable_path: str = '/home/zeyuan.yin/adv/TTP/imagenet_lsvrc_2015_synsets.txt'
    ):
        self.dir_names = dir_names
        self.class_values = class_values
        self.f2l = self.load_labels(lable_path)

        super().__init__(
                root,
                transform)

    def load_labels(self, path):
        """
        Load ImageNet labels from file
        """
        with open(path) as f:
            lines = f.readlines()
        return [line.split('\n')[0] for line in lines]

    def find_classes(self, root):
        """
        Override the find_classes function in DatasetFolder
        """
        if self.dir_names is not None:
            classes = self.dir_names
            class_to_idx = {class_: self.f2l.index(class_) for class_ in classes}
            return classes, class_to_idx
        elif self.class_values is not None:
            class_to_idx = {self.f2l[class_value]: class_value for class_value in self.class_values}
            classes = list(class_to_idx.keys())
            return classes, class_to_idx
        else:
            raise ValueError("dir_names and class_values cannot be None at the same time")



if __name__ == '__main__':

    # source_set, target_set = get_source_set_and_target_set(root='/home/zeyuan.yin/adv/TTP/dataset/trainset', target_idx=4)

    # source_set = ImageNet_TargetAttack(target_class=3, target=False, all_train=False)
    # target_set = ImageNet_TargetAttack(target_class=3, target=True, all_train=False)

# 69: 'n01768244',
#         71: 'n01770393',

    set = SubImageFolder(root='/home/zeyuan.yin/imagenet/val', dir_names=['n01768244', 'n01770393',])
    print(set.classes)
    print(set.class_to_idx)
    print(set.targets)
    print(len(set.samples))
    set_2= SubImageFolder(root='/home/zeyuan.yin/imagenet/val', class_values=[69,])
    print(set_2.classes)
    print(set_2.class_to_idx)
    print(set_2.targets)
    print(len(set_2.samples))

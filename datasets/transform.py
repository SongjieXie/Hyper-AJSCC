import torchvision.transforms as transforms
from torchvision.transforms.transforms import Resize
import random

def recon_transform():
    return transforms.Compose(
        [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=5, pad_if_needed=True, fill=0, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
def recon_transform_test():
    return transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def simple_transform_mnist():
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

def simple_transform(s): 
    return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
            ])
def simple_transform_test(s):
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
            ])



def imagenet_transform(s):
    return transforms.Compose([
            transforms.Resize(s),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

def imagenet_transform_aug(s):
    return transforms.Compose([
            transforms.Resize(72),
            transforms.RandomResizedCrop(s),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.4,.4,.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

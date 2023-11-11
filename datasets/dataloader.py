import torch
import torchvision.datasets as dsets
import os

from .transform import simple_transform_mnist, simple_transform, imagenet_transform, \
    simple_transform_test, imagenet_transform_aug, recon_transform, recon_transform_test

root = r'/home/sxieat/data/'
def get_data(data_set, batch_size, shuffle=True, n_worker=0, train = True, add_noise=0, model_type='recon'):   
    if data_set == 'CIFAR10':
        if train:
            tran = recon_transform() if model_type == 'recon' else simple_transform(32)
        else:
            tran = recon_transform_test() if model_type == 'recon' else simple_transform_test(32)
            
        dataset = dsets.CIFAR10(root+'CIFAR10/', train=train, transform=tran, target_transform=None, download=False)
        
    else:
        print('Sorry! Cannot support ...')
        
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_worker)
    return dataloader

    


        

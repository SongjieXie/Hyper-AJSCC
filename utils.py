import torch
import torch.nn as nn
import numpy as np 

def sample(d_range: list, num_pts: int, sampling: str):
    if sampling == 'uniform':
        return (d_range[1]-d_range[0])*torch.rand(num_pts, 1) + d_range[0]
    elif sampling == 'discrete':
        return torch.randint(1, 21, size=(num_pts, 1)).float()
    else:
        pl = torch.empty(num_pts, 1)
        return nn.init.trunc_normal_(pl, 10, float(sampling), 0, 20)

def snr2logd(x):
    return torch.log(torch.tensor(10))*(-x/20)

def mse2psnr(x):
    return 10*np.log10(4/x)


def psnr(img_true, img_fake):
    img_gen_numpy = img_fake.detach().cpu().float().numpy()
    # img_gen_numpy = (np.transpose(img_gen_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    img_gen_numpy = (img_gen_numpy+1)/2.0 * 255.0
    # img_gen_numpy = img_gen_numpy * 255.0
    img_gen_int8 = img_gen_numpy.astype(np.uint8)

    origin_numpy = img_true.detach().cpu().float().numpy()
    # origin_numpy = (np.transpose(origin_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    origin_numpy = (origin_numpy+1)/2.0 * 255.0
    origin_int8 = origin_numpy.astype(np.uint8)

    diff = np.mean((np.float64(img_gen_int8) - np.float64(origin_int8))**2)

    PSNR = 10 * np.log10((255**2) / diff)
    return PSNR

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    


if __name__ == "__main__":
    # x = torch.randn(128, 3, 200, 200)
    # num_pts = x.shape[0]
    # y = sample([11.1, 11.1], num_pts)
    # print(y)
    # x_hat = AWGN(x, y)
    # print(x_hat.shape)
    print(snr2logd(-1))
    print(snr2logd(0))
    print(snr2logd(10))
    print(snr2logd(15))
    print(snr2logd(20))
    print('===================')
    print(torch.exp(snr2logd(-1)))
    print(torch.exp(snr2logd(0)))
    print(torch.exp(snr2logd(10)))
    print(torch.exp(snr2logd(15)))
    print(torch.exp(snr2logd(20)))
    

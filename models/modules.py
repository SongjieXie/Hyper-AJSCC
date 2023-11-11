from torch import nn
import torch
    
class ActivationLayer(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.features = features
        self.hyper_block_scale = nn.Linear(1, self.features, bias=True)
        
        # self.init_params()
    
    def forward(self, inputs: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        scale = self.hyper_block_scale(betas) if len(inputs.shape) <=3 else self.hyper_block_scale(betas).unsqueeze(-1).unsqueeze(-1)
        return scale * inputs
    
    def init_params(self):
        nn.init.xavier_uniform_(self.hyper_block_scale.weight)


class ActivatedLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False) -> None:
       super().__init__()
       self.pre = nn.Linear(in_channels, out_channels, bias=bias)
       self.act = ActivationLayer(1)
    #    self.act = ActivationLayer(out_channels)
       
       self.is_bias = bias
       
       self.act.init_params()
       
       self.init_params()
       
    
    def forward(self, x, betas):
        return self.act(self.pre(x), betas)
    
    def init_params(self):
        nn.init.xavier_uniform_(self.pre.weight)
        if self.is_bias:
            nn.init.xavier_uniform_(self.pre.bias)
    
class ActivatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True) -> None:
       super().__init__()
       self.pre = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                            stride=stride, padding=padding, bias=bias)
       self.act = ActivationLayer(out_channels)
       
    def forward(self, x, betas):
        return self.act(self.pre(x), betas)
    
class ActivatedDeconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, output_padding=0, bias=True) -> None:
       super().__init__()
       self.pre = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, output_padding=output_padding, bias=bias)
       self.act = ActivationLayer(out_channels)
       
    def forward(self, x, betas):
        return self.act(self.pre(x), betas)
    
class ActivatedConv1d(nn.Module):
    pass

class PreActActivatedResblock2d(nn.Module):
    def __init__(self, in_channel) -> None:
       super().__init__()
       self.norm_1 = nn.BatchNorm2d_1(in_channel)
       self.norm_2 = nn.BatchNorm2d_2(in_channel)
       self.act_1 = nn.ReLU()
       self.act_2 = nn.ReLU()
       self.activatedConv2d_1 = ActivatedConv2d(in_channel, in_channel, 3, 1, 1, bias=True)
       self.activatedConv2d_2 = ActivatedConv2d(in_channel, in_channel, 3, 1, 1, bias=True)
       
    def forward(self, x, betas):
        y = self.act_1(self.norm_1(x))
        y = self.activatedConv2d_1(y, betas)
        y = self.act_2(self.norm_2(y))
        y = self.activatedConv2d_2(y, betas)
        return x + y 
    
class ActivatedResblock2d(nn.Module):
    def __init__(self, in_channel) -> None:
       super().__init__()
       self.norm_1 = nn.BatchNorm2d(in_channel)
       self.norm_2 = nn.BatchNorm2d(in_channel)
       self.act_1 = nn.ReLU()
       self.act_2 = nn.ReLU()
       self.activatedConv2d_1 = ActivatedConv2d(in_channel, in_channel, 3, 1, 1, bias=False)
       self.activatedConv2d_2 = ActivatedConv2d(in_channel, in_channel, 3, 1, 1, bias=False)
       
    def forward(self, x, betas):
        y = self.activatedConv2d_1(x, betas)
        y = self.act_1(self.norm_1(y))
        y = self.activatedConv2d_2(y, betas)
        y = self.act_2(self.norm_2(y))
        return x + y 
    
class PreActResblock2d(nn.Module):
    def __init__(self, in_channel) -> None:
       super().__init__()
       self.norm_1 = nn.BatchNorm2d(in_channel)
       self.norm_2 = nn.BatchNorm2d(in_channel)
       self.act_1 = nn.ReLU()
       self.act_2 = nn.ReLU()
       self.Conv2d_1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True)
       self.Conv2d_2 = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True)
       
    def forward(self, x):
        y = self.act_1(self.norm_1(x))
        y = self.Conv2d_1(y)
        y = self.act_2(self.norm_2(y))
        y = self.Conv2d_2(y)
        return x + y 
    
class Resblock2d(nn.Module):
    def __init__(self, in_channel) -> None:
       super().__init__()
       self.norm_1 = nn.BatchNorm2d(in_channel)
       self.norm_2 = nn.BatchNorm2d(in_channel)
       self.act_1 = nn.ReLU()
       self.act_2 = nn.ReLU()
       self.Conv2d_1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=False)
       self.Conv2d_2 = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=False)
       
    def forward(self, x):
        y = self.Conv2d_1(x)
        y = self.act_1(self.norm_1(y))
        y = self.Conv2d_2(y)
        y = self.act_2(self.norm_2(y))
        return x + y 
    
class AFLayer(nn.Module):
    def __init__(self, channels) -> None:
       super().__init__()
       self.pool = nn.AdaptiveAvgPool2d((1,1))
       self.dense = nn.Sequential(
           nn.Linear(channels+1, channels//16),
           nn.ReLU(True),
           nn.Linear(channels//16, channels),
           nn.Sigmoid()
       )
    
    def forward(self, x, snr):
        y = self.pool(x)
        y = torch.squeeze(torch.squeeze(y, -1), -1)
        y = torch.cat((y, snr), -1)
        y = self.dense(y).unsqueeze(-1).unsqueeze(-1)
        return x*y

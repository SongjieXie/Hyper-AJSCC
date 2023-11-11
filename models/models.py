import torch
import torch.nn as nn
from .modules import ActivatedLinear, ActivatedConv2d, ActivatedDeconv2d, ActivatedConv1d, Resblock2d, AFLayer, ActivatedResblock2d
from .pytorch_gdn import GDN


 
class HyperAJSCC_encoder(nn.Module):
    def __init__(self, in_channels, symbol_channels, device) -> None:
        super().__init__()        
        self.conv_1 = ActivatedConv2d(in_channels, 256, 9, 2, 4)
        self.act_1 = nn.Sequential(
            GDN(256, device),
            nn.PReLU()
        )
        self.conv_2 = ActivatedConv2d(256, 256, 5, 2, 2)
        self.act_2 = nn.Sequential(
            GDN(256, device),
            nn.PReLU(),
        )
        self.conv_3 = ActivatedConv2d(256, 256, 5, 1, 2)
        self.act_3 = nn.Sequential(
            GDN(256, device),
            nn.PReLU()
        )
        self.conv_4 = ActivatedConv2d(256, 256, 5, 1, 2)
        self.act_4 = nn.Sequential(
            GDN(256, device),
            nn.PReLU(),
        )
        self.conv_5 = ActivatedConv2d(256, symbol_channels, 5, 1, 2)
        self.act_5 = nn.Sequential(
            GDN(symbol_channels, device)
        )
    def forward(self, x, snr):
        y = self.act_1(self.conv_1(x, snr))
        y = self.act_2(self.conv_2(y, snr))
        y = self.act_3(self.conv_3(y, snr))
        y = self.act_4(self.conv_4(y, snr))
        return self.act_5(self.conv_5(y, snr))
            
class HyperAJSCC_decoder(nn.Module):
    def __init__(self, symbol_channels, out_channels, device) -> None:
        super().__init__()
        self.deconv_1 = ActivatedDeconv2d(symbol_channels, 256, 5, 1, 2)
        self.act_1 = nn.Sequential(
            GDN(256, device, inverse=True),
            nn.PReLU()
        )
        self.deconv_2 = ActivatedDeconv2d(256, 256, 5, 1, 2)
        self.act_2 = nn.Sequential(
            GDN(256, device, inverse=True),
            nn.PReLU(),
        )
        self.deconv_3 = ActivatedDeconv2d(256, 256, 5, 1, 2)
        self.act_3 = nn.Sequential(
            GDN(256, device, inverse=True),
            nn.PReLU(),
        )
        self.deconv_4 = ActivatedDeconv2d(256, 256, 5, 2, 2, output_padding=1)
        self.act_4 = nn.Sequential(
            GDN(256, device, inverse=True),
            nn.PReLU(),
        )
        self.deconv_5 = ActivatedDeconv2d(256, out_channels, 9, 2, 4, output_padding=1)
        self.act_5 = nn.Sequential(
            GDN(out_channels, device, inverse=True),
            nn.Tanh()
        )
    def forward(self, x, snr):
        y = self.act_1(self.deconv_1(x, snr))
        y = self.act_2(self.deconv_2(y, snr))
        y = self.act_3(self.deconv_3(y, snr))
        y = self.act_4(self.deconv_4(y, snr))
        return self.act_5(self.deconv_5(y, snr)) 
      
class T_HyperAJSCC_encoder(nn.Module):
    
    def __init__(self, in_channels, symbol_channels, device):
        super().__init__()
        
        self.prep = ActivatedConv2d(3,64,kernel_size = 3,stride = 1, padding = 1, bias = False)
        self.prep_act = nn.Sequential(
                    nn.BatchNorm2d(64),
                    nn.ReLU()
                    ) # 64x32x32
        self.layer1 = ActivatedConv2d(64,128,kernel_size = 3,stride = 1, padding = 1, bias = False)
        self.layer1_act = nn.Sequential(
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False)
                    )# 128x16x16
        self.layer1_res = ActivatedResblock2d(128)
        self.layer1_res_act = nn.Sequential(
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    )# 128x16x16
        self.layer2 = ActivatedConv2d(128,256,kernel_size = 3,stride = 1, padding = 1, bias = False)
        self.layer2_act = nn.Sequential(
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2)
                    )# 256x8x8
        self.layer3 = ActivatedConv2d(256,512,kernel_size = 3,stride = 1, padding = 1, bias = False)
        self.layer3_act = nn.Sequential(
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False)
                    )# 512x4x4
        self.encoder1 = ActivatedConv2d(512,4,kernel_size = 3,stride = 1, padding = 1, bias = False)
        self.encoder1_act = nn.Sequential(
                        nn.BatchNorm2d(4),
                        nn.ReLU(),
                        nn.Flatten()
                        )# 64
        self.encoder2 = ActivatedLinear(64,symbol_channels)
        self.encoder2_act = nn.Sequential(
                        nn.Sigmoid(),
                        )#  sc
        
        
    def forward(self, x, snr):
        x = self.prep_act(self.prep(x, snr))
        x = self.layer1_act(self.layer1(x, snr))
        x = self.layer1_res_act(self.layer1_res(x, snr))
        x = self.layer2_act(self.layer2(x, snr))
        x = self.layer3_act(self.layer3(x, snr))
        x = self.encoder1_act(self.encoder1(x, snr))
        x = self.encoder2_act(self.encoder2(x, snr))
        return x 
    
class T_HyperAJSCC_decoder(nn.Module):
    
    def __init__(self, symbol_channels, num_classes, device):
        super().__init__() 
        
        self.pre = ActivatedLinear(symbol_channels, 64)
        self.pre_act = nn.ReLU()
        self.linear1 = ActivatedLinear(64, 64)
        self.linear1_act = nn.ReLU()
        self.linear2 = ActivatedLinear(64, 64)
        self.linear2_act = nn.ReLU()
        
        self.dlayer_1 = ActivatedConv2d(4,512,kernel_size = 3,stride = 1, padding = 1, bias = False)
        self.dlayer_1_act = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )# 512x4x4
        self.res = ActivatedResblock2d(512)# 512x4x4
        self.res_act = nn.Sequential(
            nn.MaxPool2d(kernel_size = 4, stride = 4, padding = 0, dilation = 1, ceil_mode = False),
            nn.Flatten(),
        )# 512
        self.final = ActivatedLinear(512, num_classes, bias=False)
        
    def forward(self, x, snr):
        x = self.pre_act(self.pre(x, snr))
        x = self.linear1_act(self.linear1(x, snr))
        x = self.linear2_act(self.linear2(x, snr)).reshape(-1, 4, 4, 4)# 4x4x4
        x = self.dlayer_1_act(self.dlayer_1(x, snr))# 512x4x4
        x = self.res_act(self.res(x, snr))# 512
        return self.final(x, snr)

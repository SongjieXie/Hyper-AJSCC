import torch 
import torch.nn as nn


class AWGN_complex(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def power_normalize_channel(self, x):
        """Power normalization for data-oriented communication.
        Normalize features of each image, and the input features with tensor shape of [b c h w]

        Args:
            x (tensor): [b c h w] 
 
        Returns:
            normalized features: [b c h w]
        """
        power_norm = torch.sqrt(2*(x**2).mean((-3, -2, -1), keepdim=True))
        return x/power_norm
    
    def power_normalize_all(self, x):
        """ Power normalization for task-oriented.
        The number of channel symbols for each image in task-oriented comm is too small, with the range of 8~32. 
        To reduce the variance, we normalize all the channel symbols for all the image.

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        numele = x.numel()//2
        power_norm = torch.sqrt(
            torch.sum(x**2)/numele
        )
        return x/power_norm
    
    def snr2logd(self, snrs):
        return torch.log(torch.tensor(10).to(snrs.device))*(-snrs/20)
    
    def channel(self, x, snrs):
        """
        The complex Gaussian variable is equivalent to the real and img part are iid normaly
        distributed with same mean and variance of \delta^2/2
        """
        logdelta = self.snr2logd(snrs)
        delta = torch.exp(logdelta)/torch.sqrt(torch.tensor(2)) if len(x.shape) <=2 else (torch.exp(logdelta)/torch.sqrt(torch.tensor(2))).unsqueeze(-1).unsqueeze(-1)
        noise = torch.randn_like(x).to(x.device)
        return x + delta*noise
    
    def forward(self, x, snrs):
        x = self.power_normalize_all(x) if len(x.shape) <=2 else self.power_normalize_channel(x)
        return self.channel(x, snrs)
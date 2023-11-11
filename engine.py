from datetime import datetime
from typing import Iterable
import torch 
from tensorboardX import SummaryWriter
from models.channel_models import AWGN_complex
from utils import sample, snr2logd, mse2psnr, psnr, accuracy



def train_one_epoch(dataloader:Iterable,  model_enc:torch.nn.Module, model_dec:torch.nn.Module,
                    channel: str, range_snr: list,
                    optimizer:torch.optim.Optimizer, criterion:torch.nn.Module,
                    writer: SummaryWriter, epoch: int, is_base: bool, args):
    loss_value = 0.
    model_enc.train()
    model_dec.train()
    batches_start = datetime.now()
    for i_batch, (imgs, labs) in enumerate(dataloader):
        num_imgs = imgs.shape[0]
        imgs = imgs.to(args.device)
        labs = labs.to(args.device)
        if channel == 'awgn':
            # Sampling the values of SNR uniformly
            awgn = AWGN_complex() 
            snrs = sample(range_snr, num_imgs, args.sampling).to(args.device)
            z = model_enc(imgs) if is_base else model_enc(imgs, snrs)
            z_hat = awgn(z, snrs)
            imgs_hat = model_dec(z_hat) if is_base else model_dec(z_hat, snrs)
            
        elif channel == 'fading':
            pass
        loss = criterion(imgs, imgs_hat) if args.type == 'recon' else criterion(imgs_hat, labs)
        optimizer.zero_grad()
        loss.backward()
        if args.maxnorm > 0:
            torch.nn.utils.clip_grad_norm_(model_enc.parameters(), args.maxnorm)
            torch.nn.utils.clip_grad_norm_(model_dec.parameters(), args.maxnorm)
        optimizer.step()

        
        writer.add_scalar('train/losses', loss.item(), args.n_iter)
        args.n_iter += 1
        loss_value += loss.item()
        
        
        if i_batch %100 == 0:
            batches_end = datetime.now()
            avg_time = (batches_end - batches_start)/80
            print('\n \n average batch time for batch size of', imgs.shape[0],':', avg_time)
            batches_start = datetime.now()
            print('[%d][%d/%d]\t Losses:%.4f\t'
                  %(epoch, i_batch, len(dataloader), loss.item()))
            # print('acc1:%.4f\t acc3:%.4f'%(acc[0].item(), acc[1].item()))
    
    return loss_value/len(dataloader)

def test(dataloader:Iterable,  model_enc:torch.nn.Module, model_dec:torch.nn.Module, 
             channel: str, snr: float, criterion:torch.nn.Module, 
             writer: SummaryWriter, idx: int, is_base: bool, args):
    psnr255 = 0.
    mse = 0.
    with torch.no_grad():
        model_enc.eval()
        model_dec.eval()
        print("Testing")
        for imgs, labs in dataloader:
            imgs = imgs.to(args.device)
            labs = labs.to(args.device)
            
            if channel == "awgn":
                awgn = AWGN_complex() 
                snrs = snr*torch.ones(imgs.shape[0], 1).to(args.device) if is_base else sample([0, 20], imgs.shape[0], args.sampling).to(args.device)
                z = model_enc(imgs) if is_base else model_enc(imgs, snrs)
                z_hat = awgn(z, snrs)
                imgs_hat = model_dec(z_hat) if is_base else model_dec(z_hat, snrs)
            elif channel == "fading":
                pass
            else:
                assert 1 == 0, "Unfinished ..."
                
            mse += criterion(imgs, imgs_hat).item() if args.type == 'recon' else accuracy(imgs_hat, labs, (1,3))[0].item()
            if args.type == 'recon':
                psnr255 += psnr(imgs, imgs_hat)
            
    mse = mse/len(dataloader)
    if args.type == 'recon':
        psnr255 = psnr255/len(dataloader)
        writer.add_scalar('val/psnr', mse2psnr(mse), idx)
        writer.add_scalar('val/psnr255', psnr255, idx)
        writer.add_scalar('val/snr', snr, idx)
    else:
        writer.add_scalar('val/acc', mse, idx)
        
    return mse
        

def evaluate(dataloader:Iterable,  model_enc:torch.nn.Module, model_dec:torch.nn.Module, 
             channel: str, snr: float, criterion:torch.nn.Module, 
             writer: SummaryWriter, idx: int, is_base: bool, args):
    psnr255 = 0.
    mse = 0.
    with torch.no_grad():
        model_enc.eval()
        model_dec.eval()
        print("Testing")
        for imgs, labs in dataloader:
            imgs = imgs.to(args.device)
            labs = labs.to(args.device)
            
            if channel == "awgn":
                awgn = AWGN_complex() 
                snrs = snr*torch.ones(imgs.shape[0], 1).to(args.device)
                z = model_enc(imgs) if is_base else model_enc(imgs, snrs)
                z_hat = awgn(z, snrs)
                imgs_hat = model_dec(z_hat) if is_base else model_dec(z_hat, snrs)
            else:
                assert 1 == 0, "Unfinished ..."
                
            mse += criterion(imgs, imgs_hat).item() if args.type == 'recon' else accuracy(imgs_hat, labs, (1,3))[0].item()
            if args.type == 'recon':
                psnr255 += psnr(imgs, imgs_hat)
    
    mse = mse/len(dataloader)
    if args.type == 'recon':
        psnr255 = psnr255/len(dataloader)
        writer.add_scalar('val/psnr', mse2psnr(mse), idx)
        writer.add_scalar('val/psnr255', psnr255, idx)
        writer.add_scalar('val/snr', snr, idx)
    else:
        writer.add_scalar('val/acc', mse, idx)
            

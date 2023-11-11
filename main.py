import os
import itertools
import torch
import torch.nn as nn 
from tensorboardX import SummaryWriter
import torch.optim as optim 
import torch.optim.lr_scheduler as lr_scheduler

from models.models import HyperAJSCC_encoder, HyperAJSCC_decoder, T_HyperAJSCC_encoder, T_HyperAJSCC_decoder
    
from datasets.dataloader import get_data
from engine import train_one_epoch, evaluate, test

def main(args):
    """ Model and Opimizer """
    # Each complex symbol comprises two values, representing both the imaginary and real parts
    num_symbols_real_img = args.sc * 2
    model_enc = HyperAJSCC_encoder(3, num_symbols_real_img, args.device) if args.type == 'recon' else T_HyperAJSCC_encoder(3, num_symbols_real_img, args.device)
    model_dec = HyperAJSCC_decoder(num_symbols_real_img, 3, args.device) if args.type == 'recon' else T_HyperAJSCC_decoder(num_symbols_real_img, args.num_classes, args.device)
    

    model_enc.to(args.device)
    model_dec.to(args.device)
    
    models_params = itertools.chain(model_enc.parameters(), model_dec.parameters())
    optimizer = optim.AdamW(params=models_params, lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.5)    

    """ Criterion """
    criterion = nn.MSELoss() if args.type == 'recon' else nn.CrossEntropyLoss()
    criterion.train()
    
    """ dataloader """
    dataloader_train =  get_data(args.dataset, args.N, n_worker= args.num_workers, model_type=args.type)
    dataloader_vali = get_data(args.dataset, 1000, n_worker= args.num_workers, train=False, model_type=args.type)
    
    """ writer """
    root_d = './logs_{0}/'.format(args.type) if args.train == 1 else './result_{0}/'.format(args.type)
    log_writer = SummaryWriter(root_d + name)
    
    
    """ Training """
    if args.train == 1:
        """ HERE!!! """
        range_snr = [args.snr_low, args.snr_high] 
        
        current_epoch = 0
        best_mse = 1e6 
        best_acc = 0.
        if os.path.isfile(path_to_backup):
            checkpoint = torch.load(path_to_backup, map_location='cpu')
            model_enc.load_state_dict(checkpoint['model_enc_states'])
            model_dec.load_state_dict(checkpoint['model_dec_states'])
            optimizer.load_state_dict(checkpoint['optimizer_states'])
            current_epoch = checkpoint['epoch']  
            
        for epoch in range(current_epoch, args.epoches):
            
            val_loss = train_one_epoch(dataloader_train, model_enc, model_dec, args.channel, 
                            range_snr, optimizer,
                            criterion, log_writer, epoch, False, args)
            scheduler.step()
            if (epoch > 0): 
                val = test(dataloader_vali, model_enc, model_dec, args.channel, 
                    args.snr, criterion, log_writer, epoch, False, args)
                print(">>>>ACC:", val)
                if args.type == 'recon':
                    if (epoch == 0) or (val < best_mse):
                        best_mse = val
                        with open('{0}/best.pt'.format(path_to_backup, epoch), 'wb') as f:
                            torch.save(
                            {
                            'epoch': epoch, 
                            'model_enc_states': model_enc.state_dict(),
                            'model_dec_states': model_dec.state_dict(),
                            'optimizer_states': optimizer.state_dict(),
                            }, f
                        )
                else:
                    if (epoch == 0) or (val > best_acc):
                        best_acc = val
                        with open('{0}/best.pt'.format(path_to_backup, epoch), 'wb') as f:
                            torch.save(
                            {
                            'epoch': epoch, 
                            'model_enc_states': model_enc.state_dict(),
                            'model_dec_states': model_dec.state_dict(),
                            'optimizer_states': optimizer.state_dict(),
                            }, f
                        )
                    
                    
    
    else:
        """ load model """
        path_to_model = '{0}/best.pt'.format(path_to_backup)
        checkpoint = torch.load(path_to_model, map_location='cpu')
        model_enc.load_state_dict(checkpoint['model_enc_states'])
        model_dec.load_state_dict(checkpoint['model_dec_states'])
        
        SNRs = list(range(0,21, 1))
        
        for idx, snr in enumerate(SNRs):
            evaluate(dataloader_vali, model_enc, model_dec, args.channel, 
                     snr, criterion, log_writer, idx, False, args)
            


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    

    parser = argparse.ArgumentParser(description='Hypernet-DJSCC')
    
    parser.add_argument('-d', '--dataset', type=str, default='CIFAR10', help='dataset name')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes for image classification')
    parser.add_argument('--type', type=str, default='task', help='The types including recon and classification')
    parser.add_argument('--sc', type=int, default=8, help='The number of channels for symbol transmission, or the number of symbols for taks-oriented')
    parser.add_argument('--channel', type=str, default='awgn', help='The channel models including awgn and fading')
    
    # only for the base model
    parser.add_argument('--snr', type=float, default=10.0, help='The snr of AWGN channel when training base model')
    
    # only for the adaptive model
    parser.add_argument('--snr_low', type=float, default=0.0, help='The minimal snr for training adaptive JSCC')
    parser.add_argument('--snr_high', type=float, default=20.0, help='The maximal value of snr for training JSCC')
    parser.add_argument('--sampling', type=str, default='uniform', help='The distribution of snr for training JSCC')
    parser.add_argument('--adaptive_layers', type=int, default=None, help='The number of adaptive layers, None refers to the full layers')
    # parser.add_argument('--debug', type=str, default=None, help='The number of adaptive layers, None refers to the full layers')
    parser.add_argument('--id', type=str, default='origin', help='')
    
    parser.add_argument('-r', '--root', type=str, default='./trained_models', help='The root of trained models')
    parser.add_argument('--device', type=str, default='cuda:0', help= 'The device')
    
    parser.add_argument('--train', type=int, default=1, help='1: train, 0: evaluation')
    parser.add_argument('-e', '--epoches', type=int, default=320, help='Number of epoches')
    parser.add_argument('--N', type=int, default=128, help='The batch size of training data')
    parser.add_argument('--lr', type=float, default=1e-3, help='learn rate')
    parser.add_argument('--maxnorm', type=float, default=1., help='The max norm of flip')
    parser.add_argument('--step', type=int, default=120, help='learn rate')
    

    parser.add_argument('--num_workers', type=int, default=mp.cpu_count() - 8,
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    
    args = parser.parse_args()
    args.n_iter = 0
    
    print('number of workers (default: {0})'.format(args.num_workers))
    
    name = 'HyperAJSCC-' + str(args.sc) + '-' + args.dataset + '-' + args.channel + '-' + str(args.type)  + '-' + args.id
 
    path_to_backup = os.path.join(args.root, name)
    if not os.path.exists(path_to_backup):
        print('Making ', path_to_backup, '...')
        os.makedirs(path_to_backup)

    args.device = torch.device(args.device if(torch.cuda.is_available()) else "cpu")
    print('Device: ', args.device)

    main(args)

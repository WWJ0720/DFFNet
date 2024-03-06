import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import AverageMeter
from datasets.loader import PairLoader
from models import *
# from utils.CR import ContrastLoss
from utils.CR_res import ContrastLoss_res
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed = 3407  # 设置种子值
set_seed(seed)  # 设置随机种子

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DFFNet_T', type=str, help='model name')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='/sdb/wwj/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='3', type=str, help='GPUs used for training')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def train(train_loader, network, criterion, optimizer, scaler):
    losses = AverageMeter()

    torch.cuda.empty_cache()

    network.train()

    for batch in train_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with autocast(args.no_autocast):
            output = network(source_img)

            output_fft = torch.fft.fft2(output, dim=(-2, -1))
            output_fft = torch.stack((output_fft.real, output_fft.imag), -1)

            target_img_fft = torch.fft.fft2(target_img, dim=(-2, -1))
            target_img_fft = torch.stack((target_img_fft.real, target_img_fft.imag), -1)

            loss = criterion[0](output, target_img) + criterion[1](output, target_img, source_img) * 0.1 + criterion[0](
                output_fft, target_img_fft) * 0.1

        losses.update(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return losses.avg


def valid(val_loader, network):
    PSNR = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    for batch in val_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        factor = 16
        h, w = source_img.shape[2], source_img.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        source_img = F.pad(source_img, (0, padw, 0, padh), 'reflect')

        with torch.no_grad():  # torch.no_grad() may cause warning
            output = network(source_img)[:, :, :h, :w].clamp_(-1, 1)

        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), source_img.size(0))

    return PSNR.avg


if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    # pretrain weights loader
    # checkpoint=torch.load('/home/ubuntu/515wwj/ImageDehazing/DFFNet/weights/DFFNet_S/indoor/DFFNet_S.pth')

    checkpoint = None
    network = eval(args.model)()
    network = nn.DataParallel(network).cuda()
    if checkpoint is not None:
        network.load_state_dict(checkpoint['state_dict'])

    criterion = []
    criterion.append(nn.L1Loss())
    criterion.append(ContrastLoss_res(ablation=False).cuda())

    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
    else:
        raise Exception("ERROR: unsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'],
                                                           eta_min=setting['lr'] * 1e-2)
    scaler = GradScaler()

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        best_psnr = checkpoint['best_psnr']
        start_epoch = checkpoint['epoch'] + 1
    else:
        best_psnr = 0
        start_epoch = 0

    best_psnr = 0

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_dataset = PairLoader(dataset_dir, 'train', 'train',
                               setting['patch_size'],
                               setting['edge_decay'],
                               setting['only_h_flip'])
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
    val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'],
                             setting['patch_size'])
    val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size'],
                            num_workers=args.num_workers,
                            pin_memory=True)

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    # if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):
    print('==> Start training, current model name: ' + args.model)

    train_ls, test_ls, idx = [], [], []

    for epoch in tqdm(range(start_epoch, setting['epochs'] + 1)):
        loss = train(train_loader, network, criterion, optimizer, scaler)

        train_ls.append(loss)
        idx.append(epoch)

        scheduler.step()

        if epoch % setting['eval_freq'] == 0:
            avg_psnr = valid(val_loader, network)
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                print(avg_psnr)

                torch.save({'state_dict': network.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': scheduler.state_dict(),
                            'scaler': scaler.state_dict(),
                            'epoch': epoch,
                            'best_psnr': best_psnr
                            },
                           os.path.join(save_dir, args.model + '.pth'))

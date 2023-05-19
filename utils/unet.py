# The implementatoin below is adopted from: https://github.com/milesial/Pytorch-UNet
import torch, os, sys
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./'))

from utils.utils_data import *
from lib.loss import *


''' ---- UNet Model ----'''

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)

        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



''' ---- Dice Loss ----'''

def dice_coeff(input, target, reduce_batch_first: bool = False, epsilon=1e-6):
        # Average of Dice coefficient for all batches, or for a single mask
        assert input.size() == target.size()
        if input.dim() == 2 and reduce_batch_first:
            raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

        if input.dim() == 2 or reduce_batch_first:
            inter = torch.dot(input.reshape(-1), target.reshape(-1))
            sets_sum = torch.sum(input) + torch.sum(target)
            if sets_sum.item() == 0:
                sets_sum = 2 * inter

            return (2 * inter + epsilon) / (sets_sum + epsilon)
        else:
            # compute and average metric for each batch element
            dice = 0
            for i in range(input.shape[0]):
                dice += dice_coeff(input[i, ...], target[i, ...])
            return dice / input.shape[0]

def dice_coeff_multi_class(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]

def dice_loss(input, target, multiclass: bool = False): 
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = dice_coeff_multi_class if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)



''' ---- Train/Eval Pipeline ---- '''

def eval_one_scene(net, sample):
    rgb, depth, label, meta, suffix = \
        sample['rgb'], sample['depth'], sample['label'], sample['meta'], sample['suffix']
    meta = load_pickle(meta[0])
    print(f"Shapes: rgb {rgb.shape, rgb.dtype}, depth {depth.shape}, label {label.shape}")

    deflate = DeflateLabel()

    pred = net(rgb)
    pred = (F.sigmoid(pred) > 0.9).float()
    pred = deflate(pred).squeeze()
    label = label.squeeze().to(device=torch.device('cpu'))

    print(f'pred shape: {pred.shape}')
    print(f'rgb shape: {rgb.shape}')


    visualize_one_scene(rgb, depth, pred, meta, suffix)     # Predicted segmentation
    visualize_one_scene(rgb, depth, label, meta, suffix)     # Predicted segmentation




def train_net(net, loader, p, checkpoint=False):
    '''Input p for Params: 
        "device":device,                    
        'bz': 2, 'shuffle': True, 'num_workers':1,  # 1. For loader
        "epochs": 1, "lr": 1e-5,                    # 2. For learning
        "scale":0.5,    "amp": False                # 3. Grad scaling
    '''
    
    logging.info(f'''Starting training:
        Epochs:          {p['epochs']}
        Learning rate:   {p['lr']}
        Training size (num batches):   {len(loader)}
        Batch size:      {p['bz']}
        Checkpoints:     {p['checkpoint']}
        Device:          {p["device"]}
        Images scaling:  {p['scale']}
        Mixed Precision: {p['amp']}
    ''')
    

    # 1. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=p['lr'], weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=p['lr'])

    # NOTE: ReduceLROnPlateau should only be called after validation stage
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, total_iters=p['epochs'])
    grad_scaler = torch.cuda.amp.GradScaler(init_scale=p['scale'], enabled=p['amp'])
    criterion = nn.CrossEntropyLoss()
    criterion_focal = FocalLoss()
    global_step = 0

    
    # 5. Begin training
    net.train()
    total_epochs = p['epochs']
    for epoch in range(1, p['epochs']+1):
        
        with tqdm(total=len(loader)*p['bz'], desc=f'Epoch {epoch}/{total_epochs}', unit='img') as pbar:
            
            epoch_loss = 0

            for batch in loader:
                
                rgb, depth, label, meta, suffix = \
                    batch['rgb'], batch['depth'], batch['label'], batch['meta'], batch['suffix']
                meta = [load_pickle(mfile) for mfile in meta]

                assert rgb.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {rgb.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                rgb = rgb.to(device=p['device'], dtype=torch.float32)     # (N, C, H, W)
                label = label.to(device=p['device'], dtype=torch.float32) # (N, C, H, W)
                
                # https://stackoverflow.com/questions/63383347/runtimeerror-expected-object-of-scalar-type-long-but-got-scalar-type-float-for
                # https://blog.paperspace.com/unet-architecture-image-segmentation/
                # https://towardsdatascience.com/a-machine-learning-engineers-tutorial-to-transfer-learning-for-multi-class-image-segmentation-b34818caec6b


                with torch.cuda.amp.autocast(enabled=p['amp']):
                    masks_pred = net(rgb)

                    # loss = criterion(masks_pred, label) +\
                    #     dice_loss(F.softmax(masks_pred, dim=1).float(), label, multiclass=True)
                    # loss_dice = dice_loss(F.softmax(masks_pred, dim=1).float(), label, multiclass=True)

                    loss_entropy = criterion(masks_pred, label)

                    B,C,H,W = masks_pred.shape
                    masks_pred = masks_pred.reshape(B,C,-1).contiguous()
                    label = torch.argmax(label, dim=1).reshape(B,-1).contiguous()
                    loss_focal = criterion_focal(masks_pred, label)

                    loss = loss_focal + loss_entropy

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.step()

                pbar.update(rgb.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

            if checkpoint and epoch % 4 == 0:
                fname = f'./exp/unet_weight_checkpoint_{epoch}_.pt'
                torch.save(net.state_dict(), fname)

            print(epoch_loss)
            

if __name__ == '__main__':
    print('---- Testing UNet ----')
    input = torch.rand((1,3,128,256))
    print(f'\t Input: {input.shape}')

    
    model = UNet(n_channels=3, n_classes=82, bilinear=True)
    out = model(input)
    print(f'\t Output: {out.shape}')


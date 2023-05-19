# https://github1s.com/delta-onera/segnet_pytorch/blob/master/segnet.py

import torch, sys, logging, os
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import optim

sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./'))

from utils.utils_data import *
from lib.loss import *
from torchvision.transforms import Pad


class SegNet(nn.Module):

    def __init__(self, input_nbr=3, label_nbr=82, BN_momentum=0.5):
        super(SegNet, self).__init__()

        #SegNet Architecture
        #Takes input of size in_chn = 3 (RGB images have 3 channels)
        #Outputs size label_chn (N # of classes)

        #ENCODING consists of 5 stages
        #Stage 1, 2 has 2 layers of Convolution + Batch Normalization + Max Pool respectively
        #Stage 3, 4, 5 has 3 layers of Convolution + Batch Normalization + Max Pool respectively

        #General Max Pool 2D for ENCODING layers
        #Pooling indices are stored for Upsampling in DECODING layers

        self.input_nbr = input_nbr
        self.label_nbr = label_nbr

        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True) 

        self.ConvEn11 = nn.Conv2d(self.input_nbr, 64, kernel_size=3, padding=1)
        self.BNEn11 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNEn12 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BNEn21 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn22 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.BNEn31 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn33 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.BNEn41 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn43 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn51 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn53 = nn.BatchNorm2d(512, momentum=BN_momentum)


        #DECODING consists of 5 stages
        #Each stage corresponds to their respective counterparts in ENCODING

        #General Max Pool 2D/Upsampling for DECODING layers
        self.MaxDe = nn.MaxUnpool2d(2, stride=2) 

        self.ConvDe53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe53 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe51 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvDe43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe43 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.BNDe41 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvDe33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe33 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.BNDe31 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvDe22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNDe22 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvDe21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.BNDe21 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvDe12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNDe12 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvDe11 = nn.Conv2d(64, self.label_nbr, kernel_size=3, padding=1)
        self.BNDe11 = nn.BatchNorm2d(self.label_nbr, momentum=BN_momentum)

    def forward(self, x):

        #ENCODE LAYERS
        #Stage 1
        x = F.relu(self.BNEn11(self.ConvEn11(x))) 
        x = F.relu(self.BNEn12(self.ConvEn12(x))) 
        x, ind1 = self.MaxEn(x)
        size1 = x.size()

        #Stage 2
        x = F.relu(self.BNEn21(self.ConvEn21(x))) 
        x = F.relu(self.BNEn22(self.ConvEn22(x))) 
        x, ind2 = self.MaxEn(x)
        size2 = x.size()

        #Stage 3
        x = F.relu(self.BNEn31(self.ConvEn31(x))) 
        x = F.relu(self.BNEn32(self.ConvEn32(x))) 
        x = F.relu(self.BNEn33(self.ConvEn33(x)))   
        x, ind3 = self.MaxEn(x)
        size3 = x.size()

        #Stage 4
        x = F.relu(self.BNEn41(self.ConvEn41(x))) 
        x = F.relu(self.BNEn42(self.ConvEn42(x))) 
        x = F.relu(self.BNEn43(self.ConvEn43(x)))   
        x, ind4 = self.MaxEn(x)
        size4 = x.size()

        #Stage 5
        x = F.relu(self.BNEn51(self.ConvEn51(x))) 
        x = F.relu(self.BNEn52(self.ConvEn52(x))) 
        x = F.relu(self.BNEn53(self.ConvEn53(x)))   
        x, ind5 = self.MaxEn(x)
        size5 = x.size()

        #DECODE LAYERS
        #Stage 5
        x = self.MaxDe(x, ind5, output_size=size4)
        x = F.relu(self.BNDe53(self.ConvDe53(x)))
        x = F.relu(self.BNDe52(self.ConvDe52(x)))
        x = F.relu(self.BNDe51(self.ConvDe51(x)))

        #Stage 4
        x = self.MaxDe(x, ind4, output_size=size3)
        x = F.relu(self.BNDe43(self.ConvDe43(x)))
        x = F.relu(self.BNDe42(self.ConvDe42(x)))
        x = F.relu(self.BNDe41(self.ConvDe41(x)))

        #Stage 3
        x = self.MaxDe(x, ind3, output_size=size2)
        x = F.relu(self.BNDe33(self.ConvDe33(x)))
        x = F.relu(self.BNDe32(self.ConvDe32(x)))
        x = F.relu(self.BNDe31(self.ConvDe31(x)))

        #Stage 2
        x = self.MaxDe(x, ind2, output_size=size1)
        x = F.relu(self.BNDe22(self.ConvDe22(x)))
        x = F.relu(self.BNDe21(self.ConvDe21(x)))

        #Stage 1
        x = self.MaxDe(x, ind1)
        x = F.relu(self.BNDe12(self.ConvDe12(x)))
        x = self.ConvDe11(x)

        # x = F.softmax(x, dim=1)
        return x


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
    assert input.size() == target.size(), f'input: {input.shape}, target: {target.shape}'
    fn = dice_coeff_multi_class if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def train_model(net, loader, p, checkpoint=False):
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
    ''')
    

    # 1. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(net.parameters(), lr=p['lr'], weight_decay=1e-8, momentum=0.9)
    optimizer = optim.SGD(net.parameters(), lr=p['lr'], momentum=0.9)
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

                assert rgb.shape[1] == net.input_nbr, \
                    f'Network has been defined with {net.input_nbr} input channels, ' \
                    f'but loaded images have {rgb.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                rgb = rgb.to(device=p['device'], dtype=torch.float32)     # (N, C, H, W)
                label = label.to(device=p['device'], dtype=torch.long) # (N, C, H, W)
                
                # https://stackoverflow.com/questions/63383347/runtimeerror-expected-object-of-scalar-type-long-but-got-scalar-type-float-for
                # https://blog.paperspace.com/unet-architecture-image-segmentation/
                # https://towardsdatascience.com/a-machine-learning-engineers-tutorial-to-transfer-learning-for-multi-class-image-segmentation-b34818caec6b


                with torch.cuda.amp.autocast(enabled=p['amp']):
                    masks_pred = net(rgb)
                    
                    # loss_dice = dice_loss(F.softmax(masks_pred, dim=1).float(), label, multiclass=True)

                    # B,C,H,W = masks_pred.shape
                    # masks_pred = masks_pred.reshape(B,C,-1).contiguous()
                    # label = torch.argmax(label, dim=1).reshape(B,-1).contiguous()
                    # loss_focal = criterion_focal(masks_pred, label)

                    # loss = loss_focal + loss_dice

                    # loss = criterion(masks_pred, label) +\
                    #     dice_loss(F.softmax(masks_pred, dim=1).float(), label, multiclass=True)
                    
                    # print(f'mask:{masks_pred.shape}, label:{label.squeeze(1).shape, label.dtype}')
                    loss = criterion(masks_pred, label.squeeze(1)) 
                    

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(rgb.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

            if checkpoint and epoch % 4 == 0:
                fname = f'./exp/segnet_weight_checkpoint_{epoch}_.pt'
                torch.save(net.state_dict(), fname)

            print(epoch_loss)

if __name__ == '__main__':
    print('---- Testing Segnet ----')
    input = torch.rand((1,3,512,512))
    pred = torch.randint(10, (1,512,512))
    print(f'\t Input: {input.shape}')

    # pad = Pad((0,280))
    # input = pad(input)
    # print(f'\t Padded input: {input.shape}')
    
    model = SegNet(3,82)
    out = model(input)
    print(f'\t Output: {out.shape}')
    print(f'\t pred gt: {pred.shape}')
    criterion = nn.CrossEntropyLoss()
    B,C,H,W = out.shape
    loss = criterion(out, pred)



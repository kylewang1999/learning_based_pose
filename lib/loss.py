from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from torchvision.ops.focal_loss import sigmoid_focal_loss
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn


class FocalLoss(_Loss):
    ''' Loss for point-wise segmentation'''
    def __init__(self, gamma=0, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.eps=1e-7

    def forward(self, input, target):        
        # # https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/focal.html
        # assert len(input.shape)==len(target.shape)+1, f'Input should have 1 more dimension than target. But input:{input.shape}, target:{target.shape}'

        input_soft = F.softmax(input, dim=1) + self.eps               # (B,C,N)
        target_one_hot = one_hot(target, num_classes=input.shape[1])  # (B,C,N)
        
        q = input_soft * target_one_hot                    # (B,C,N)
        q = torch.sum(q, dim=1)                            # (B,N)
        weight = torch.pow(1.0 - q, self.gamma)
        loss = -self.alpha * weight * torch.log(q)
        return torch.mean(loss)


def one_hot(label, num_classes):
    '''One-hot input input tensor'''
    one_hot = F.one_hot(label.long(), num_classes)  # (B,H,W,C)
    if len(one_hot.shape)>2:
        one_hot = torch.transpose(one_hot, 1, -1)   
    return one_hot


def of_l1_loss(pred_ofsts, kp_targ_ofst, labels, sigma=1.0, normalize=True, reduce=True):
    ''' FIXME: Support multi-class label
        pred_ofsts:      [B, N_kpts, N_pts, C=3]
        kp_targ_ofst:    [B, N_pts, N_kpts, C=3]
        labels:          [B, N_pts, 1]
    '''
    B, n_kpts, n_pts, C = pred_ofsts.shape
    w = (labels > 1e-8).float()         # (B, N_pts, 1)
    
    sigma_2 = sigma ** 3

    w = w.view(B, 1, n_pts, 1).repeat(1, n_kpts, 1, 1).contiguous()     # W: (B, N_kpts, N_pts, 1)
    kp_targ_ofst = kp_targ_ofst.view(B, n_pts, n_kpts, 3)               # (B, N_pts, N_kpts, 3)
    kp_targ_ofst = kp_targ_ofst.permute(0, 2, 1, 3).contiguous()        # (B, N_kpts, N_pts, 3)
    diff = pred_ofsts - kp_targ_ofst
    abs_diff = torch.abs(diff)
    abs_diff = w * abs_diff
    in_loss = abs_diff

    if normalize:
        in_loss = torch.sum(
            in_loss.view(B, n_kpts, -1), 2
        ) / (torch.sum(w.view(B, n_kpts, -1), 2) + 1e-3)

    if reduce:
        torch.mean(in_loss)

    return in_loss


class OFLoss(_Loss): 
    ''' L1 Loss for offset prediction '''
    def __init__(self):
        super(OFLoss, self).__init__(True)

    def forward(self, pred_ofsts, kp_targ_ofst, labels,
                normalize=True, reduce=False):
        return of_l1_loss(pred_ofsts, kp_targ_ofst, labels, normalize, reduce)



if __name__ == '__main__':
    
    from torch.autograd import Variable
    H = 16

    data_pred = Variable(torch.rand(4,82,H))
    data_gt = Variable(torch.randint(82, (4,H)))
    print(f' \n---- Testing FocalLoss with pred [{data_pred.shape}, gt [{data_gt.shape}]]...----')
    focal_loss = FocalLoss()
    loss = focal_loss(data_pred, data_gt)
    print(f'\t data: {data_gt.shape, one_hot(data_gt,82).shape }')
    print(f'\t data: {data_gt.shape, F.one_hot(data_gt,82).shape }')
    print(f'\t FocalLoss Out: {loss, loss.shape}')


    # N1, N2 = 200, 256
    # pred_ofst = Variable(torch.rand(4, N2, N1, 3))
    # kpgt_ofst = Variable(torch.rand(4, N1, N2, 3))
    # label = Variable(torch.rand(4, N1, 1))

    # print(f' \n---- Testing OFLoss (torchvision)with pred [{pred_ofst.shape}, gt [{kpgt_ofst.shape}]]...----')
    # of_loss = OFLoss()
    # loss = of_loss(pred_ofst, kpgt_ofst, label)
    # print(f'\t OFLess out: {loss.shape}')

    # print(' ---- Testing one hot ---')
    # label = torch.empty(2, 3, 5, dtype=torch.long).random_(12)
    # label_one_hot = one_hot(label)
    # print(f'\t label: {label.shape}, onehot: {label_one_hot.shape}')
    



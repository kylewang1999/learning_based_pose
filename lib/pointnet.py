# https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py
# https://github.com/yanx27/Pointnet_Pointnet2_pytorch

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F



class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        assert x.shape[1]==3, f'STN3D expect input of shape (B, 3, N), but got {x.shape}'
        # Output (B,3,3): A transformation for 1 batch of N points.
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()

        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        try:
            assert x.shape[1]==self.k, f'STNkD expect input of shape (B, {self.k}, N), but got {x.shape}'
        except:
            assert x.shape[-1] == self.k, f'failed when trying again using channel-last formatting. input shape: {x.shape}'
            x = x.permute(0, 2, 1)

        # Output (B,3,3): A transformation for 1 batch of N points.
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False, C=6):
        super(PointNetfeat, self).__init__()

        self.stn = STNkd(k=C)                        # TODO: Remove stn
        self.conv1 = torch.nn.Conv1d(C, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)              # TODO: Max pool
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        assert len(x.shape)==3, f'PointNetfeat expect input of shape (B, C, N), but got {x.shape}'
        # Output (B, 1088, N): Per-point feature concat with global feature
        
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetExtractor(nn.Module):
    ''' Use this in PVN3D '''
    def __init__(self, feature_transform=False, C=6):
        super(PointNetExtractor, self).__init__()
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform, C=C)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)  
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        assert len(x.shape)==3, f'Input should be of shape (B,C,N), but got {x.shape}'
        # Output (B, 128, N): 128-dim feature for each point.
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        return x, trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':

    C = 3
    sim_data = Variable(torch.rand(1,C,400))
    print(f' \n---- Testing pointnet with input shape [{sim_data.shape}] ---- ')

    trans = STNkd(k=C)
    out = trans(sim_data)
    print('\tstn64d', out.size())
    print('\tloss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True, feature_transform=True, C=C)
    out, trans, trans_feat = pointfeat(sim_data)
    print('PointNetfeat (global)', out.size(), trans.size(), trans_feat.size())

    pointfeat = PointNetfeat(global_feat=False,feature_transform=True, C=C) 
    out, trans, trans_feat = pointfeat(sim_data)
    print('PointNetfeat (non-global)', out.size(), trans.size(), trans_feat.size())

    # cls = PointNetCls(k = 5)
    # out, _, _ = cls(sim_data)
    # print('PointNetCls ', out.size())

    # seg = PointNetSeg(k = 3)
    # out, _, _ = seg(sim_data)
    # print('PointNetSeg', out.size())

    seg = PointNetExtractor(feature_transform=True, C=C)
    out, _, _ = seg(sim_data)
    print('\tPointNetExtractor', out.size())


from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import logging, sys, os

sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./'))

from utils.utils_data import *
import lib.resnet as resnet
from   lib.pspnet import psp_models, modified_psp_models
from   lib.pointnet import *
from   lib.scheduler import *
from   lib.loss import *


class ModifiedResnet(nn.Module):
    # FIXME: If input image size too small, output doesn't match input H,W
    def __init__(self):
        super(ModifiedResnet, self).__init__()

        # self.model = modified_psp_models['resnet34'.lower()]()
        self.model = modified_psp_models['resnet18'.lower()]()

    def forward(self, x):
        x, x_seg = self.model(x)
        return x, x_seg


class DenseFusion(nn.Module):
    def __init__(self, num_points=360*720):
        super(DenseFusion, self).__init__()
        self.conv2_rgb = torch.nn.Conv1d(128, 256, 1)
        self.conv2_cld = torch.nn.Conv1d(128, 256, 1)

        self.conv3 = torch.nn.Conv1d(256, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)

        self.num_points = num_points
        self.ap1 = torch.nn.AvgPool1d(num_points)

    def forward(self, rgb_emb, cld_emb, choose=None): #TODO: Implement choose
        assert len(rgb_emb.shape)==4, f'rgb embedding should be of shape (B,C,H*W), but got {rgb_emb.shape}'
        assert len(cld_emb.shape)==3, f'cld embedding should be of shape (B,C,H*W), but got {cld_emb.shape}'
        H, W, N = rgb_emb.shape[-2], rgb_emb.shape[-1], cld_emb.shape[-1]
        assert H*W==N, f'rgb, cld embedding should have same N points. rgb points [{H*W}]. cld points [{N}]'
        assert self.num_points <= N, f'DenseFusion num_points too large ({self.num_points}), should be less than N={N}'
        
        B, C = rgb_emb.shape[:2]
        rgb_emb = rgb_emb.view(B, C, -1)    # rgb_emb, cld_emb now have the same shape  
        feat_1 = torch.cat((rgb_emb, cld_emb), dim=1)   # 128+128 = 256

        rgb = F.relu(self.conv2_rgb(rgb_emb))   # 128->256
        cld = F.relu(self.conv2_cld(cld_emb))   # 128->256
        feat_2 = torch.cat((rgb, cld), dim=1)   # 256+256=512

        rgbd = F.relu(self.conv3(feat_1))   # 256 -> 512
        rgbd = F.relu(self.conv4(rgbd))     # 512 -> 1024 (32, 1024, 1024)

        ap_x = rgbd         # FIXME: What's avg pooling for?

        # print(f'H,W,N: {H, W, N}')
        # print(f'\t\t Fusion feat1: {feat_1.shape}')
        # print(f'\t\t Fusion feat2: {feat_2.shape}')
        # print(f'\t\t Fusion rgbd: {rgbd.shape}')

        # ap_x = self.ap1(rgbd)               # (32)
        # print(f'\t\t Fusion ap_x: {ap_x.shape}')
        # ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, N)
        # print(f'\t\t Fusion ap_x (after repeat): {ap_x.shape}')

        return torch.cat([feat_1, feat_2, ap_x], 1) # 256 + 512 + 1024 = 1792


class MLP(nn.Module):
    def __init__(self, cout=81):
        super(MLP, self).__init__()
        
        self.conv1 = torch.nn.Conv1d(1792, 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, cout, 1)


    def forward(self, x):
        assert x.shape[1] == 1792, f'input to MLP layer must have shape (B, 1792, N), but got {x.shape}'
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x


class PVN3D(nn.Module):
    '''
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        pcd_input_c: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        pcld_use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
        num_kps: int = 8
            Number of keypoints to predict
        num_points: int 8192
            Number of sampled points from point clouds.
    '''
    def __init__(
        self, num_classes=81, pcd_input_c=3, pcld_use_xyz=True,
        num_kps=16, num_points=1000
    ):
        super(PVN3D, self).__init__()
        self.num_kps = num_kps
        self.num_classes = num_classes
        self.num_points = num_points
        self.pcd_input_c = pcd_input_c
        
        self.cnn = ModifiedResnet()
        self.pointnet = PointNetExtractor(C=pcd_input_c+3)
        self.fusion = DenseFusion(num_points=num_points)

        self.SEG  = MLP(cout = self.num_classes)    # 1792 - 1024 - 512 - 128 - N_cls=81
        self.KPOF = MLP(cout = 3*self.num_kps)      # 1792 - 1024 - 512 - 128 - N_KP*3  (Keypoint offsets)
        self.CTOF = MLP(cout=3)                     # 1792 - 1024 - 512 - 128 - 3       (Center offset)


    def forward(self, pcd, rgb, choose=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pcd (B, C=3/6, N) : Variable(torch.cuda.FloatTensor) 
                NOTE: Original code require pcd to be channel-last
            rgb (B, C=3, H, W): Variable(torch.cuda.FloatTensor) 
            choose (B, 1, N): Variable(torch.cuda.LongTensor)
                indexs of chosen points(pixels).
        """
        assert len(rgb.shape) == 4 and rgb.shape[1]==3, f'rgb shape: {rgb.shape}'

        pcd_emb,_,_ = self.pointnet(pcd)    #nvii
        rgb_emb, _ = self.cnn(rgb)          # (B,C,H,W)
        B, C, _, _ = rgb_emb.size()
        _, _, N = pcd.shape

        # choose = choose.repeat(1, C, 1)
        # rgb_emb = torch.gather(rgb_emb, 2, choose).contiguous()   # TODO: Choose?
        # print(f'pvnd3d. pcd:{pcd_emb.shape}')
        # print(f'pvnd3d. rgb:{rgb_emb.shape}')
        
        rgbd_feature = self.fusion(rgb_emb, pcd_emb)

        rgbd_seg = self.SEG(rgbd_feature).transpose(1, 2).contiguous()  # (B, N , N_cls=81)
        
        kpof = self.KPOF(rgbd_feature).view(B, self.num_kps, 3, N)      # (B, 3*num_kp, N) -> (B, num_kp, 3, N)
        kpof = kpof.permute(0, 1, 3, 2).contiguous()                    # (B, num_kp, N, 3)   NOTE: Channels last now

        ctof = self.CTOF(rgbd_feature).view(B, 1, 3, N)                 # (B, 3, N) -> (B, 1, 3, N)
        ctof = ctof.permute(0, 1, 3, 2).contiguous()                    # (B, 1, N, 3)        NOTE: Channels last now

        return kpof, rgbd_seg, ctof


def train_pvn3d(model, loader, p):

    logging.info(f''' ---- Starting training: ----
        Epochs:          {p['epochs']}
        Learning rate:   {p['lr']}
        Batch size:      {p['bz']}
        Checkpoints:     {p['checkpoint']}
        Device:          {p["device"]}
        Images scaling:  {p['scale']}
        Mixed Precision: {p['amp']}
        Weight Decay:    {p['decay']}
        Num Batches:     {len(loader)}
        Num Samples:     {len(loader)*p['bz']}
        N_objects:       {p['objects']}
        N_keypoints:     {p['keypoints']}
        N_samplepoints:  {p['samplepoints']}
    ''')
    model = model.to(device=p["device"])
    optimizer = optim.Adam(model.parameters(), lr=p['lr'], weight_decay=p['decay'])
    # lr_scheduler = CyclicLR(
    #     optimizer, base_lr=1e-5, max_lr=1e-3,
    #     # step_size=p['epochs'] * (len(loader) / p['bz']) // 6,
    #     # mode = 'triangular'
    # )

    criterion_focal, criterion_kpof, criterion_ctof= FocalLoss(), OFLoss(), OFLoss()
    
    total_epochs = p['epochs']
    for epoch in range(1, p['epochs']+1):
        with tqdm(total=len(loader)*p['bz'], desc=f'Epoch {epoch}/{total_epochs}', unit='img') as pbar:
            
            epoch_loss = 0

            for batch in loader:

                rgb    = batch['rgb'].to(device=p["device"])        # (B,3,H,W)
                label  = batch['label'].to(device=p["device"])      # (B,1,H,W)
                kpof_gt= batch['kpof_gt'].to(device=p["device"])    # (B, N_pts, N_kps, 3)
                
                ctof_gt= batch['ctof_gt'].to(device=p["device"])    # (B, N_pts, 3)
                ctof_gt= ctof_gt.unsqueeze(2)                       # (B, N_pts, 1, 3)

                pcd_rgb= batch['pcd_rgb'].to(device=p["device"])    # (B, N_pts, 6)
                pcd_rgb = pcd_rgb.permute(0,2,1) # -> (B, 6, N_pts) 
 
                
                # print(f'input shapes: {pcd_rgb.shape}, {rgb.shape}')

                kpof_pred, seg_pred, ctof_pred = model(pcd_rgb, rgb, choose = None)

                loss_seg = criterion_focal(
                    seg_pred.view(label.numel(), -1),  # (B, N_pts, N_cls=82) -> (B*N_pts, 82)
                    label.reshape(-1)                  # (B*H*W) 
                ).sum()

                
                loss_kpof = criterion_kpof(
                  kpof_pred,  # (B, N_kps, N_pts, 3)
                  kpof_gt,    # (B, N_pts, N_kps, 3)
                  label
                 ).sum()

                loss_ctof = criterion_ctof(
                  ctof_pred,   # (B, 1, N_pts, 3)
                  ctof_gt,     # (B, N_pts, 3)
                  label).sum() # (B,1,H,W)
                
                # loss = 2.0*loss_seg + 1.0*loss_ctof + 1.0*loss_kpof
                loss = loss_seg 

                optimizer.zero_grad()
                loss.backward()
                epoch_loss += loss.item()

                pbar.update(rgb.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})

            if epoch % 10 == 0:
                fname = f'./exp/pvn3d_weight_checkpoint_{epoch}_.pt'
                torch.save(model.state_dict(), fname)

            print(epoch_loss)




if __name__ == '__main__':
    # print(' ---- Testing pvn3d.py ...----')
    # from torch.autograd import Variable
    # H = 40
    # C = 0
    # sim_data_rgb = Variable(torch.rand(2,C+3,H,H))
    # sim_data_pcd = Variable(torch.rand(2,C+3,H*H))

    # print(f' \n---- Testing `pspnet` with input shape [{sim_data_rgb.shape, sim_data_pcd.shape}] ---- ')

    # resnet = ModifiedResnet()
    # resnet_out1, resnet_out2 = resnet(sim_data_rgb)
    # print(f'\t ModifiedResnet Out (feature, seg): {resnet_out1.shape}, {resnet_out2.shape}')

    # pointnet = PointNetExtractor(feature_transform=True, C=C+3)
    # pointnet_feat, _, _ = pointnet(sim_data_pcd)    
    # print(f'\t PointNetExtractor Out (feat): {pointnet_feat.shape}')
    
    # model = DenseFusion(num_points=256)
    # fused = model(resnet_out1, pointnet_feat)
    # print(f'\t DenFusion Out: {fused.shape}')

    # mlp = MLP(cout=8*3)
    # out = mlp(fused)
    # print(f'\t MLP Out (KPOF): {out.shape}')

    # pvn3d = PVN3D(pcd_input_c=C)
    # kpof, rgbd_seg, ctof = pvn3d(sim_data_pcd, sim_data_rgb)
    # print(f'\t PVN3D Out: kpof, rgbd_seg, ctof: {kpof.shape, rgbd_seg.shape, ctof.shape}')


    ''' More complete test'''
    
    from utils.utils_data import *
    from lib.pvn3d import *
    import logging
    from datetime import datetime

    checkpoint = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # H, W = 360, 640     # Image height and width
    H, W = 120, 160
    transforms = {
        'rgb'  : Compose([Resize((H, W)), RandomHorizontalFlip(), ColorJitter()]),
        'depth': Compose([Resize((H, W))]),
        'label': Compose([Resize((H, W))])
    }
    p = {
        "device":device,                    
        'bz': 2, 'shuffle': False, 'num_workers':1,   # For loader TODO: Modify this
        'objects': 82, 'keypoints': 16, 'samplepoints': H*W,    # For PVN3D model
        "epochs": 1,  "lr": 1e-5, 'decay': 0,         # 2. For learning
        "scale":0.5,    "amp": False,                 # 3. Grad scaling
        "checkpoint": None
    }

    # 1. Initialize Network and Dataloader
    model = PVN3D(
        num_classes=p['objects'], pcd_input_c=3, num_kps=16, num_points=H*W
    )
    model.train()
    loader_train = get_loader(SegDataset(split_name='train', transforms=transforms, one_hot_label=False, N_samples=H*W), params=p)


    ''' ===== Logging ===== '''
    with open('./training_data/objects_present.pickle', 'rb') as f:
        OBJECTS = list(pickle.load(f))  # [(i_d1, name_1), ..., (id_n, name_n)]
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')
    logging.info(f'There are [{len(OBJECTS)}] objects in total')   # 23 objects
    logging.info(f'Network:\n'
                    f'\t{model.num_kps}     number of keypoints\n'
                    f'\t{model.num_points}  number of sample points\n'
                    f'\t{model.pcd_input_c} number of pcd augment channels\n'
                    f'\t{model.num_classes} output channels (classes)\n')
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))
        logging.info(f'Resuming from checkpoint {checkpoint}')
    ''' ===== End Logging ===== '''


    # 2. Train model, save checkpoint
    now = datetime.now().strftime("%d-%H:%M")
    epochs = p['epochs']
    try:        
        train_pvn3d(model, loader_train, p)
        fname = f'./exp/pvn3d_weight_epochs{epochs}_{now}.pt'
        torch.save(model.state_dict(), fname)
    except KeyboardInterrupt:
        fname = f'./exp/pvn3d_weight_INTERRUPTED_{now}_.pt'
        torch.save(model.state_dict(), fname)

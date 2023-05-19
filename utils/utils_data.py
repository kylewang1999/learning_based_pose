import os
import json
import pickle
import torch
import copy
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from open3d import io, geometry
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import *
import torch.nn.functional as F
from tqdm import tqdm
import utils.utils_bbox as utils

# FIXME: Can only do batch_size = 1 now. Since Different scenes have different number of objects.

class SegDataset(Dataset):

    def __init__(self, split_name='train', train=True, transforms=None, 
        one_hot_label=False, N_samples=720*1280, calc_offsets=False):
        '''
        Input: (from utils_icp.get_split_files())
            - rgb_files: List of dir entries 
            - label_files: List of dir entries
        '''
        expected_names = {
            'train': 'train', 'val': 'val',
            'val_small': 'val_small', 'val_tiny': 'val_tiny',
            'test': './testing_data',
            'test2': './testing_data_2',
            'test3': './testing_data_final'
        }
        assert split_name in expected_names.keys()


        if 'train' in split_name or 'val' in split_name:
            file_dict = get_split_files(expected_names[split_name])
        else:
            file_dict = get_test_files(expected_names[split_name])

        self.N_samples = N_samples
        self.rgb_files = file_dict['rgb_files']
        self.depth_files = file_dict['depth_files']

        self.label_files = None
        if 'label_files' in file_dict.keys():
            self.label_files = file_dict['label_files']
        self.label_files_unet = file_dict['label_files_unet']

        self.meta_files = file_dict['meta_files']
        self.suffixes = file_dict['suffixes']

        self.transforms = transforms
        self.one_hot_label = one_hot_label

        with open('./training_data/objects_present.pickle', 'rb') as f:  # [(i_d1, name_1), ..., (id_n, name_n)]
            OBJECTS = list(pickle.load(f))
            print(f'There are [{len(OBJECTS)}] objects in total')   # 23 objects

        self.OBJECTS = OBJECTS  # [(i_d1, name_1), ..., (id_n, name_n)]
        self.train = train
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.calc_offsets = calc_offsets

        print('---- Initializing Dataset ---- ')
        print(f'\t Length: {self.__len__()}')
        print(f'\t Sample points per scene: {N_samples}')
        print(f'\t Split: {split_name}')
        print(f'\t Train: {train}')
        print(f'\t transforms: {transforms}')


    def __len__(self):
        return len(self.rgb_files)


    def __getitem__(self, idx: int):
        
        ''' ---- Convert to Pytorch Tensors ---- '''
        to_tensor, expand_label = Compose([PILToTensor()]), Compose([InflateLabel()])
        
        rgb = to_tensor(Image.open(self.rgb_files[idx])) / 255      # convert 0-255 to 0-1 (3, 720, 1280)
        depth = to_tensor(Image.open(self.depth_files[idx])) / 1000 # convert from mm to m (1, 720, 1280)
        label = to_tensor(Image.open(self.label_files[idx])) if self.label_files is not None else None
        meta, suffix = self.meta_files[idx], self.suffixes[idx]     # Path to meta file, not loaded yet. # Suffix string

        if self.transforms is not None:
            if 'rgb' in self.transforms:
                rgb = self.transforms['rgb'](rgb)
            if 'depth' in self.transforms:
                depth = self.transforms['depth'](depth)
            if 'label' in self.transforms:
                label = self.transforms['label'](label)

        if self.one_hot_label:
            label = expand_label(label)        # (82, 720, 1280)
        label = label.to(dtype=torch.int64)

        # if suffix is None: print('suffix none!!!!')
        # if meta is None:   print('meta none!!!!')
        # if label is None: print('label none!!!!!')
        
        if label is None: sample = {'rgb': rgb, 'depth': depth, 'meta': meta, 'suffix': suffix}
        else: sample = {'rgb': rgb, 'label': label, 'depth': depth, 'meta': meta, 'suffix': suffix}
        
        if self.train and self.calc_offsets:
            print('\tCalculating KPOF, CTOF. Concat xyz,rgb features for pcd....')

            rgb_np = rgb.detach().numpy()
            dpt_np = depth.squeeze().detach().numpy()
            lab_np = label.squeeze().detach().numpy()
            meta_loaded = load_pickle(meta)
            # Get object pcds in world and canonical frame
            pcd_camera = img_to_camera(
                dpt_np, rgb_np, meta_loaded['intrinsic'], view=False)   # (n,3)
            pcd_world = camera_to_world(pcd_camera, meta_loaded['extrinsic'])


            objects = crop_obj(pcd_world, lab_np, meta_loaded, train=True)
            objs_world     = [obj[0] for obj in objects]
            objs_canonical = [obj[1] for obj in objects]
            objs_mask      = [obj[2] for obj in objects]
            objs_pose      = [obj[3] for obj in objects]
            objs_id        = [obj[4] for obj in objects]
            objs_name      = [obj[5] for obj in objects]

            # print('plotting')
            # plot_pcds(objs_world)
            # plot_pcds(objs_canonical)


            # print('plot pcd')
            # plot_pcd(pcd_world, ax_lim=[[-1,1], [0,1], [-1,1]])

            cls_ids = meta_loaded['object_ids'].flatten().astype(np.uint32)
            cls_names = meta_loaded['object_names']
            n_objects, n_keypoints, n_sample_points = len(cls_ids), 16, self.N_samples

            poses   = np.zeros((n_objects, 4, 4))
            kps_gt  = np.zeros((n_objects, n_keypoints, 3))         # Keypoints in world frame
            ctr_gt  = np.zeros((n_objects, 3))                      # Center    in world frame
            kpof_gt = np.zeros((n_sample_points, n_keypoints, 3))
            ctof_gt = np.zeros((n_sample_points, 3))
            

            for i, (obj_world, obj_canonical, obj_mask, pose, id, name) in \
                enumerate(zip(objs_world, objs_canonical, objs_mask, objs_pose, objs_id, objs_name)):

                poses[i,:,:] = pose
                R, t = pose[:3,:3], pose[:3,3]

                ''' keypoints and center in world frame'''
                kps_canonical = np.load(f'/home/kyle/Desktop/cse291.1_dl3d/cse291.1_hw3/models/{name}/visual_meshes/keypoints_16.npy')
                ctr_canonical = np.load(f'/home/kyle/Desktop/cse291.1_dl3d/cse291.1_hw3/models/{name}/visual_meshes/center.npy').flatten()
                kps_world = kps_canonical @ R.T + t  # world frame
                ctr_world = ctr_canonical @ R.T + t  # world frame

                kps_gt[i,:,:] = kps_world
                ctr_gt[i,:]   = ctr_world

                ''' calculate gt ctof, kpof '''
                _kpof_gt = calc_kpof(obj_world, kps_world)   # (N_sample_obj, N_kp, 3)      #FIXME: Make sure mask & kpof are in the same sequential order
                _ctof_gt = calc_ctof(obj_world, ctr_world)   # (N_sample_obj, 3)
                kpof_gt[obj_mask[:,0],:,:] = _kpof_gt
                ctof_gt[obj_mask[:,0],:] = _ctof_gt
            
            # Augment pcd with rgb features
            pcd_rgb = torch.cat([torch.from_numpy(pcd_world),
                                rgb.permute(1,2,0).contiguous().reshape(-1,3)], axis=1)

            # sample['poses']   = poses     # Don't theses, leads to ragged Tensor to be collated
            # sample['kps_gt']  = kps_gt 
            # sample['ctr_gt']  = ctr_gt 
            sample['kpof_gt'] = torch.from_numpy(kpof_gt.astype(np.float32))
            sample['ctof_gt'] = torch.from_numpy(ctof_gt.astype(np.float32))
            sample['pcd_rgb'] = pcd_rgb.to(dtype=torch.float32)
        
        return sample   # Additional keys: 'rgb', 'label', 'depth', 'meta', 'suffix'


''' ---- KPOF, CTOF Helper '''

def calc_ctof(pcd, ctr):
    ''' Calculate for each point in pcd, the offset to center'''
    pcd = pcd.reshape(-1, 3)    # (N,3)
    ctr = ctr.reshape(-1, 3)    # (1,3)
    offset = pcd - ctr
    return offset               # (N,3)


def calc_kpof(pcd, kps):
    ''' Calculate for each point in pcd, the offset to each keypoint'''
    pcd = pcd.reshape(-1, 3)
    kps = kps.reshape(-1, 3)
    kp_offsets = np.array([pcd - kp for kp in kps])
    return kp_offsets.transpose(1,0,2)  # (N_sample, N_kp, 3)
    

def crop_obj(pcd_world, label, meta, train=False):
    '''Given 1 scene, crop out each object in canonical frame. 
    Input
        - pcd_world : np.array (n,3). Points in the world frame.
        - labe      : np.array (H,W). Segmented img
        - meta      : loaded pickel object. Metadata
    Output
        - List[tuples]: Each tuple is either
            (obj_pcd_world, obj_pcd_canonical, id, name) for training data
            (obj_pcd_world,                    id, name) for testing data
    '''
    assert pcd_world.shape[0]==label.shape[0]*label.shape[1], \
        f'pcd, label should have same number of points, but pcd: {pcd_world.shape}, label{label.shape}'
    if train:
        obj_ids, obj_names, obj_poses = meta['object_ids'], meta['object_names'], meta['poses_world']
        objects = []
        for id, name in zip(obj_ids, obj_names):
            obj_mask = np.transpose(np.nonzero(label.reshape(-1) == id))
            obj_pcd_world = pcd_world[obj_mask[:,0]]
            pose = obj_poses[id]
            obj_pcd_canonical = world_to_canonical(obj_pcd_world, pose)
            objects.append((obj_pcd_world, obj_pcd_canonical, obj_mask, pose, id, name))    # mask (H*W, 1)
    else:
        obj_ids, obj_names = meta['object_ids'], meta['object_names']
        objects = []
        for id, name in zip(obj_ids, obj_names):
            obj_pcd_world = pcd_world[np.nonzero(label.reshape(-1) == id)]
            objects.append((obj_pcd_world, id, name))

    return objects



class InflateLabel(object):

    def __call__(self, label):
        ''' Inflate number of channels of a segmentation mask (i.e. label)
        Input:
            - label: Tensor (1, H, W). min=0, max=82. Segmentation mask.
        Output
            - label: Tensor (82, H, W). Each channel is 0/1-valued segmentation mask.
        '''
        num_classes = torch.max(label)+1

        label_inflated = []
        for i in range(num_classes):
            label_inflated.append((label[0, :, :] == i).long())

        label_inflated = torch.stack(label_inflated)
        # foo = F.one_hot(label.long(), num_classes).permute(0, 3, 1, 2)
        # print(f"foo: {foo.dtype}, {foo.shape}, {torch.max(foo)}")
        return label_inflated


class DeflateLabel(object):

    def __call__(self, label):
        ''' Inflate number of channels of a segmentation mask (i.e. label)
        Input:
            - label: Tensor (82, H, W). Each channel is 0/1-valued segmentation mask.
        Output:
            - label: Tensor (1, H, W). min=0, max=82. Segmentation mask.
        '''
        # print(f'  deflate: {label.shape}')

        if len(label.shape) == 4:
            label = label.squeeze(0)

        num_classes = label.shape[0]
        label_deflated = torch.zeros(
            (1, label.shape[1], label.shape[2]), dtype=torch.uint8)

        for c in range(num_classes):
            label_deflated[0, label[c, :, :] != 0] = c

        return label_deflated


def get_loader(dataset, params):
    loader = DataLoader(dataset,
                        batch_size=params['bz'], shuffle=params['shuffle'],
                        num_workers=params['num_workers'])
    return loader




''' ---- [READ] Dataset helpers ----'''


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_split_files(split_name):
    '''
    Input:
        - split_name: "train"(n=50000), "val" (n=2800), "val_small", "val_tiny"
    Output:
        - rgb   : List (n,) of png files. RGB images (720, 1280, 3)
        - depth : List (n,) of png files. Depth maps (720, 1280) 
        - label : List (n,) of png files. Segmented images (720, 1280) 
        - meta  : List (n,) of pkl files. Metadata. Can be looked up using .keys() after loading.
            {
                'poses_world': List. Each entry is (4,4) object pose in the world frame, `None` if object nonexistent in scene 
                'extents','scales': List. Each element <-> 1 object in the scene. extent * scale = box size
                    - `extent`: (3,) array. Size of each obj. in its canonical frame
                    - `scale`: (3,) array. Scale of each obj.
                'object_names: List of objects present in the scene
                'object_ids': List of object ids present in the scene
                'extrinsic': (4,4) matrix [world -> viewer(opencv)]
                'intrinsic': (3,3) matrix [viewer(opencv) -> image]
            }
        - suffixes: List (n,). Scene suffixes (e.g. "1-1-1")
    '''
    training_data_dir = "./training_data/v2.2"  # TODO: Change these?
    split_dir = "./training_data/splits/v2"

    assert split_name in ['train', 'val', 'val_small', 'val_tiny']

    with open(os.path.join(split_dir, f"{split_name}.txt"), 'r') as f:
        prefix = [os.path.join(training_data_dir, line.strip())
                  for line in f if line.strip()]

        file_dict = {
            'rgb_files': [p + "_color_kinect.png" for p in prefix],
            'depth_files': [p + "_depth_kinect.png" for p in prefix],
            'label_files': [p + "_label_kinect.png" for p in prefix],
            'label_files_unet': [p + "_label_unet.png" for p in prefix],
            'meta_files': [p + "_meta.pkl" for p in prefix],
            'suffixes': [p.split('/')[-1] for p in prefix]
        }

    return file_dict


def get_test_files(data_dir='./testing_data'):
    assert data_dir in ['./testing_data', './testing_data_2', './testing_data_final']

    with open(f'{data_dir}/test.txt', 'r') as f:
        prefix = [os.path.join(f'{data_dir}/v2.2', line.strip())
                  for line in f if line.strip()]
        file_dict = {
            'rgb_files': [p + "_color_kinect.png" for p in prefix],
            'depth_files': [p + "_depth_kinect.png" for p in prefix],
            'label_files' : [p + "_label_kinect.png" for p in prefix],
            'label_files_unet': [p + "_label_unet.png" for p in prefix],
            'meta_files': [p + "_meta.pkl" for p in prefix],
            'suffixes': [p.split('/')[-1] for p in prefix]
        }
    return file_dict


# def load_files(rgb, depth, label, label_unet, meta):
def load_files(rgb, depth, label, meta):
    ''' Given paths to files, return loaded np.array or pickel'''
    rgb = np.array(Image.open(rgb)) / \
        255        # convert 0-255 to 0-1 (720, 1280, 3)
    # convert from mm to m (720, 1280)
    depth = np.array(Image.open(depth)) / 1000
    label = np.array(Image.open(label))          # (720, 1280)
    # label_unet = np.array(Image.open(label_unet))          # (720, 1280)
    meta = load_pickle(meta)

    # return rgb, depth, label, label_unet, meta
    return rgb, depth, label, meta


''' ---- Image Visualization ---- '''


def color_palette(num_objects=79):
    ''' Return 82-way colored palette for visualization
        https://matplotlib.org/stable/tutorials/colors/colormaps.html
    Output:
        - COLOR_PALETTE: np.array (82, 3)
    '''
    from matplotlib.cm import get_cmap
    cmap = get_cmap('rainbow', num_objects)
    COLOR_PALETTE = np.array([cmap(i)[:3]
                              for i in range(num_objects + 3)])  # (82, 3)
    COLOR_PALETTE = np.array(COLOR_PALETTE * 255, dtype=np.uint8)

    np.random.seed(10)
    np.random.shuffle(COLOR_PALETTE)
    COLOR_PALETTE[-3] = [119, 135, 150]  # Table
    COLOR_PALETTE[-2] = [176, 194, 216]  # Ground
    COLOR_PALETTE[-1] = [255, 255, 225]  # Robot

    return COLOR_PALETTE


def visualize_one_scene(rgb, depth, label, meta, suffix=None, verbose=False):

    if isinstance(meta, list):
        meta = meta[0]

    if isinstance(meta, str):
        meta = load_pickle(meta)

    if isinstance(rgb, torch.Tensor):
        rgb = rgb.squeeze()
        rgb = rgb.permute(1, 2, 0).detach().numpy()

    if isinstance(depth, torch.Tensor):
        if len(depth.shape) == 4:
            depth = depth.squeeze(0)
        depth = torch.squeeze(depth.permute(1, 2, 0), -1).detach().numpy()

    if isinstance(label, torch.Tensor):

        print(f"in visualize: label {label.shape}")
        if len(label.shape) > 2:
            deflation = DeflateLabel()
            label = deflation(label)
            label = torch.squeeze(label.permute(1, 2, 0), -1)
        label = label.detach().numpy()

    # COLOR_PALETTE = _color_palette(np.unique(label).shape[0])
    COLOR_PALETTE = color_palette()

    if verbose:
        print(type(rgb), rgb.dtype, rgb.shape, np.max(rgb))         # Max 1.0
        print(type(depth), depth.dtype, depth.shape, np.max(depth))  # Max 6.006
        print(type(label), label.dtype, label.shape, np.max(label))  # Max 81

    rgb = np.array(rgb, dtype='float64', order='C')
    depth = np.array(depth, dtype='float64', order='C')
    label = np.array(label, dtype='uint8', order='C')

    ''' 1. Print objects and ids present in this scene'''
    name_and_id = np.column_stack(
        (np.array(meta['object_ids']), np.array(meta['object_names'])))
    print(name_and_id, f'Scene: {suffix}')

    ''' 2. Get bbox-ed image '''
    if "poses_world" in meta.keys():
        poses_world, box_sizes = get_poses_world(meta), get_box_sizes(meta)
        rgb_boxed = draw_bbox(rgb, poses_world, box_sizes,
                              meta['extrinsic'], meta['intrinsic'])

        plt.figure(figsize=(40, 20))
        plt.subplot(2, 2, 1)
        plt.imshow(rgb)
        plt.subplot(2, 2, 2)
        plt.imshow(depth)
        plt.subplot(2, 2, 3)
        plt.imshow(COLOR_PALETTE[label])  # draw colorful segmentation
        plt.subplot(2, 2, 4)
        plt.imshow(rgb_boxed)
    else:
        plt.figure(figsize=(30, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(rgb)
        plt.subplot(1, 3, 2)
        plt.imshow(depth)
        plt.subplot(1, 3, 3)
        plt.imshow(COLOR_PALETTE[label])  # draw colorful segmentation


def visualize_training_data(rgb, depth, label, meta, suffix=None, verbose=False):
    '''Visualize [rgb, depth, labels, rgb_boxed] for 1 batch.
    Input:
        - rgb, depth, label: Tensor of ([B]atch, [C]hannel, [H]eight, [W]idth)
        - meta             : List of String. Path to meta file
        - meta             : List of String. Suffixes
    '''
    if len(rgb.shape) == 4:     # Input is a batch

        for batch_i in range(rgb.shape[0]):
            _rgb = rgb[batch_i]
            _depth = depth[batch_i]
            _label = label[batch_i]
            _meta = meta[batch_i]
            _suffix = suffix[batch_i]
            visualize_one_scene(_rgb, _depth, _label, _meta, _suffix, verbose)

    else:                       # Input is a single sample
        visualize_one_scene(rgb, depth, label, meta, suffix, verbose)


def mask_to_png(mask, path='foo.png'):
    '''Save segmentation mask (torch tensor) as png image'''
    assert mask.shape[
        0] == 1, f'Image tensor should be of shape (1, H, W), but got {mask.shape}'
    from PIL import Image
    mask = torch.squeeze(mask.permute(1, 2, 0), -1)
    mask = mask.detach().numpy()
    mask = np.array(mask, dtype='uint8', order='C')

    im = Image.fromarray(mask)
    im.save(path)


''' ---- Pose and Bbox ----'''

def get_poses_world(meta):
    poses_world = np.array([meta['poses_world'][idx]
                           for idx in meta['object_ids']])
    return poses_world


def get_box_sizes(meta):
    box_sizes = np.array([meta['extents'][idx] * meta['scales'][idx]
                         for idx in meta['object_ids']])
    return box_sizes


def draw_bbox(rgb, poses_world, box_sizes, extrinsic, intrinsic):
    '''Given [rgb img, poses_world, box_sizes], draw bbox for each object'''

    rgb = copy.deepcopy(rgb)
    for i in range(len(poses_world)):
        utils.draw_projected_box3d(
            rgb, poses_world[i][:3, 3], box_sizes[i],   # img, center, size
            poses_world[i][:3, :3],                             # rotation
            extrinsic, intrinsic,
            thickness=4)
    return rgb


''' ---- .dae Canonical Object Model Helpers ---- '''


def iterative_farthest_sampling(points, n=4e3):
    points = np.array(points)

    '''1. Add a random point to the farthest point set S'''
    farthest_set = [points[0]]

    '''2. Iteratively find the point w/ largest distance to S, then add to S'''
    farthest_set = np.zeros((int(n), 3))
    distances = np.inf * np.ones(len(points))

    for i in tqdm(range(int(n))):
        farthest_set[i] = points[np.argmax(distances)]
        _dist = np.linalg.norm((points-farthest_set[i]), axis=1)
        distances = np.minimum(_dist, distances)
    return farthest_set


''' --- [WRITE]  .json pose estimation result ---'''


def save_preds(preds, name="foo"):
    with open(f'{name}.json', 'w') as f:
        json.dump(preds, f, indent=1)


def append_preds(new_preds, name="foo"):

    if os.path.exists(f"{name}.json"):

        with open(f"{name}.json", 'r') as f:
            preds = json.load(f)

        preds.update(new_preds)

        with open(f"{name}.json", 'w') as f:
            json.dump(preds, f)

    else:
        with open(f"{name}.json", 'w') as f:
            json.dump(new_preds, f)


''' --- Perspective Transformations --- '''

def img_to_camera(depth, rgb, intrinsic, view=False):
    ''' Given (720, 1280) depth map and `pickel` meta, return (and visualize) point cloud in camera frame
        Note: This is a 2D-3D mapping.
    Input
        - depth: np.array (H,W). Depth map
        - meta : pickle meta file
    Output
        - pcd_viewer  : np.array (H*W, 3). Point cloud in the viewer (camera) frame.
    '''
    if isinstance(depth, torch.Tensor) and len(depth.shape) == 4:
        depth = depth.squeeze().detach().numpy()
    if isinstance(rgb, torch.Tensor) and len(rgb.shape) == 4:
        rgb = rgb.squeeze(0).detach().numpy()
    z = depth
    y_scale = 720 // z.shape[0]
    x_scale = 1280 // z.shape[1]

    v, u = np.indices(z.shape)  # v, u of shape (720,1280) max(v)=719, max(u)=1279
    uv1 = np.stack([(u+0.5)*x_scale, (v+0.5)*y_scale, np.ones_like(z)], axis=-1)
    pcd_viewer = uv1 @ np.linalg.inv(np.array(intrinsic)).T * z[..., None] 
    pcd_viewer = pcd_viewer.reshape((-1, 3))

    ''' Convert pcd_viewer to open3d point cloud, only for visualization'''
    pcd_o3d = pcd_to_o3d_rgb(pcd_viewer, rgb)
    if view:
        # will open another window to show the point cloud
        o3d.visualization.draw_geometries([pcd_o3d])

    return pcd_viewer


def camera_to_world(pcd_camera, extrinsic):
    ''' Apply inverse of `extrinsic` to `pcd_camera`
     Input:
        - pcd_camera: np.array(n, 3). Point cloud coordinates in camera frame
        - extrinsic: np.array(4, 4). Extrinsic matrix
    Output:
        - pcd_world: np.array(n, 3). Point cloud coordinates in world frame
    '''

    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    pcd_world = (pcd_camera - t) @ R
    return pcd_world


def world_to_canonical(pcd_world, pose_world):
    '''  Apply inverse of `extrinsic` to `pcd_world`
    Input:
        - pcd_world: np.array(n, 3). Point cloud coordinates in world frame
        - pose_world: np.array(4, 4). point cloud pose in world frame
    Output:
        - pcd_canonical: np.array(n, 3). Point cloud coordinates in canonical frame
    '''
    R, t = pose_world[:3, :3], pose_world[:3, 3]
    pcd_canonical = (pcd_world - t) @ R
    return pcd_canonical




class SampleData:
    def __init__(self):
        prefix = '/home/kyle/Desktop/cse291.1_dl3d/cse291.1_hw3/training_data'
        paths = {
            'rgb': f'{prefix}/v2.2/1-1-0_color_kinect.png',
            'dpt': f'{prefix}/v2.2/1-1-0_depth_kinect.png',
            'label': f'{prefix}/v2.2/1-1-0_label_kinect.png',
            'meta': f'{prefix}/v2.2/1-1-0_meta.pkl'
        }
        self.rgb = np.array(Image.open(paths['rgb'])) / 255
        self.dpt = np.array(Image.open(paths['dpt'])) / 1000
        self.label = np.array(Image.open(paths['label']))

        with open(paths['meta'], 'rb') as f:
            meta = pickle.load(f)
        self.meta = meta


def plot_img_2d(img, path='foo.png'):
    plt.imshow(img)
    plt.savefig(path)


''' ---- ICP helpers ---- '''


def compare_points(points1, points2, iter=0, view=False):
    if not isinstance(points1, np.ndarray):
        points1 = np.asarray(points1.points)    # (16384*3)
        points2 = np.asarray(points2.points)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # For pose estimation dataset
    # ax.set_xlim3d([-0.5, 0.5])
    # ax.set_ylim3d([-0.25, 0.25])
    # ax.set_zlim3d([-0.25, 0.5])

    # For banana:
    ax.set_xlim3d([-0.5, 1.5]), ax.set_ylim3d([0, 1.5]), ax.set_zlim3d([0, 2])

    ax.scatter(points1[:, 0], points1[:, 2], points1[:, 1],
               marker='.', alpha=0.01, edgecolors='none')
    ax.scatter(points2[:, 0], points2[:, 2], points2[:, 1],
               marker='.', alpha=0.01, edgecolors='none')

    if view:
        plt.savefig(f"./imgs/compare_points/{iter}.png")
        plt.show()


def transformation_error(pred, gt):
    ''' Given predicted and groud truth 4-by-4 transformation matrices, compute their differences
    Input:
        - pred: Predicted 4*4 transformation matrix
        - gt:   Ground truth 4*4 transformation matrix
    Output:
        - err_rot: Float. Rotation error (upper-left 3*3 sub-matrix)
        - err_tra: Float. Translation error (right-most column of the transformation matrix)
    '''
    # diff = gt - pred
    # err_rot = np.linalg.norm(diff[:3,:3])
    # err_tra = np.linalg.norm(diff[:,3])

    R_pred, R_gt = pred[:3, :3], gt[:3, :3]
    t_pred, t_gt = pred[:3, 3], gt[:3, 3]

    # err_rot = np.rad2deg(np.arccos(\
    #     np.clip(0.5 * (np.trace(R_pred.T @ R_gt) - 1), -1.0, 1.0)))
    err_rot = np.arccos(
        np.clip(0.5 * (np.trace(R_pred.T @ R_gt) - 1), -1.0, 1.0))
    err_tra = np.linalg.norm(t_pred - t_gt)

    return err_rot, err_tra


def find_nn_corr(src, tgt):
    ''' Given two input point clouds, find nearest-neighbor correspondence (from source to target) 
    Input:
        - src: Source point cloud (n*3), either array or open3d pcd
        - tgt: Target point cloud (n*3), either array or open3d pcd
    Output:
        - idxs: Array indices corresponds to src points, 
            array elements corresponds to nn in tgt points (n, np.array)
    '''

    ''' Way1: Sklearn'''
    if not isinstance(src, np.ndarray):
        src = np.asarray(src.points)    # (16384*3)
        tgt = np.asarray(tgt.points)

    neighbors = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(tgt)
    dists, idxs = neighbors.kneighbors(src)  # (16384*1), (16384*1)
    return idxs.flatten()


def align(src, tgt, corr):
    ''' Given source, target point clouds and their correspondence, 
            compute transformation matrix that best aligns the two
        The notation follows this ppt: https://haosulab.github.io/ml-meets-geometry/WI21/Lectures/L11_6D_Pose_Estimation.pdf
    Input:
        - src: Source point cloud (n, 3), either array or open3d pcd
        - tgt: Target point cloud (k, 3), either array or open3d pcd
        - corr: Correspondence    (n, )
        - R: Initial rotation matrix     (3, 3)
        - t: Initial translation  vector (3,)
    Output: 
        - R: Transformed rotation matrix (3, 3)
        - t: Transformed translation vector (3,)
        - trans_pred: Augmented matrix containing R,t (4,4)
    '''
    if not isinstance(src, np.ndarray):
        src = np.asarray(src.points)    # (16384*3)
        tgt = np.asarray(tgt.points)

    # 0. Target should be src's nn correspondences
    tgt = tgt[corr]

    # 1. Produce mean-subtracted P, Q matrices from src, tgt
    src_mean, tgt_mean = np.mean(src, axis=0), np.mean(tgt, axis=0)     # (3,)
    src = src - src_mean   # (n, 3)
    tgt = tgt - tgt_mean   # (n, 3)
    P, Q = src.T, tgt.T    # (3, n)

    # 2. Perform singular value decomposition
    M = Q @ P.T    # (3, 3) = (3, n) * (n, 3)
    U, S, Vh = np.linalg.svd(M)

    # 3. Compute rotation from SVD result
    R_ = U @ Vh
    if np.linalg.det(R_) == -1:
        Vh[-1, :] = -Vh[-1, :]
        R_ = U @ Vh

    # 4. Compute translation
    t_ = tgt_mean - (R_ @ src_mean)

    # 5. Update initial R, t
    # R = R_ @ R
    # t = R_ @ t  + t_

    trans_pred_ = np.eye(4)
    trans_pred_[:3, :3] = R_
    trans_pred_[:3, -1] = t_
    return R_, t_, trans_pred_


def init_transformation(src, tgt, init_trans=None,  translate_only=False, voxel_size=0.02):
    ''' Given source, target point clouds, perform global registration using RANSAC. 
        http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
        FIXME: Why putting the method before icp changes result?
    Input:
        - src, tgt: Open3d point clouds
    Output:
        - transformation: (4,4) transformation matrix
    '''
    def _get_fpfh(pcd, voxel_size, down_sample=False):
        '''Extract FPFH features from point cloud: http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html#Extract-geometric-feature'''

        # 1. Down sample pcd, estimate normal using search radius = 2*voxel_size
        if down_sample:
            pcd = pcd.voxel_down_sample(voxel_size)

        pcd.estimate_normals(geometry.KDTreeSearchParamHybrid(
            radius=voxel_size, max_nn=30))

        # 2. Compte FPFH feature with radius = 5*voxel_size
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size*5, max_nn=100)
        )
        return pcd, pcd_fpfh

    def _init_translation(src, tgt, init_trans):

        rotation = init_trans
        rotation[:3, 3] = np.array([0, 0, 0])  # remove translation component
        src = copy.deepcopy(src).transform(rotation)
        translation = tgt.get_center() - src.get_center()
        return translation

    '''Source/target pcd is void. This might be cause by occlusion.'''
    if len(src.points) == 0 or len(tgt.points) == 0:    #
        if init_trans is not None:
            return init_trans
        else:
            return np.eye(4)

    if init_trans is not None:

        ''' Global Registration (slow and inferior to lego pose)'''
        # src = copy.deepcopy(src)
        # tgt = copy.deepcopy(tgt)

        # src, src_fpfh = _get_fpfh(src, voxel_size)
        # tgt, tgt_fpfh = _get_fpfh(tgt, voxel_size)  # Down sampling target lego might cause void pcd

        # # RANSAC, distance threshold = 1.5*voxel size
        # dist_thresh = voxel_size * 1.5
        # ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        #     src, tgt, src_fpfh, tgt_fpfh, True, dist_thresh,
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint(
        #         False), 3,
        #     [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
        #             dist_thresh)
        #     ], o3d.pipelines.registration.RANSACConvergenceCriteria(100, 0.999))

        trans = np.eye(4)
        # Rotation is better initialized with initial pose from lego
        trans[:3, :3] = init_trans[:3, :3]

        # trans[:3,3] = ransac_result.transformation[:3,3]  # Translation also could get from global reg.
        # trans[:3,3] = _init_translation(src, tgt, init_trans)  # Translation is better initialized using global registration
        trans[:3, 3] = init_trans[:3, 3]

        return trans
    else:
        src = copy.deepcopy(src)
        tgt = copy.deepcopy(tgt)

        src, src_fpfh = _get_fpfh(src, voxel_size)
        # Down sampling target lego might cause void pcd
        tgt, tgt_fpfh = _get_fpfh(tgt, voxel_size)

        # RANSAC, distance threshold = 1.5*voxel size
        dist_thresh = voxel_size * 1.5
        ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src, tgt, src_fpfh, tgt_fpfh, True, dist_thresh,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    dist_thresh)
             ], o3d.pipelines.registration.RANSACConvergenceCriteria(100, 0.999))
        return ransac_result.transformation


def pcd_to_o3d(pcd):
    ''' Convert np array (n,3) to open3d pcd'''
    points = o3d.utility.Vector3dVector(pcd.reshape([-1, 3]))
    points_o3d = o3d.geometry.PointCloud()
    points_o3d.points = points
    return points_o3d


def pcd_to_o3d_rgb(pcd, rgb):
    ''' Convert np array (n,3) to open3d pcd'''
    points = o3d.utility.Vector3dVector(pcd.reshape([-1, 3]))
    colors = o3d.utility.Vector3dVector(
        rgb.reshape([-1, 3]))  # TODO: What's this
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = points
    pcd_o3d.colors = colors
    return pcd_o3d


def icp(src, tgt, init_trans=None, iters=100, downsample_fraction=1, thresh=0.005, verbose=False):
    '''(self-implemented) Iterative closest point to align src and tgt point clouds
    Input:
        - src, tgt: Open3d point clouds
    '''

    if isinstance(src, np.ndarray):
        src = pcd_to_o3d(src)
        tgt = pcd_to_o3d(tgt)

    if len(src.points) == 0 or len(tgt.points) == 0:
        return np.eye(4), 0.0

    ''' Initialized `R, t, trans_pred` '''
    trans_pred = init_transformation(
        src, tgt, init_trans=init_trans, translate_only=True)

    R, t = trans_pred[:3, :3], trans_pred[:3, -1]

    for i in (range(iters)):

        src_ = copy.deepcopy(src).transform(trans_pred)  # Transform
        corr = find_nn_corr(src_, tgt)                  # Find correspondence
        R_, t_, trans_pred_ = align(src_, tgt, corr)    # Align

        if (np.linalg.norm(trans_pred-trans_pred_) < 1e-8):
            break

        R = R_ @ R              # Update `R, t, trans_pred`
        t = R_ @ t + t_
        trans_pred = np.eye(4)
        trans_pred[:3, :3] = R
        trans_pred[:3, -1] = t

        if verbose and (i < 10 or i % (iters//10)) == 0:
            compare_points(src_, tgt, iter=i, view=True)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        src, tgt, thresh, trans_pred, o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500))

    src = src.transform(trans_pred)
    return trans_pred, reg_p2p.fitness


def icp_o3d(src, tgt, init_trans=None, downsample_fraction=1, thresh=0.001):
    '''http://www.open3d.org/docs/latest/tutorial/pipelines/icp_registration.html#Point-to-point-ICP
    Input:
        - src, tgt: Source, target pcd
        - downsample_fraction: The fraction of src, tgt points used for ICP
    Output:
        - trans: Transformation output by ICP
        - fitness: Overlapping area (http://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.RegistrationResult.html)
    '''

    if isinstance(src, np.ndarray):
        src = pcd_to_o3d(src)
        tgt = pcd_to_o3d(tgt)

    if len(src.points) == 0 or len(tgt.points) == 0:
        if init_trans is not None:
            return init_trans, 0, np.inf
        else:
            return np.eye(4), 0, np.inf

    src.points = src.points[::downsample_fraction]
    tgt.points = tgt.points[::downsample_fraction]

    ''' Initialized `R, t, trans_pred` '''
    trans_init = init_transformation(src, tgt, init_trans=init_trans)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        src, tgt, thresh, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000))

    trans, fitness, rmse = reg_p2p.transformation, reg_p2p.fitness, reg_p2p.inlier_rmse

    src = src.transform(trans)
    return trans, fitness, rmse


def icp_one_scene_lego(OBJ_LEGO, rgb, depth, label, meta,
                       icp_option='o3d', view=False, train_set_fraction=1, icp_thresh=0.001):
    '''
    Input
        - OBJ_LEGO: Dict{(id,name): [(lego1, pose1), ...]}. Canonical legos and poses of objects
        - rgb, depth, label, meta
    Output
        - poses_world: List[79]. Poses for every object in the scene. 
    '''

    pcd_camera = img_to_camera(
        depth, rgb, meta['intrinsic'], view=False)   # (n,3)
    pcd_world = camera_to_world(pcd_camera, meta['extrinsic'])

    objects = crop_obj(pcd_world, label, meta, train=False)
    poses_world = [None] * 79

    ''' For each obj'''
    for (obj_world, id, name) in objects:

        ''' For each training lego: do ICP and get fitness'''
        lego_poses, lego_fitnesses, lego_rmse, legos = [], [], [], []
        for i, (lego_canonical, lego_pose) in enumerate(OBJ_LEGO[(id, name)][::train_set_fraction]):
            if icp_option == 'o3d':
                pose, fitness, rmse = icp_o3d(
                    lego_canonical, obj_world, init_trans=lego_pose, thresh=icp_thresh, downsample_fraction=1)
            else:
                pose, fitness = icp(lego_canonical, obj_world,
                                    init_trans=lego_pose, iters=500)
            lego_poses.append(pose)
            lego_fitnesses.append(fitness)
            legos.append(lego_canonical)
            if rmse:
                lego_rmse.append(rmse)

        lego_poses, lego_fitnesses, lego_rmse = np.array(
            lego_poses), np.array(lego_fitnesses), np.array(lego_rmse)

        best_lego = legos[np.argmax(lego_fitnesses)]
        best_pose = lego_poses[np.argmax(lego_fitnesses)]

        '''Self-implemented ICP has better accuracy. So finally perform it.'''
        best_pose, best_fitness = icp(
            best_lego, obj_world, init_trans=best_pose, iters=1000)
        poses_world[id] = best_pose.tolist()

    meta['poses_world'] = poses_world

    if view:
        print(
            f'In icp_one_scene_leg. Shapes: rgb {rgb.shape}, depth {depth.shape}, label {label.shape}')
        visualize_one_scene(rgb, depth, label, meta)
    return poses_world


def icp_one_scene(MEAN_OBJS, rgb, depth, label, meta, iters=1000, view=False):
    '''
    Input
        - MEAN_OBJS: Dict{(obj_id, obj_name): np.array()}. Canonical point cloud of objects
        - rgb, depth, label, meta
    Output
        - poses_world: List[79]. Poses for every object in the scene. 
    '''

    pcd_camera = img_to_camera(
        depth, rgb, meta['intrinsic'], view=False)   # (n,3)
    pcd_world = camera_to_world(pcd_camera, meta['extrinsic'])

    ''' ICP on each object in the scene '''
    objects = crop_obj(pcd_world, label, meta, train=False)

    poses_world = [None] * 79                               # NUM_OBJECTS = 79
    for (obj_world, id, name) in objects:
        ''' 1. Points in world vs. canonical frame before ICP'''
        obj_canonical = MEAN_OBJS[(id, name)][::500]
        obj_canonical = pcd_to_o3d(obj_canonical)
        obj_world = pcd_to_o3d(obj_world)
        if view:
            # Compare scr, tgt before icp
            compare_points(obj_canonical, obj_world)

        ''' 2. Do ICP'''
        pose_pred, fitness, rmse = icp_o3d(obj_canonical, obj_world)
        poses_world[id] = pose_pred.tolist()

        if view:
            # Compare scr, tgt after icp
            compare_points(obj_canonical, obj_world)

    return poses_world


''' --- Build mean object. TODO: This is likely trash. ----'''


def get_mean_objects(rgb_files, depth_files, label_files, meta_files):

    MEAN_OBJECTS = {}

    ''' For each training scene '''
    # Meta keys: ['poses_world', 'extents', 'scales', 'object_ids', 'object_names', 'extrinsic', 'intrinsic']
    with tqdm(total=len(rgb_files)) as pbar:
        for i, (rgb, depth, label, meta) in tqdm(enumerate(
                zip(rgb_files, depth_files, label_files, meta_files))):

            rgb, depth, label, meta = load_files(rgb, depth, label, meta)

            pcd_camera = img_to_camera(
                depth, rgb, meta['intrinsic'], view=False)   # (n,3)
            pcd_world = camera_to_world(pcd_camera, meta['extrinsic'])
            objects = crop_obj(pcd_world, label, meta, train=True)

            for (obj_world, obj_canonical, id, name) in objects:
                if (id, name) in MEAN_OBJECTS.keys():
                    old_obj = MEAN_OBJECTS[(id, name)]
                    MEAN_OBJECTS[(id, name)] = np.concatenate(
                        [old_obj, obj_canonical])
                else:
                    MEAN_OBJECTS[(id, name)] = obj_canonical

            pbar.update(1)

    with open('objects_mean.pickle', 'wb') as handle:
        pickle.dump(MEAN_OBJECTS, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return MEAN_OBJECTS


def get_lego_poses(loader, mask_option='unet', fname='./training_data/lego_poses_unte.pickle'):

    print(f'Producing legos according to [{mask_option}] segmentation mask...')
    OBJ_LEGO = {}

    for sample in tqdm(loader):
        rgb, depth, label, label_unet, meta, scene = \
            sample['rgb'], sample['depth'], sample['label'], sample['label_unet'], sample['meta'], sample['suffix']
        meta = load_pickle(meta[0])

        deflate = DeflateLabel()
        if mask_option == 'unet':
            mask = deflate(label_unet)
        else:
            mask = deflate(label)

        rgb = rgb.squeeze().detach().numpy()
        depth = depth.squeeze().detach().numpy()
        label = label.squeeze().detach().numpy()
        mask = mask.squeeze().detach().numpy()

        pcd_camera = img_to_camera(
            depth, rgb, meta['intrinsic'], view=False)   # (n,3)
        pcd_world = camera_to_world(pcd_camera, meta['extrinsic'])
        objects = crop_obj(pcd_world, mask, meta, train=True)

        for (pcd_world, pcd_canonica, pose, id, name) in objects:
            if (id, name) in OBJ_LEGO.keys() and OBJ_LEGO[(id, name)] is not None:
                existing_lego_and_poses = OBJ_LEGO[(id, name)]
                existing_lego_and_poses.append((pcd_canonica, pose))
                OBJ_LEGO[(id, name)] = existing_lego_and_poses
            else:
                OBJ_LEGO[(id, name)] = [(pcd_canonica, pose)]

    with open(fname, 'wb') as handle:
        pickle.dump(OBJ_LEGO, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_center(pcd):
    ''' Get centroid of input pcd (n,3) '''
    center = pcd.mean(0).copy()
    return np.expand_dims(center, 0)


''' --- Visualization --- '''


def plot_pcd(pcd, ax_lim=None, path=""):
    ''' Plot np.array (n,3)
        - ax_lim: 2d List (3,2). Min, max limit for x,y,z axis.
    '''
    if not isinstance(pcd, np.ndarray):
        pcd = np.array(pcd.points)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if ax_lim is None:
        ax.set_xlim3d([-0.5, 0.5])
        ax.set_ylim3d([-0.25, 0.25])
        ax.set_zlim3d([-0.25, 0.25])
    else:
        ax.set_xlim3d(ax_lim[0])
        ax.set_ylim3d(ax_lim[1])
        ax.set_zlim3d(ax_lim[2])

    ax.scatter(pcd[:, 0], pcd[:, 2], pcd[:, 1],
               marker='.', alpha=1, edgecolors='none')

    plt.show()
    if len(path) > 0:
        plt.savefig(f"./{path}.png")


def plot_pcd_ctr(pcd, ctr, ax_lim=None, path=""):
    ''' Plot np.array (n,3)
        - ax_lim: 2d List (3,2). Min, max limit for x,y,z axis.
    '''
    if not isinstance(pcd, np.ndarray):
        pcd = np.array(pcd.points)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if ax_lim is None:
        ax.set_xlim3d([-0.5, 0.5])
        ax.set_ylim3d([-0.25, 0.25])
        ax.set_zlim3d([-0.25, 0.25])
    else:
        ax.set_xlim3d(ax_lim[0])
        ax.set_ylim3d(ax_lim[1])
        ax.set_zlim3d(ax_lim[2])

    ax.scatter(pcd[:, 0], pcd[:, 2], pcd[:, 1],
               marker='.', alpha=0.5, edgecolors='none')
    ax.scatter(ctr[:, 0], ctr[:, 2], ctr[:, 1],
               marker='.', alpha=1, edgecolors='r')

    plt.show()
    if len(path) > 0:
        plt.savefig(f"./{path}.png")


def plot_pcds(pcds, ax_lim=None, path=""):
    ''' - pcds: List of np.array (n,3)
        - ax_lim: 2d List (3,2). Min, max limit for x,y,z axis.
    '''
    fig = plt.figure(figsize=(16, 7))
    num_cols, num_rows = len(pcds)//2+1, 2

    for i, pcd in enumerate(pcds):

        idx = num_cols * (i % 2) + (i//num_rows + 1)
        # idx=i+1
        ax = fig.add_subplot(num_rows, num_cols, idx, projection='3d')

        if ax_lim is None:
            ax.set_xlim3d([-0.5, 0.5])
            ax.set_ylim3d([-0.25, 0.25])
            ax.set_zlim3d([-0.25, 0.25])
        else:
            ax.set_xlim3d(ax_lim[0])
            ax.set_ylim3d(ax_lim[1])
            ax.set_zlim3d(ax_lim[2])

        ax.scatter(pcd[:, 0], pcd[:, 2], pcd[:, 1],
                   marker='.', alpha=1, edgecolors='none')

    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()

    if len(path) > 0:
        plt.savefig(f"./imgs/{path}.png")


if __name__ == '__main__':
    print(' ---- Testing `utils_icp.py` ... ---- ')

    sample_data = SampleData()
    rgb = sample_data.rgb
    dpt = sample_data.dpt
    label = sample_data.label
    meta = sample_data.meta
    K = meta['intrinsic']

    # pcd, choose = dpt_2_cld(dpt, 1, K)
    # plot_img_2d(pcd.reshape(720, 1280, 3), 'foo1.png')
    # plot_img_2d(choose.reshape(720, 1280), 'foo2.png')
    # plot_pcd(pcd, ax_lim = [[-1,1], [-1,1], [-1,1]], path='foo3')
    # plot_pcd(pcd[choose], ax_lim = [[-1,1], [-1,1], [-1,1]], path='foo4')

    print(
        f' dpt_2_cld: Input [{dpt.shape}]. Output (pcd, choose, pcd_chosen) [{pcd.shape, choose.shape, pcd[choose].shape}]')

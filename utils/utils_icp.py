import copy, pickle, torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from open3d import io, geometry
from utils.utils_data import *


""" Everything you need for closed-form ICP 
    See `hw2_answer.ipynb`, `see_data.ipynb` for demonstration
"""


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
        if down_sample: pcd = pcd.voxel_down_sample(voxel_size)

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
        rotation[:3,3] = np.array([0,0,0])  # remove translation component
        src = copy.deepcopy(src).transform(rotation)
        translation = tgt.get_center() - src.get_center()
        return translation

    '''Source/target pcd is void. This might be cause by occlusion.'''
    if len(src.points) == 0 or len(tgt.points) == 0:    #
        if init_trans is not None:  
            return init_trans
        else: return np.eye(4)

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
        trans[:3,:3] = init_trans[:3,:3]    # Rotation is better initialized with initial pose from lego
        
        # trans[:3,3] = ransac_result.transformation[:3,3]  # Translation also could get from global reg.
        # trans[:3,3] = _init_translation(src, tgt, init_trans)  # Translation is better initialized using global registration
        trans[:3,3] = init_trans[:3,3]  
        
        return trans
    else:
        src = copy.deepcopy(src)
        tgt = copy.deepcopy(tgt)

        src, src_fpfh = _get_fpfh(src, voxel_size) 
        tgt, tgt_fpfh = _get_fpfh(tgt, voxel_size)  # Down sampling target lego might cause void pcd

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
    
    if len(src.points)==0 or len(tgt.points)==0:
        return np.eye(4), 0.0

    ''' Initialized `R, t, trans_pred` '''
    trans_pred = init_transformation(src, tgt,init_trans=init_trans, translate_only=True)  

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
            src,tgt, thresh, trans_pred, o3d.pipelines.registration.TransformationEstimationPointToPoint(),
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
    
    if len(src.points)==0 or len(tgt.points)==0:
        if init_trans is not None:
            return init_trans, 0, np.inf
        else:
            return np.eye(4),0, np.inf
    
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


def icp_one_scene_lego(OBJ_LEGO, rgb, depth, label, meta, \
    icp_option='o3d', view=False, train_set_fraction=1, icp_thresh=0.001):
    '''
    Input
        - OBJ_LEGO: Dict{(id,name): [(lego1, pose1), ...]}. Canonical legos and poses of objects
        - rgb, depth, label, meta
    Output
        - poses_world: List[79]. Poses for every object in the scene. 
    '''

    pcd_camera = img_to_camera(depth, rgb, meta['intrinsic'], view=False)   # (n,3)
    pcd_world = camera_to_world(pcd_camera, meta['extrinsic'])

    objects = crop_obj(pcd_world, label, meta, train=False) 
    poses_world = [None] * 79

    ''' For each obj'''
    for (obj_world, id, name) in objects:

        ''' For each training lego: do ICP and get fitness'''
        lego_poses, lego_fitnesses, lego_rmse,legos = [], [], [], []
        for i, (lego_canonical, lego_pose) in enumerate(OBJ_LEGO[(id,name)][::train_set_fraction]):
            if icp_option=='o3d':
                pose, fitness, rmse = icp_o3d(lego_canonical, obj_world, init_trans=lego_pose, thresh=icp_thresh, downsample_fraction=1)
            else: 
                pose, fitness = icp(lego_canonical, obj_world, init_trans=lego_pose, iters=500)
            lego_poses.append(pose)
            lego_fitnesses.append(fitness)
            legos.append(lego_canonical)
            if rmse: lego_rmse.append(rmse)

        lego_poses, lego_fitnesses, lego_rmse = np.array(lego_poses), np.array(lego_fitnesses), np.array(lego_rmse)
        
        best_lego = legos[np.argmax(lego_fitnesses)]    
        best_pose = lego_poses[np.argmax(lego_fitnesses)]

        '''Self-implemented ICP has better accuracy. So finally perform it.'''
        best_pose, best_fitness = icp(best_lego, obj_world, init_trans=best_pose, iters=1000)
        poses_world[id] = best_pose.tolist()

    meta['poses_world'] = poses_world

    
    if view: 
        print(f'In icp_one_scene_leg. Shapes: rgb {rgb.shape}, depth {depth.shape}, label {label.shape}')
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

    pcd_camera = img_to_camera(depth, rgb, meta['intrinsic'], view=False)   # (n,3)
    pcd_world = camera_to_world(pcd_camera, meta['extrinsic'])

    ''' ICP on each object in the scene '''
    objects = crop_obj(pcd_world, label, meta, train=False) 

    poses_world = [None] * 79                               # NUM_OBJECTS = 79
    for (obj_world, id, name) in objects:
        ''' 1. Points in world vs. canonical frame before ICP'''
        obj_canonical = MEAN_OBJS[(id, name)][::500]
        obj_canonical = pcd_to_o3d(obj_canonical)
        obj_world = pcd_to_o3d(obj_world)
        if view: compare_points(obj_canonical, obj_world)   # Compare scr, tgt before icp

        ''' 2. Do ICP'''
        pose_pred, fitness, rmse = icp_o3d(obj_canonical, obj_world)  
        poses_world[id] = pose_pred.tolist()

        if view: compare_points(obj_canonical, obj_world)   # Compare scr, tgt after icp
    
    return poses_world




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
    if isinstance(depth, torch.Tensor) and len(depth.shape)==4:
        depth = depth.squeeze().detach().numpy()
    if isinstance(rgb, torch.Tensor) and len(rgb.shape)==4:
        rgb = rgb.squeeze(0).detach().numpy()  
    z = depth
    # v, u of shape (720,1280) max(v)=719, max(u)=1279
    v, u = np.indices(z.shape)
    uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
    # [H, W, 3]  TODO: Why?
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
    
    R, t = extrinsic[:3,:3], extrinsic[:3,3]
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
    R, t = pose_world[:3,:3], pose_world[:3,3]
    pcd_canonical = (pcd_world - t) @ R
    return pcd_canonical


''' --- Build mean object. TODO: This is likely trash. ----''' 

def crop_obj(pcd_world, label, meta, train=False):
    '''Given 1 scene, crop out each object. 
    Input
        - pcd_world : np.array (n,3). Points in the world frame.
        - labe      : np.array (H,W). Segmented img
        - meta      : loaded pickel object. Metadata
    Output
        - List[tuples]: Each tuple is either
            (obj_pcd_world, obj_pcd_canonical, id, name) for training data
            (obj_pcd_world,                    id, name) for testing data
    '''
    if train:
        obj_ids, obj_names, obj_poses = meta['object_ids'], meta['object_names'], meta['poses_world']
        objects = []
        for id, name in zip(obj_ids, obj_names):
            obj_pcd_world = pcd_world[np.nonzero(label.reshape(-1) == id)] 
            pose = obj_poses[id]
            obj_pcd_canonical = world_to_canonical(obj_pcd_world, pose)
            objects.append((obj_pcd_world, obj_pcd_canonical, pose, id, name))
    else:
        obj_ids, obj_names = meta['object_ids'], meta['object_names']
        objects = []
        for id, name in zip(obj_ids, obj_names):
            obj_pcd_world = pcd_world[np.nonzero(label.reshape(-1) == id)]     
            objects.append((obj_pcd_world, id, name))

    return objects
        

def get_mean_objects(rgb_files, depth_files, label_files, meta_files):
    
    MEAN_OBJECTS = {}

    ''' For each training scene '''
    # Meta keys: ['poses_world', 'extents', 'scales', 'object_ids', 'object_names', 'extrinsic', 'intrinsic']
    with tqdm(total=len(rgb_files)) as pbar:
        for i, (rgb, depth, label, meta) in tqdm(enumerate(
            zip(rgb_files, depth_files, label_files, meta_files))):
            
            rgb, depth, label, meta = load_files(rgb, depth, label, meta) 
            
            pcd_camera = img_to_camera(depth, rgb, meta['intrinsic'], view=False)   # (n,3)
            pcd_world = camera_to_world(pcd_camera, meta['extrinsic'])
            objects = crop_obj(pcd_world, label, meta, train=True)

            for (obj_world, obj_canonical, id, name) in objects:
                if (id, name) in MEAN_OBJECTS.keys():
                    old_obj = MEAN_OBJECTS[(id,name)]
                    MEAN_OBJECTS[(id,name)] = np.concatenate([old_obj, obj_canonical])
                else:
                    MEAN_OBJECTS[(id,name)] = obj_canonical
        
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

        pcd_camera = img_to_camera(depth, rgb, meta['intrinsic'], view=False)   # (n,3)
        pcd_world = camera_to_world(pcd_camera, meta['extrinsic'])
        objects = crop_obj(pcd_world, mask, meta, train=True)
        
        for (pcd_world, pcd_canonica, pose, id, name) in objects:
            if (id,name) in OBJ_LEGO.keys() and OBJ_LEGO[(id,name)] is not None:
                existing_lego_and_poses = OBJ_LEGO[(id,name)]
                existing_lego_and_poses.append((pcd_canonica, pose))
                OBJ_LEGO[(id,name)] = existing_lego_and_poses
            else:
                OBJ_LEGO[(id,name)] = [(pcd_canonica, pose)]

    with open(fname, 'wb') as handle:
        pickle.dump(OBJ_LEGO, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
        ax.set_xlim3d([-0.5, 0.5]); ax.set_ylim3d([-0.25,0.25]); ax.set_zlim3d([-0.25, 0.25])
    else:
        ax.set_xlim3d(ax_lim[0]); ax.set_ylim3d(ax_lim[1]); ax.set_zlim3d(ax_lim[2])

    ax.scatter(pcd[:, 0], pcd[:, 2], pcd[:, 1], marker='.', alpha=0.01, edgecolors='none')

    plt.show()
    if len(path) > 0: plt.savefig(f"./imgs/{path}.png")


def plot_pcds(pcds, ax_lim=None, path=""): 
    ''' - pcds: List of np.array (n,3)
        - ax_lim: 2d List (3,2). Min, max limit for x,y,z axis.
    '''
    fig = plt.figure(figsize=(16,7))
    num_cols, num_rows = len(pcds)//2+1, 2
     
    for i,pcd in enumerate(pcds):
        
        idx = num_cols * (i%2) + (i//num_rows + 1)
        # idx=i+1
        ax = fig.add_subplot(num_rows, num_cols, idx, projection='3d')

        if ax_lim is None:
            ax.set_xlim3d([-0.5, 0.5]); ax.set_ylim3d([-0.25,0.25]); ax.set_zlim3d([-0.25, 0.25])
        else:
            ax.set_xlim3d(ax_lim[0]); ax.set_ylim3d(ax_lim[1]); ax.set_zlim3d(ax_lim[2])

        ax.scatter(pcd[:, 0], pcd[:, 2], pcd[:, 1], marker='.', alpha=0.01, edgecolors='none')

    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.show()
    if len(path) > 0: plt.savefig(f"./imgs/{path}.png")

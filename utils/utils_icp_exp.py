from utils_icp import *

def icp_one_scene_lego_without_final_round(OBJ_LEGO, rgb, depth, label, meta, \
    icp_option='o3d', view=False, train_set_fraction=1, icp_thresh=0.005):
    ''' Remove self-implemented ICP'''

    pcd_camera = img_to_camera(depth, rgb, meta['intrinsic'], view=False)   # (n,3)
    pcd_world = camera_to_world(pcd_camera, meta['extrinsic'])

    objects = crop_obj(pcd_world, label, meta, train=False) 
    poses_world = [None] * 79

    ''' For each obj'''
    for (obj_world, id, name) in objects:

        ''' For each training lego: do ICP and get fitness'''
        lego_poses, lego_fitnesses, legos = [], [], []
        for i, (lego_canonical, lego_pose) in enumerate(OBJ_LEGO[(id,name)][::train_set_fraction]):
            if icp_option=='o3d':
                pose, fitness = icp_o3d(lego_canonical, obj_world, init_trans=lego_pose, thresh=icp_thresh, downsample_fraction=1)
            else: 
                pose, fitness = icp(lego_canonical, obj_world, init_trans=lego_pose, iters=500)
            lego_poses.append(pose)
            lego_fitnesses.append(fitness)
            legos.append(lego_canonical)

        lego_poses, lego_fitnesses = np.array(lego_poses), np.array(lego_fitnesses)

        best_pose = lego_poses[np.argmax(lego_fitnesses)]
        best_lego = legos[np.argmax(lego_fitnesses)]    # FIXME: Where's the best lego?  This is fixed now. Try it again

        poses_world[id] = best_pose.tolist()

    meta['poses_world'] = poses_world
    if view: visualize_training_data(rgb, depth, label, meta)
    return poses_world


def icp_one_scene_lego_without_init(OBJ_LEGO, rgb, depth, label, meta, \
    icp_option='o3d', view=False, train_set_fraction=1, icp_thresh=0.005):
    ''' Remove self-implemented ICP'''

    pcd_camera = img_to_camera(depth, rgb, meta['intrinsic'], view=False)   # (n,3)
    pcd_world = camera_to_world(pcd_camera, meta['extrinsic'])

    objects = crop_obj(pcd_world, label, meta, train=False) 
    poses_world = [None] * 79

    ''' For each obj'''
    for (obj_world, id, name) in objects:

        ''' For each training lego: do ICP and get fitness'''
        lego_poses, lego_fitnesses, legos = [], [], []
        for i, (lego_canonical, lego_pose) in enumerate(OBJ_LEGO[(id,name)][::train_set_fraction]):
            if icp_option=='o3d':
                pose, fitness = icp_o3d(lego_canonical, obj_world, init_trans=None, thresh=icp_thresh, downsample_fraction=1)
            else: 
                pose, fitness = icp(lego_canonical, obj_world, init_trans=lego_pose, iters=500)
            lego_poses.append(pose)
            lego_fitnesses.append(fitness)
            legos.append(lego_canonical)

        lego_poses, lego_fitnesses = np.array(lego_poses), np.array(lego_fitnesses)

        best_pose = lego_poses[np.argmax(lego_fitnesses)]
        best_lego = legos[np.argmax(lego_fitnesses)]    # FIXME: Where's the best lego?  This is fixed now. Try it again

        poses_world[id] = best_pose.tolist()

    meta['poses_world'] = poses_world
    if view: visualize_training_data(rgb, depth, label, meta)
    return poses_world


"""Utilities for pose evaluation (symmetric-aware and size-aware).

For many applications related to 6D pose estimation, it is important to take symmetry and size into consideration.
We consider rotational symmetry with a finite order (like a cube) or an infinite order (like a cylinder).

Notes:
    For TOC benchmark, I try to use as few additional libraries as possible.
"""

import itertools
import numpy as np


def compute_rre(R_est: np.ndarray, R_gt: np.ndarray):
    """Compute the relative rotation error (geodesic distance of rotation)."""
    assert R_est.shape == (3, 3), 'R_est: expected shape (3, 3), received shape {}.'.format(R_est.shape)
    assert R_gt.shape == (3, 3), 'R_gt: expected shape (3, 3), received shape {}.'.format(R_gt.shape)
    # relative rotation error (RRE)
    rre = np.arccos(np.clip(0.5 * (np.trace(R_est.T @ R_gt) - 1), -1.0, 1.0))
    return rre


def compute_rte(t_est: np.ndarray, t_gt: np.ndarray):
    assert t_est.shape == (3,), 't_est: expected shape (3,), received shape {}.'.format(t_est.shape)
    assert t_gt.shape == (3,), 't_gt: expected shape (3,), received shape {}.'.format(t_gt.shape)
    # relative translation error (RTE)
    rte = np.linalg.norm(t_est - t_gt)
    return rte


def get_rotation_matrix(axis, angle):
    """Returns a 3x3 rotation matrix that performs a rotation around axis by angle.

    Args:
        axis (np.ndarray): axis to rotate about
        angle (float): angle to rotate by

    Returns:
        np.ndarray: 3x3 rotation matrix A.

    References:
        https://en.wikipedia.org/wiki/Rotation_matrix
    """
    axis = np.asarray(axis)
    assert axis.ndim == 1 and axis.size == 3
    u = axis / np.linalg.norm(axis)
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    cross_prod_mat = np.cross(np.eye(3), u)
    R = cos_angle * np.eye(3) + sin_angle * cross_prod_mat + (1.0 - cos_angle) * np.outer(u, u)
    return R


def _get_symmetry_rotations(sym_axes, sym_orders):
    """Get symmetry rotations from axes and orders.

    Args:
        sym_axes: [N] list, each item is [3] array.
        sym_orders: [N] list, each item is a scalar (can be inf) or None.
            None is for no symmetry.

    Returns:
        list: [N] list, each item is a [sym_order] list of [3, 3] symmetry rotations.
        np.array or None: if there exists a symmetry axis with inf order.
    """
    sym_rots = []
    rot_axis = None
    assert len(sym_axes) == len(sym_orders)
    for sym_axis, sym_order in zip(sym_axes, sym_orders):
        if sym_order is None:
            sym_rots.append([np.eye(3)])
        elif np.isinf(sym_order):
            if rot_axis is None:
                rot_axis = sym_axis
            else:
                raise ValueError('Multiple rotation axes.')
            sym_rots.append([np.eye(3)])
        else:
            assert sym_order > 0
            Rs = []
            for i in range(0, sym_order):
                angle = i * (2 * np.pi / sym_order)
                R = get_rotation_matrix(sym_axis, angle)
                Rs.append(R)
            sym_rots.append(Rs)
    return sym_rots, rot_axis


def get_symmetry_rotations(sym_axes, sym_orders, unique=False, verbose=False):
    """Check _get_symmetry_rotations."""
    sym_rots_per_axis, rot_axis = _get_symmetry_rotations(sym_axes, sym_orders)

    sym_rots = []
    range_indices = list(range(len(sym_axes)))
    for indices in itertools.permutations(range_indices):
        sym_rots_per_axis_tmp = [sym_rots_per_axis[i] for i in indices]
        for Rs in itertools.product(*sym_rots_per_axis_tmp):
            R_tmp = np.eye(3)
            for R in Rs:
                R_tmp = R_tmp @ R
            sym_rots.append(R_tmp)

    sym_rots = np.array(sym_rots)

    if unique:
        ori_size = sym_rots.shape[0]
        sym_rots_flat = sym_rots.reshape(-1, 9)  # [?, 9]
        pdist = np.linalg.norm(sym_rots_flat[:, np.newaxis] - sym_rots_flat[np.newaxis], axis=-1)
        mask = np.tril(pdist < 1e-6, k=-1)
        mask = np.any(mask, axis=1)  # [?]
        sym_rots = sym_rots[~mask]
        if verbose:
            print(ori_size, sym_rots.shape[0])

    return sym_rots, rot_axis


def compute_rre_symmetry(R_est: np.ndarray, R_gt: np.ndarray,
                         sym_rots: np.ndarray, rot_axis=None):
    assert R_est.shape == (3, 3), 'R_est: expected shape (3, 3), received shape {}.'.format(R_est.shape)
    assert R_gt.shape == (3, 3), 'R_gt: expected shape (3, 3), received shape {}.'.format(R_gt.shape)

    if rot_axis is None:
        R_gt_sym = R_gt @ sym_rots
        rre_sym_all = np.arccos(np.clip(0.5 * (np.trace(R_est.T @ R_gt_sym, axis1=-2, axis2=-1) - 1), -1.0, 1.0))
        rre_best = np.min(rre_sym_all)
    else:
        R_gt_sym = R_gt @ sym_rots
        rot_axis_gt = R_gt_sym @ rot_axis  # [?, 3]
        rot_axis_est = R_est @ rot_axis  # [3]
        rre_sym = np.arccos(np.clip(np.dot(rot_axis_gt, rot_axis_est), -1.0, 1.0))  # [?]
        rre_best = np.min(rre_sym)
    return rre_best


def get_corners():
    """Get 8 corners of a cuboid. (The order follows OrientedBoundingBox in open3d)

            (y)
            2 -------- 7
           /|         /|
          5 -------- 4 .
          | |        | |
          . 0 -------- 1 (x)
          |/         |/
          3 -------- 6
          (z)
    """
    corners = np.array([[0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [0.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0],
                        [1.0, 1.0, 0.0],
                        ])
    return corners - [0.5, 0.5, 0.5]


def compute_rre_symmetry_with_scale(R_est: np.ndarray,
                                    R_gt: np.ndarray,
                                    sym_rots: np.ndarray,
                                    rot_axis=None,
                                    scales=np.ones(3),
                                    ):
    assert R_est.shape == (3, 3), 'R_est: expected shape (3, 3), received shape {}.'.format(R_est.shape)
    assert R_gt.shape == (3, 3), 'R_gt: expected shape (3, 3), received shape {}.'.format(R_gt.shape)

    pts = get_corners() * scales

    if rot_axis is None:
        R_gt_sym = R_gt @ sym_rots
        pts_gt_all = pts @ np.transpose(R_gt_sym, [0, 2, 1])  # [?, 8, 3]
        pts_est = pts @ R_est.T  # [8, 3]
        pts_err_all = np.linalg.norm(pts_est - pts_gt_all, axis=-1).mean(-1)  # [?]
        pts_err = np.min(pts_err_all)
    else:
        R_gt_sym = R_gt @ sym_rots
        pts = np.dot(pts, rot_axis)[:, np.newaxis] * rot_axis
        pts_gt_all = pts @ np.transpose(R_gt_sym, [0, 2, 1])  # [?, 8, 3]
        pts_est = pts @ R_est.T  # [8, 3]
        pts_err_all = np.linalg.norm(pts_est - pts_gt_all, axis=-1).mean(-1)  # [?]
        pts_err = np.min(pts_err_all)
    return pts_err


# ---------------------------------------------------------------------------- #
# Unittest
# ---------------------------------------------------------------------------- #
def test_compute_rre():
    from scipy.spatial.transform import Rotation

    R_gt = Rotation.from_quat(np.random.randn(4)).as_matrix()
    angle = np.random.uniform(0, np.pi)
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    R_rel = Rotation.from_rotvec(angle * axis).as_matrix()
    R_est = R_gt @ R_rel

    # compute_rre
    rre = compute_rre(R_est, R_gt)
    np.testing.assert_allclose(rre, angle)

    # BTW, test get_rotation_matrix
    np.testing.assert_allclose(get_rotation_matrix(axis, angle), R_rel)


def test_get_symmetry_rotations():
    sym_rots = get_symmetry_rotations(np.eye(3), [None, None, None], unique=True, verbose=True)[0]
    assert sym_rots.shape[0] == 1

    sym_rots = get_symmetry_rotations(np.eye(3), [None, None, 2], unique=True, verbose=True)[0]
    assert sym_rots.shape[0] == 2

    sym_rots = get_symmetry_rotations(np.eye(3), [2, 2, None], unique=True, verbose=True)[0]
    assert sym_rots.shape[0] == 4

    sym_rots = get_symmetry_rotations(np.eye(3), [2, None, 4], unique=True, verbose=True)[0]
    assert sym_rots.shape[0] == 8

    sym_rots = get_symmetry_rotations(np.eye(3), [4, 4, 4], unique=True, verbose=True)[0]
    assert sym_rots.shape[0] == 24


def test_compute_rre_symmetry():
    from scipy.spatial.transform import Rotation

    # identity
    R_gt = Rotation.from_quat(np.random.randn(4)).as_matrix()
    R_est = R_gt
    rre = compute_rre(R_est, R_gt)
    np.testing.assert_allclose(rre, 0.0, atol=1e-6)
    sym_rots, rot_axis = get_symmetry_rotations(np.eye(3), [2, None, 2])
    rre_symmetry = compute_rre_symmetry(R_est, R_gt, sym_rots, rot_axis)
    np.testing.assert_allclose(rre_symmetry, 0.0, atol=1e-6)

    # no rot_axis
    R_gt = Rotation.from_quat(np.random.randn(4)).as_matrix()
    R_z120 = Rotation.from_euler('z', np.pi * 2 / 3, degrees=False).as_matrix()
    R_est = R_gt @ R_z120
    rre = compute_rre(R_est, R_gt)
    np.testing.assert_allclose(rre, np.pi * 2 / 3)
    sym_rots, rot_axis = get_symmetry_rotations(np.eye(3), [None, None, 3])
    rre_symmetry = compute_rre_symmetry(R_est, R_gt, sym_rots, rot_axis)
    np.testing.assert_allclose(rre_symmetry, 0.0, atol=1e-6)

    # rot_axis
    R_gt = Rotation.from_quat(np.random.randn(4)).as_matrix()
    angle = np.random.uniform(0, np.pi)
    R_rel = Rotation.from_euler('XY', [np.pi * 2 / 3, angle]).as_matrix()
    R_est = R_gt @ R_rel
    angle_desired = np.linalg.norm(Rotation.from_matrix(R_rel).as_rotvec())
    rre = compute_rre(R_est, R_gt)
    np.testing.assert_allclose(rre, angle_desired)
    sym_rots, rot_axis = get_symmetry_rotations(np.eye(3), [2, np.inf, None])
    rre_symmetry = compute_rre_symmetry(R_est, R_gt, sym_rots, rot_axis)
    np.testing.assert_allclose(rre_symmetry, np.pi / 3, atol=1e-6)


def test_compute_rre_symmetry_with_scale():
    from scipy.spatial.transform import Rotation

    # no rot_axis
    R_gt = Rotation.from_quat(np.random.randn(4)).as_matrix()
    R_z120 = Rotation.from_euler('z', np.pi * 2 / 3, degrees=False).as_matrix()
    R_est = R_gt @ R_z120
    pts_err = compute_rre_symmetry_with_scale(R_est, R_gt, np.array([np.eye(3)]))
    np.testing.assert_allclose(pts_err, np.sqrt(6) / 2, atol=1e-6)
    sym_rots, rot_axis = get_symmetry_rotations(np.eye(3), [None, None, 3])
    pts_err = compute_rre_symmetry_with_scale(R_est, R_gt, sym_rots, rot_axis)
    np.testing.assert_allclose(pts_err, 0.0, atol=1e-6)

    # rot_axis
    R_gt = Rotation.from_quat(np.random.randn(4)).as_matrix()
    angle = np.random.uniform(0, np.pi)
    R_rel = Rotation.from_euler('XY', [np.pi / 2, angle]).as_matrix()
    R_est = R_gt @ R_rel
    pts_err = compute_rre_symmetry_with_scale(R_est, R_gt, np.array([np.eye(3)]), np.array([0., 1., 0.]))
    np.testing.assert_allclose(pts_err, np.sqrt(2) / 2, atol=1e-6)
    sym_rots, rot_axis = get_symmetry_rotations(np.eye(3), [4, np.inf, None])
    pts_err = compute_rre_symmetry(R_est, R_gt, sym_rots, rot_axis)
    np.testing.assert_allclose(pts_err, 0.0, atol=1e-6)

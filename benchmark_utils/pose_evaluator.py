import os
import csv
from collections import OrderedDict
import numpy as np

try:
    from . import pose_utils
except ImportError as e:
    import pose_utils

DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), 'objects_v1.csv')


class PoseEvaluator(object):
    def __init__(self, csv_file=DEFAULT_CSV_PATH):
        with open(csv_file, 'r') as f:
            objects_db = OrderedDict()
            for row in csv.DictReader(f):
                objects_db[row['object']] = row
        self.objects_db = objects_db
        self.generate_symmetry_info()

    def parse_symmetry_annotation(self, object_name):
        sym_orders = [None, None, None]

        object_info = self.objects_db[object_name]
        sym_labels = object_info['geometric_symmetry'].split('|')

        def _parse_fn(x):
            return float(x) if x == 'inf' else int(x)

        for sym_label in sym_labels:
            if sym_label[0] == 'x':
                sym_orders[0] = _parse_fn(sym_label[1:])
            elif sym_label[0] == 'y':
                sym_orders[1] = _parse_fn(sym_label[1:])
            elif sym_label[0] == 'z':
                sym_orders[2] = _parse_fn(sym_label[1:])
            elif sym_label == 'no':
                continue
            else:
                raise ValueError('Can not parse the symmetry label: {}.'.format(sym_label))

        return sym_orders

    def generate_symmetry_info(self):
        for object_name in self.objects_db:
            sym_axes = np.eye(3)
            sym_orders = self.parse_symmetry_annotation(object_name)
            sym_rots, rot_axis = pose_utils.get_symmetry_rotations(sym_axes, sym_orders, unique=True)

            object_db = self.objects_db[object_name]
            object_db['sym_rots'] = sym_rots
            object_db['rot_axis'] = rot_axis

    def evaluate(self,
                 object_name: str,
                 R_est: np.ndarray,
                 R_gt: np.ndarray,
                 t_est: np.ndarray,
                 t_gt: np.ndarray,
                 scales: np.ndarray,
                 ):
        object_db = self.objects_db[object_name]
        object_template_scale = np.array([
            float(object_db['width']),
            float(object_db['length']),
            float(object_db['height']),
        ])
        rre = pose_utils.compute_rre(R_est, R_gt)
        rre_symmetry = pose_utils.compute_rre_symmetry(R_est, R_gt,
                                                       sym_rots=object_db['sym_rots'],
                                                       rot_axis=object_db['rot_axis'])
        rte = pose_utils.compute_rte(t_est, t_gt)
        pts_err = pose_utils.compute_rre_symmetry_with_scale(R_est, R_gt,
                                                             sym_rots=object_db['sym_rots'],
                                                             rot_axis=object_db['rot_axis'],
                                                             scales=object_template_scale * scales)

        return {
            'rre': np.rad2deg(rre),
            'rre_symmetry': np.rad2deg(rre_symmetry),
            'rte': rte,
            'pts_err': pts_err,
        }


def test():
    from scipy.spatial.transform import Rotation

    R_gt = Rotation.from_quat(np.random.randn(4)).as_matrix()
    # angle = np.random.uniform(0, np.pi)
    angle = np.random.uniform(0, np.pi / 180)
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    R_rel = Rotation.from_rotvec(angle * axis).as_matrix()
    R_est = R_gt @ R_rel

    t_gt = np.random.uniform(-1, 1, [3])
    # t_est = np.random.uniform(-1, 1, [3])
    t_est = t_gt + [1., 0., 0.]

    pose_evaluator = PoseEvaluator()
    print(pose_evaluator.evaluate('bowl', R_est, R_gt, t_est, t_gt, np.ones(3)))

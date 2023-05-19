"""Benchmark script.

gt_labels={
    "{scene_id}": {
        "object_ids": [],
        "object_names": [],
        "poses_world": [],
    }
}

pred_results={
    "{scene_id}": {
        "poses_world": [],
    }
}
"""

import os
import argparse
import csv
import json
import numpy as np


CUR_DIR = os.path.dirname(__file__)

try:
    from .pose_evaluator import PoseEvaluator
except ImportError as e:
    import sys

    sys.path.append(os.path.abspath(CUR_DIR))
    from pose_evaluator import PoseEvaluator

METRICS = ['rre', 'rre_symmetry', 'rte', 'pts_err']
MAX_METRIC_VALUES = [180.0, 180.0, 1.0, 1.0]
POSE_THRESH_LIST = [(5, 0.01), (10, 0.01), (10, 0.02), (15, 0.02)]


def parse_args():
    parser = argparse.ArgumentParser(description='Generate evaluation data')
    parser.add_argument(
        '--gt-path',
        default='dummy_gt.json',
        type=str,
        help='path to gt label',
    )
    parser.add_argument(
        '--pred-path',
        type=str,
        required=True,
        help='path to prediction',
    )
    parser.add_argument(
        '--result-path',
        type=str,
        help='path to export result. '
             'If relative path is provided, it will be exported under the pred_dir.',
    )
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Whether to output a table that can be parsed as csv (required tabulate).'
    )
    parser.add_argument(
        '--summary-path',
        type=str,
        help='Path to summarize evaluation result into csv (required tabulate).'
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Load GT
    with open(args.gt_path, 'r') as f:
        gt_labels = json.load(f)
    # Load predictions
    with open(args.pred_path, 'r') as f:
        pred_results = json.load(f)

    eval_result, err_dict_all, headers, table = evaluate(gt_labels, pred_results)
    # print(eval_result)

    if args.pretty:
        from tabulate import tabulate
        print(tabulate(table, headers=headers, tablefmt='psql', floatfmt='.4f'))
        print('You can parse the following string into a sheet.')
        print(tabulate(table, headers=headers, tablefmt='tsv', floatfmt='.4f'))
    else:
        import pprint
        for row in table:
            pprint.pprint([f'{header}={x}' for header, x in zip(headers, row)])

    # Export to csv
    if args.result_path is not None:
        if os.path.isabs(args.result_path):
            result_path = args.result_path
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
        else:
            result_path = os.path.join(args.pred_dir, args.result_path)

        fieldnames = ['scene_id', 'level_id', 'variant_id', 'object_name', 'gt_scale'] + METRICS
        with open(result_path, 'w') as f:
            csv_writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            csv_writer.writeheader()
            csv_writer.writerows(err_dict_all)
        print(f'The detailed evaluation result is saved to {result_path}.')

    if args.summary_path:
        summary_path = args.summary_path
        with open(summary_path, 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(headers)
            csv_writer.writerows(table)
        print(f'The overall evaluation result is saved to {summary_path}.')


def evaluate(gt_labels, pred_results, level_ids=[1,2,3,4,5]):
    pose_evaluator = PoseEvaluator(os.path.join(CUR_DIR, 'objects_v1.csv'))
    NUM_OBJECTS = len(pose_evaluator.objects_db)
    scene_ids = list(gt_labels.keys())

    err_dict_all = []
    # Iterate over all the specified scenes
    for scene_id in scene_ids:
        gt_label = gt_labels[scene_id]
        pred_result = pred_results[scene_id]

        object_ids = gt_label['object_ids']
        object_names = gt_label['object_names']
        assert len(object_ids) == len(object_names)

        gt_poses_world = gt_label['poses_world']
        gt_scales = gt_label['scales']
        assert len(gt_poses_world) == len(gt_scales) == NUM_OBJECTS
        if 'visibility' not in gt_label:
            import warnings
            warnings.warn('Please update your evaluation data with visibility annotation.')
            visibility = np.zeros(NUM_OBJECTS, dtype=bool)
            visibility[object_ids] = True
        else:
            visibility = gt_label['visibility']

        pred_poses_world = pred_result['poses_world']
        assert len(pred_poses_world) == NUM_OBJECTS

        for object_id, object_name in zip(object_ids, object_names):
            if not visibility[object_id]:
                # Ignore invisible objects
                continue

            pose_est = pred_poses_world[object_id]
            pose_gt = gt_poses_world[object_id]
            gt_scale = gt_scales[object_id]

            if pose_est is None:
                err_dict = {name: np.nan for name in METRICS}
            else:
                pose_est = np.asarray(pose_est)
                pose_gt = np.asarray(pose_gt)
                gt_scale = np.asarray(gt_scale)
                err_dict = pose_evaluator.evaluate(object_name,
                                                   R_est=pose_est[:3, :3],
                                                   R_gt=pose_gt[:3, :3],
                                                   t_est=pose_est[:3, 3],
                                                   t_gt=pose_gt[:3, 3],
                                                   scales=gt_scale,
                                                   )
                err_dict = {name: err_dict[name] for name in METRICS}

            # add additional info
            err_dict['scene_id'] = scene_id
            level_id, variant_id = scene_id.split('-', 1)
            err_dict['level_id'] = int(level_id)
            err_dict['variant_id'] = str(variant_id)
            err_dict['object_name'] = object_name
            err_dict['gt_scale'] = gt_scale
            err_dict_all.append(err_dict)

    # ---------------------------------------------------------------------------- #
    # Summarize
    # ---------------------------------------------------------------------------- #
    eval_result = {}  # specified for online benchmark.
    table = []

    # Analyze by level_ids
    for level_id in level_ids:
        err_dict_filtered = list(filter(lambda x: x['level_id'] == level_id, err_dict_all))
        err_dict_median = summarize_result(err_dict_filtered, METRICS, np.median, replace_nan=True)
        err_dict_mean = summarize_result(err_dict_filtered, METRICS, np.nanmean, replace_nan=False)
        err_dict_std = summarize_result(err_dict_filtered, METRICS, np.nanstd, replace_nan=False)
        recall = np.mean(~np.isnan([x[METRICS[0]] for x in err_dict_filtered]))

        # pose accuracy
        pose_acc_all = compute_pose_accuracy(err_dict_filtered, POSE_THRESH_LIST)

        # For online benchmark
        level_name = f'level_{level_id}'
        eval_result[level_name] = {k: err_dict_median[k] for k in METRICS}
        for idx, (rre_thresh, rte_thresh) in enumerate(POSE_THRESH_LIST):
            metric_name = f'pose_acc_{rre_thresh:.0f}deg_{rte_thresh * 100.0:.0f}cm'
            eval_result[level_name].update({metric_name: pose_acc_all[idx]})

        table.append([level_id, recall]
                     + [err_dict_median[k] for k in METRICS]
                     + [err_dict_mean[k] for k in METRICS]
                     + [err_dict_std[k] for k in METRICS]
                     + pose_acc_all
                     )

    # Overall
    err_dict_median = summarize_result(err_dict_all, METRICS, np.median, replace_nan=True)
    err_dict_mean = summarize_result(err_dict_all, METRICS, np.nanmean, replace_nan=False)
    err_dict_std = summarize_result(err_dict_all, METRICS, np.nanstd, replace_nan=False)
    recall = np.mean(~np.isnan([x[METRICS[0]] for x in err_dict_all]))

    # pose accuracy
    pose_acc_all = compute_pose_accuracy(err_dict_all, POSE_THRESH_LIST)

    # For online benchmark
    eval_result['all'] = {k: err_dict_median[k] for k in METRICS}
    for idx, (rre_thresh, rte_thresh) in enumerate(POSE_THRESH_LIST):
        metric_name = f'pose_acc_{rre_thresh:.0f}deg_{rte_thresh * 100.0:.0f}cm'
        eval_result['all'].update({metric_name: pose_acc_all[idx]})

    table.append(['all', recall]
                 + [err_dict_median[k] for k in METRICS]
                 + [err_dict_mean[k] for k in METRICS]
                 + [err_dict_std[k] for k in METRICS]
                 + pose_acc_all
                 )
    headers = ['level', 'recall'] \
              + [f'{k}(median)' for k in METRICS] \
              + [f'{k}(mean)' for k in METRICS] \
              + [f'{k}(std)' for k in METRICS] \
              + [f'pose_acc_{rre_thresh:.1f}deg_{rte_thresh * 100.0:.1f}cm'
                 for (rre_thresh, rte_thresh) in POSE_THRESH_LIST]

    return eval_result, err_dict_all, headers, table


def summarize_result(err_dict_all, fieldnames, summarize_func, replace_nan=True):
    err_dict_overall = {}
    for fieldname in fieldnames:
        values = [err_dict[fieldname] for err_dict in err_dict_all]
        if replace_nan:
            max_metric_value = MAX_METRIC_VALUES[METRICS.index(fieldname)]
            values = [max_metric_value if np.isnan(x) else x for x in values]
        err_dict_overall[fieldname] = summarize_func(values)
    return err_dict_overall


def compute_pose_accuracy(err_dict_all, pose_thresh_list):
    pose_acc_all = []
    for rre_thresh, rte_thresh in pose_thresh_list:
        rre_all = np.array([err_dict['rre_symmetry'] for err_dict in err_dict_all])
        # rre_all = rre_all[~np.isnan(rre_all)]
        rre_all[np.isnan(rre_all)] = rre_thresh + 1
        rte_all = np.array([err_dict['rte'] for err_dict in err_dict_all])
        # rte_all = rte_all[~np.isnan(rte_all)]
        rte_all[np.isnan(rte_all)] = rte_thresh + 1
        pose_acc = np.mean(np.logical_and(rre_all <= rre_thresh, rte_all <= rte_thresh))
        pose_acc_all.append(pose_acc)
    return pose_acc_all


def generate_dummy_gt():
    with open('dummy_gt.json', 'w') as f:
        dummy_gt = {
            "1-1": {
                "object_ids": [0],
                "object_names": ['a_cups'],
                "scales": [np.ones(3).tolist()] + [None] * 78,
                "poses_world": [np.eye(4).tolist()] + [None] * 78,
            }
        }
        json.dump(dummy_gt, f, indent=1)


def generate_dummy_pred():
    with open('dummy_pred.json', 'w') as f:
        dummy_pred = {
            "1-1": {
                "poses_world": [np.eye(4).tolist()] + [None] * 78,
            }
        }
        json.dump(dummy_pred, f, indent=1)


if __name__ == '__main__':
    # generate_dummy_gt()
    # generate_dummy_pred()
    main()

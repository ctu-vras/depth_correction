from __future__ import absolute_import, division, print_function
from .config import Config, PoseCorrection
from .eval import eval_loss
from .model import *
from .train import train
from collections import deque
from data.asl_laser import Dataset, dataset_names
import os
import random
import roslaunch
import roslaunch.parent
import roslaunch.rlutil
from rospkg import RosPack


package_dir = RosPack().get_path('depth_correction')
slam_eval_launch = os.path.join(package_dir, 'launch', 'slam_eval.launch')

# TODO: Generate multiple splits.
ds = ['asl_laser/%s' % name for name in dataset_names]
num_splits = 4
shift = len(ds) // num_splits
splits = []

random.seed(135)
random.shuffle(ds)
ds_deque = deque(ds)
for i in range(num_splits):
    # random.shuffle(datasets)
    ds_deque.rotate(shift)
    # copy = list(datasets)
    # random.shuffle(copy)
    # splits.append([copy[:4], copy[4:6], copy[6:]])
    ds_list = list(ds_deque)
    splits.append([ds_list[:4], ds_list[4:6], ds_list[6:]])
# for split in splits:
#     print(split)

# splits = [[['asl_laser/eth'], ['asl_laser/stairs'], ['asl_laser/gazebo_winter']]]
models = ['Polynomial', 'ScaledPolynomial']
# models = ['ScaledPolynomial']
# losses = ['min_eigval_loss']
losses = ['min_eigval_loss', 'trace_loss']
slams = ['ethzasl_icp_mapper']


def eval_slam(cfg: Config=None):
    # TODO: Actually use slam id if multiple slam pipelines are to be tested.
    assert cfg.slam == 'ethzasl_icp_mapper'
    # csv = os.path.join(cfg.log_dir, 'slam_eval.csv')
    assert cfg.slam_eval_csv
    assert cfg.slam_poses_csv
    #
    # TODO: Run slam for each sequence in split.
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    for name in cfg.test_names:
        print('SLAM evaluation on %s started.' % name)
        cli_args = [slam_eval_launch, 'dataset:=%s' % name, 'odom:=true', 'rviz:=true',
                    'slam_eval_csv:=%s' % cfg.slam_eval_csv,
                    # TODO: Output poses to structure within log dir.
                    # 'slam_poses_csv=%s' % cfg.slam_poses_csv,
                    'min_depth:=%.1f' % cfg.min_depth, 'max_depth:=%.1f' % cfg.max_depth,
                    'grid_res:=%.1f' % cfg.grid_res,
                    # 'depth_correction:=%s' % ('true' if cfg.pose_correction != PoseCorrection.none else 'false'),
                    'depth_correction:=%s' % ('true' if cfg.model_class else 'false'),
                    'nn_r:=%.2f' % cfg.nn_r,
                    # TODO: Pass eigenvalue bounds to launch.
                    'eigenvalue_bounds:=[[0, -.inf, 0.0004], [1, 0.0025, .inf]]',
                    'model_class:=%s' % cfg.model_class, 'model_state_dict:=%s' % cfg.model_state_dict]

        roslaunch_args = cli_args[1:]
        roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
        parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file, force_log=True)
        parent.start()
        parent.spin()
        print('SLAM evaluation on %s finished.' % name)


def run_baselines():
    # evaluate consistency loss on all sequences
    cfg = Config()
    # Adjust default config...
    cfg.model_class = 'BaseModel'
    cfg.model_state_dict = ''
    cfg.log_dir = cfg.get_log_dir()
    os.makedirs(cfg.log_dir, exist_ok=True)

    for test_loss in losses:
        eval_cfg = cfg.copy()
        eval_cfg.test_names = ds
        eval_cfg.loss = test_loss
        eval_cfg.loss_eval_csv = os.path.join(cfg.log_dir, '%s_baseline.csv' % test_loss)
        eval_loss(cfg=eval_cfg)

    for slam in slams:
        eval_cfg = cfg.copy()
        eval_cfg.test_names = ds
        eval_cfg.slam = slam
        eval_cfg.slam_eval_csv = os.path.join(cfg.log_dir, '%s_baseline.csv' % slam)
        eval_slam(cfg=eval_cfg)


def run_model_from_ground_truth():
    base_cfg = Config()
    base_cfg.log_dir = base_cfg.get_log_dir()
    for model in models:
        for i_fold, (train_names, val_names, test_names) in enumerate(splits):  # train, val, test dataset split (cross validation)
            for loss in losses:
                # learn depth correction model using ground truth poses using the loss
                #     get best model from validation set using the loss
                # model = fit_model(model, train, val, loss)
                # cfg = Config()
                cfg = base_cfg.copy()
                cfg.model_class = model
                cfg.train_names = train_names
                cfg.val_names = val_names
                cfg.test_names = test_names
                cfg.loss = loss
                # desc = '%s_%s_split_%i' % (cfg.model_class.lower(), cfg.loss.lower(), i_split)
                cfg.log_dir = os.path.join(base_cfg.log_dir,
                                           '%s_%s' % (cfg.model_class.lower(), cfg.loss.lower()),
                                           'fold_%i' % i_fold)
                os.makedirs(cfg.log_dir, exist_ok=True)
                print(cfg.log_dir)
                best_cfg = train(cfg)
                # evaluate consistency loss on test (train, validation) set
                for split, suffix in zip([train_names, val_names, test_names],
                                         ['train', 'val', 'test']):
                    for test_loss in losses:
                        eval_cfg = best_cfg.copy()
                        eval_cfg.test_names = split
                        eval_cfg.loss = test_loss
                        eval_cfg.loss_eval_csv = os.path.join(cfg.log_dir,
                                                              'loss_eval_%s_%s.csv' % (test_loss, suffix))
                        eval_loss(cfg=eval_cfg)
                    # evaluate slam localization on test (train, validation) set
                    for slam in slams:
                        eval_cfg = best_cfg.copy()
                        eval_cfg.test_names = split
                        eval_cfg.slam = slam
                        eval_cfg.slam_eval_csv = os.path.join(cfg.log_dir,
                                                              'slam_eval_%s_%s.csv' % (slam, suffix))
                        eval_slam(cfg=eval_cfg)


def run_model_from_slam():
    for model in models:
        for train_names, val, test in splits:  # train, val, test dataset split (cross validation)
            for loss in losses:
                for slam in slams:
                    # generate slam poses for the splits
                    train_slam, val_slam, test_slam = [eval_slam(None, split, slam) for split in [train, val, test]]
                    # learn depth correction model using poses from slam
                    #     get best model from validation set
                    # train_slam = fit_model_slam(model, train, val, loss, slam)
                    cfg = Config()
                    cfg.train_names = train
                    # model = fit_model(model, train_slam, val_slam, loss, correct_poses=True)
                    # evaluate consistency on test (train, validation) set
                    # for test_loss in losses:
                    for split in [train, val, test]:
                        eval_loss(model, loss, split)
                        for slam in slams:
                            # evaluate slam localization on test (train, validation) set
                            eval_slam(model, split, slam)


def run_calibration():
    pass


def run_experiments():
    run_baselines()
    run_model_from_ground_truth()
    # run_model_from_slam()
    # run_calibration()


def main():
    run_experiments()


if __name__ == '__main__':
    main()

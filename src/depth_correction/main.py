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

models = ['Polynomial', 'ScaledPolynomial']
losses = ['min_eigval_loss', 'trace_loss']
slams = ['ethzasl_icp_mapper']

# Debug
# splits = [[['asl_laser/eth'], ['asl_laser/stairs'], ['asl_laser/gazebo_winter']]]
# models = ['ScaledPolynomial']
# losses = ['min_eigval_loss']
# slams = ['ethzasl_icp_mapper']


def slam_poses_csv(cfg: Config, name, slam):
    path = os.path.join(cfg.log_dir, name, 'slam_poses_%s.csv' % slam)
    return path


def eval_slam(cfg: Config=None):
    # TODO: Actually use SLAM id if multiple pipelines are to be tested.
    assert cfg.slam == 'ethzasl_icp_mapper'
    # csv = os.path.join(cfg.log_dir, 'slam_eval.csv')
    assert cfg.slam_eval_csv
    assert not cfg.slam_poses_csv or len(cfg.test_names) == 1
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    for i, name in enumerate(cfg.test_names):
        # Allow overriding poses paths, assume valid if non-empty.
        poses_path = cfg.test_poses_path[i] if cfg.test_poses_path else None
        print('SLAM evaluation on %s started.' % name)
        # print(cfg.to_yaml())

        cli_args = [slam_eval_launch,
                    'dataset:=%s' % name,
                    *(['dataset_poses_path:=%s' % poses_path] if poses_path else []),
                    'odom:=true',
                    'rviz:=%s' % ('true' if cfg.rviz else 'false'),
                    'slam_eval_csv:=%s' % cfg.slam_eval_csv,
                    *(['slam_poses_csv:=%s' % cfg.slam_poses_csv] if cfg.slam_poses_csv else []),
                    'min_depth:=%.1f' % cfg.min_depth, 'max_depth:=%.1f' % cfg.max_depth,
                    'grid_res:=%.1f' % cfg.grid_res,
                    # 'depth_correction:=%s' % ('true' if cfg.pose_correction != PoseCorrection.none else 'false'),
                    'depth_correction:=%s' % ('true' if cfg.model_class != 'BaseModel' else 'false'),
                    'nn_r:=%.2f' % cfg.nn_r,
                    # TODO: Pass eigenvalue bounds to launch.
                    'eigenvalue_bounds:=[[0, -.inf, 0.0004], [1, 0.0025, .inf]]',
                    'model_class:=%s' % cfg.model_class, 'model_state_dict:=%s' % cfg.model_state_dict,
                    ]
        # print(cli_args)
        roslaunch_args = cli_args[1:]
        roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
        parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file, force_log=True)
        # parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file, verbose=True, force_log=True)
        parent.start()
        parent.spin()
        print('SLAM evaluation on %s finished.' % name)


def eval_baselines():
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
        eval_cfg.loss_eval_csv = os.path.join(cfg.log_dir, 'loss_eval_%s.csv' % test_loss)
        eval_loss(cfg=eval_cfg)

    # Generate SLAM poses as well.
    for slam in slams:
        for name in ds:
            eval_cfg = cfg.copy()
            eval_cfg.test_names = [name]
            eval_cfg.slam = slam
            eval_cfg.slam_eval_csv = os.path.join(cfg.log_dir, 'slam_eval_%s.csv' % slam)
            # Output SLAM poses to structure within log dir.
            # eval_cfg.slam_poses_csv = os.path.join(cfg.log_dir, name, 'slam_poses_%s.csv' % slam)
            eval_cfg.slam_poses_csv = slam_poses_csv(cfg, name, slam)
            os.makedirs(os.path.dirname(eval_cfg.slam_poses_csv), exist_ok=True)
            eval_slam(cfg=eval_cfg)


def train_and_eval(pose_provider=None):
    base_cfg = Config()
    base_cfg.log_dir = base_cfg.get_log_dir()
    # Allow correction of individual pose if poses are provided by SLAM.
    if pose_provider:
        base_cfg.pose_correction = PoseCorrection.pose
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
                if pose_provider:
                    cfg.train_poses_path = [slam_poses_csv(base_cfg, name, pose_provider) for name in train_names]
                    cfg.val_poses_path = [slam_poses_csv(base_cfg, name, pose_provider) for name in val_names]
                    cfg.test_poses_path = [slam_poses_csv(base_cfg, name, pose_provider) for name in test_names]
                cfg.loss = loss
                # desc = '%s_%s_split_%i' % (cfg.model_class.lower(), cfg.loss.lower(), i_split)
                cfg.log_dir = os.path.join(base_cfg.log_dir,
                                           '%s_%s_%s' % (pose_provider if pose_provider else 'gt',
                                                         cfg.model_class.lower(), cfg.loss.lower()),
                                           'fold_%i' % i_fold)
                os.makedirs(cfg.log_dir, exist_ok=True)
                print(cfg.log_dir)
                best_cfg = train(cfg)
                # evaluate consistency loss on test (train, validation) set
                # Use ground-truth poses for evaluation.
                for split, suffix in zip([train_names, val_names, test_names],
                                         ['train', 'val', 'test']):
                    for test_loss in losses:
                        eval_cfg = best_cfg.copy()
                        eval_cfg.test_names = split
                        eval_cfg.train_poses_path = []
                        eval_cfg.val_poses_path = []
                        eval_cfg.test_poses_path = []
                        eval_cfg.loss = test_loss
                        eval_cfg.loss_eval_csv = os.path.join(cfg.log_dir,
                                                              'loss_eval_%s_%s.csv' % (test_loss, suffix))
                        eval_loss(cfg=eval_cfg)

                    # evaluate slam localization on test (train, validation) set
                    for slam in slams:
                        eval_cfg = best_cfg.copy()
                        eval_cfg.test_names = split
                        eval_cfg.train_poses_path = []
                        eval_cfg.val_poses_path = []
                        eval_cfg.test_poses_path = []
                        eval_cfg.slam = slam
                        eval_cfg.slam_eval_csv = os.path.join(cfg.log_dir,
                                                              'slam_eval_%s_%s.csv' % (slam, suffix))
                        eval_slam(cfg=eval_cfg)


def run_experiments():
    eval_baselines()
    train_and_eval()
    for slam in slams:
        train_and_eval(pose_provider=slam)


def main():
    run_experiments()


if __name__ == '__main__':
    main()

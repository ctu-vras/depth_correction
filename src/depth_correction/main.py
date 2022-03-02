from __future__ import absolute_import, division, print_function
from .config import Config, PoseCorrection
from argparse import ArgumentParser
from collections import deque
from data.asl_laser import Dataset, dataset_names
from itertools import product
import os
import random
from subprocess import DEVNULL, PIPE, run
"""Launch all experiments.

Generated files:
    eval_baselines:
        gen/<preprocessing>/asl_laser/<dataset>/slam_poses_<slam>.csv
        gen/<preprocessing>/loss_eval_<loss>.csv
        gen/<preprocessing>/slam_eval_<slam>.csv
    train:
        gen/<preprocessing>/<poses>_<model>_<loss>/<fold>/train.yaml
        gen/<preprocessing>/<poses>_<model>_<loss>/<fold>/best.yaml
        gen/<preprocessing>/<poses>_<model>_<loss>/<fold>/<iter>_*_pose_deltas.pth
        gen/<preprocessing>/<poses>_<model>_<loss>/<fold>/<iter>_*_state_dict.pth
        gen/<preprocessing>/<poses>_<model>_<loss>/<fold>/events.out.tfevents.*
    eval_loss:
        gen/<preprocessing>/<poses>_<model>_<loss>/<fold>/loss_eval_<loss>_<subset>.csv
    eval_slam:
        gen/<preprocessing>/<poses>_<model>_<loss>/<fold>/slam_eval_<loss>_<subset>.csv
"""

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
pose_providers = [None] + slams

# Debug
# splits = [[['asl_laser/eth'], ['asl_laser/stairs'], ['asl_laser/gazebo_winter']]]
# models = ['ScaledPolynomial']
# losses = ['min_eigval_loss']
# slams = ['ethzasl_icp_mapper']


def cmd_out(cmd):
    # https://stackoverflow.com/questions/89228/calling-an-external-command-in-python
    # out = run(cmd, check=True, stdout=PIPE, stderr=DEVNULL).stdout.decode()
    ret = run(cmd, check=True, stdout=PIPE, stderr=DEVNULL)
    out = ret.stdout.decode()
    err = ret.stderr.decode()
    return out, err


def slam_poses_csv(cfg: Config, name, slam):
    # path = os.path.join(cfg.get_log_dir(), name, 'slam_poses_%s.csv' % slam)
    path = os.path.join(cfg.log_dir, name, 'slam_poses_%s.csv' % slam)
    return path


def eval_baselines(launch_prefix=None):
    # Avoid using ROS in global namespace to allow using scheduler.
    from .eval import eval_loss
    from .slam_eval import eval_slam
    # TODO: launch prefix
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


def train_and_eval_all(launch_prefix=None, num_jobs=0):

    num_exp = len(list(product(pose_providers, models, losses, enumerate(splits))))
    print('Number of experiments: %i' % num_exp)
    print('Maximum number of jobs: %i' % num_jobs)
    assert num_exp < 100
    base_port = 11311

    for i_exp, (pose_provider, model, loss, (i_fold, (train_names, val_names, test_names))) \
            in enumerate(product(pose_providers, models, losses, enumerate(splits))):
        if launch_prefix and i_exp >= num_jobs:
            print('Maximum number of jobs scheduled.')
            print()
            break
        port = base_port + i_exp
        print('Generating config:')
        print('pose provider: %s' % pose_provider)
        print('model: %s' % model)
        print('loss: %s' % loss)
        print('fold: %i' % i_fold)
        print('port: %i' % port)

        cfg = Config()
        # TODO: Configure preprocessing.
        cfg.log_dir = cfg.get_log_dir()
        cfg.ros_master_port = port

        # Allow correction of individual pose if poses are provided by SLAM.
        if pose_provider:
            cfg.train_poses_path = [slam_poses_csv(cfg, name, pose_provider) for name in train_names]
            cfg.val_poses_path = [slam_poses_csv(cfg, name, pose_provider) for name in val_names]
            cfg.test_poses_path = [slam_poses_csv(cfg, name, pose_provider) for name in test_names]
            cfg.pose_correction = PoseCorrection.pose

        cfg.model_class = model
        cfg.loss = loss

        cfg.train_names = train_names
        cfg.val_names = val_names
        cfg.test_names = test_names

        cfg.log_dir = os.path.join(cfg.log_dir,
                                   '%s_%s_%s' % (pose_provider if pose_provider else 'gt',
                                                 cfg.model_class.lower(), cfg.loss.lower()),
                                   'fold_%i' % i_fold)
        print('Log dir: %s' % cfg.log_dir)
        if os.path.exists(cfg.log_dir):
            print('Log dir already exists. Skipping.')
            print()
            continue
        os.makedirs(cfg.log_dir, exist_ok=True)
        print()

        if launch_prefix:
            # Save config and schedule batch job (via launch_prefix).
            cfg_path = os.path.join(cfg.log_dir, 'config.yaml')
            if os.path.exists(cfg_path):
                print('Skipping existing config %s.' % cfg_path)
                continue
            cfg.to_yaml(cfg_path)
            launch_prefix_parts = launch_prefix.split(' ')
            cmd = launch_prefix_parts + ['python', '-m', 'depth_correction.train_and_eval', '-c', cfg_path]
            print('Command line:', cmd)
            print()
            out, err = cmd_out(cmd)
            print('Output:', out)
            print('Error:', err)
            print()
        else:
            # Avoid using ROS in global namespace to allow using scheduler.
            from .train_and_eval import train_and_eval
            train_and_eval(cfg)


def run_all():
    eval_baselines()
    train_and_eval_all()


def run_from_cmdline():
    parser = ArgumentParser()
    # parser.add_argument('--launch-prefix', type=str, nargs='+')
    parser.add_argument('--launch-prefix', type=str)
    parser.add_argument('--num-jobs', type=int, default=0)  # allows debug with fewer jobs
    parser.add_argument('verb', type=str)
    args = parser.parse_args()
    print(args)
    # return
    if args.verb == 'eval_baselines':
        eval_baselines()
    elif args.verb == 'train_and_eval_all':
        train_and_eval_all(launch_prefix=args.launch_prefix, num_jobs=args.num_jobs)
    elif args.verb == 'print_config':
        print(Config().to_yaml())


def main():
    run_from_cmdline()


if __name__ == '__main__':
    main()

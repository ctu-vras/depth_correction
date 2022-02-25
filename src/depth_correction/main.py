from __future__ import absolute_import, division, print_function
from .config import Config, PoseCorrection
from .eval import eval_loss
from .slam_eval import eval_slam
from .train import train
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
        gen/<preprocessing>/<model>_<loss>/<fold>/train.yaml
        gen/<preprocessing>/<model>_<loss>/<fold>/best.yaml
        gen/<preprocessing>/<model>_<loss>/<fold>/<iter>_*_pose_deltas.pth
        gen/<preprocessing>/<model>_<loss>/<fold>/<iter>_*_state_dict.pth
        gen/<preprocessing>/<model>_<loss>/<fold>/events.out.tfevents.*
    eval_loss:
        gen/<preprocessing>/<model>_<loss>/<fold>/loss_eval_<loss>_<subset>.csv
    eval_slam:
        gen/<preprocessing>/<model>_<loss>/<fold>/slam_eval_<loss>_<subset>.csv
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


def train_and_eval(cfg: Config):
    best_cfg = train(cfg)
    # Evaluate consistency loss on all subsets.
    # Use ground-truth poses for evaluation.
    for split, suffix in zip([cfg.train_names, cfg.val_names, cfg.test_names],
                             ['train', 'val', 'test']):
        for loss in losses:
            eval_cfg = best_cfg.copy()
            eval_cfg.test_names = split
            eval_cfg.train_poses_path = []
            eval_cfg.val_poses_path = []
            eval_cfg.test_poses_path = []
            eval_cfg.loss = loss
            eval_cfg.loss_eval_csv = os.path.join(cfg.log_dir,
                                                  'loss_eval_%s_%s.csv' % (loss, suffix))
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


def train_and_eval_batch():
    for pose_provider, model, loss, (i_fold, (train_names, val_names, test_names)) \
            in product(pose_providers, models, losses, enumerate(splits)):
        print('Generating config for')
        print('    pose provider: %s' % pose_provider)
        print('    model: %s' % model)
        print('    loss: %s' % loss)
        print('    fold: %i' % i_fold)

        cfg = Config()
        # TODO: Configure preprocessing.
        cfg.log_dir = cfg.get_log_dir()

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
        os.makedirs(cfg.log_dir, exist_ok=True)
        print(cfg.log_dir)

        # Train
        train_and_eval(cfg)

        # TODO: Save config and schedule batch job.


def run_all():
    eval_baselines()
    train_and_eval_batch()


def run_from_cmdline():
    parser = ArgumentParser()
    parser.add_argument('verb', type=str)
    args = parser.parse_args()
    if args.verb == 'eval_baselines':
        eval_baselines()
    elif args.verb == 'train_and_eval_batch':
        train_and_eval_batch()


def main():
    run_from_cmdline()


if __name__ == '__main__':
    main()

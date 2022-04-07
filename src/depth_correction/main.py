from __future__ import absolute_import, division, print_function
from .config import (
    Config,
    Loss,
    Model,
    PoseCorrection,
    PoseProvider,
    SLAM,
    slam_eval_bag,
    slam_eval_csv,
    slam_poses_csv
)
from argparse import ArgumentParser
from collections import deque
from glob import glob
from itertools import product
import os
import random
from subprocess import DEVNULL, PIPE, run
import sys
import importlib
"""Launch all experiments.

Generated files:
    eval_baselines:
        gen/<preprocessing>/asl_laser/<dataset>/slam_poses_<slam>.csv
        gen/<preprocessing>/loss_eval_<loss>.csv
        gen/<preprocessing>/slam_eval_<slam>.csv
    train:
        gen/<preprocessing>/<pose_provider>_<model>_<loss>/<split>/train.yaml
        gen/<preprocessing>/<pose_provider>_<model>_<loss>/<split>/best.yaml
        gen/<preprocessing>/<pose_provider>_<model>_<loss>/<split>/<iter>_*_pose_deltas.pth
        gen/<preprocessing>/<pose_provider>_<model>_<loss>/<split>/<iter>_*_state_dict.pth
        gen/<preprocessing>/<pose_provider>_<model>_<loss>/<split>/events.out.tfevents.*
    eval_loss:
        gen/<preprocessing>/<pose_provider>_<model>_<loss>/<split>/loss_eval_<loss>_<subset>.csv
    eval_slam:
        gen/<preprocessing>/<pose_provider>_<model>_<loss>/<split>/slam_eval_<loss>_<subset>.csv
"""


def create_splits(dataset='asl_laser', num_splits=4):
    # TODO: Generate multiple splits.
    imported_module = importlib.import_module("data.%s" % dataset)
    dataset_names = getattr(imported_module, "dataset_names")
    ds = ['%s/%s' % (dataset, name) for name in dataset_names]
    shift = len(ds) // num_splits
    splits = []
    n_dats = len(dataset_names)
    assert n_dats % 4 == 0

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
        splits.append([ds_list[:2*n_dats//4], ds_list[2*n_dats//4:3*n_dats//4], ds_list[3*n_dats//4:]])
    # for split in splits:
    #     print(split)
    return splits


def cmd_out(cmd):
    # https://stackoverflow.com/questions/89228/calling-an-external-command-in-python
    # out = run(cmd, check=True, stdout=PIPE, stderr=DEVNULL).stdout.decode()
    ret = run(cmd, capture_output=True, check=True)
    out = ret.stdout.decode() if ret.stdout else ''
    err = ret.stderr.decode() if ret.stderr else ''
    return out, err


def eval_baselines(base_cfg: Config=None):

    imported_module = importlib.import_module("data.%s" % base_cfg.dataset)
    dataset_names = getattr(imported_module, "dataset_names")
    ds = ['%s/%s' % (base_cfg.dataset, name) for name in dataset_names]

    base_cfg.log_dir = base_cfg.get_log_dir()

    # Generate SLAM poses as well.
    for i_exp, (slam, name) in enumerate(product(SLAM, ds)):

        if base_cfg.launch_prefix and i_exp >= base_cfg.num_jobs:
            print('Maximum number of jobs scheduled.')
            print()
            break

        port = base_cfg.ros_master_port + i_exp
        print('Generating config:')
        print('slam: %s' % slam)
        print('dataset: %s' % name)
        print('port: %i' % port)

        eval_cfg = base_cfg.copy()
        eval_cfg.log_dir = eval_cfg.get_log_dir()
        os.makedirs(eval_cfg.log_dir, exist_ok=True)
        eval_cfg.model_class = 'BaseModel'
        eval_cfg.model_state_dict = ''
        eval_cfg.ros_master_port = port
        eval_cfg.log_dir = os.path.join(base_cfg.log_dir, name)
        os.makedirs(eval_cfg.log_dir, exist_ok=True)
        eval_cfg.test_names = [name]
        eval_cfg.slam = slam
        # eval_cfg.slam_eval_csv = os.path.join(eval_cfg.log_dir, 'slam_eval_%s.csv' % slam)
        eval_cfg.slam_eval_csv = slam_eval_csv(eval_cfg.log_dir, slam)
        os.makedirs(os.path.dirname(eval_cfg.slam_eval_csv), exist_ok=True)

        # Output bag files from evaluation.
        # eval_cfg.slam_eval_bag = os.path.join(eval_cfg.log_dir, 'slam_eval_%s.bag' % slam)
        eval_cfg.slam_eval_bag = slam_eval_bag(eval_cfg.log_dir, slam)
        os.makedirs(os.path.dirname(eval_cfg.slam_eval_bag), exist_ok=True)

        # Output SLAM poses to structure within log dir.
        # eval_cfg.slam_poses_csv = slam_poses_csv(cfg, name, slam)
        # eval_cfg.slam_poses_csv = os.path.join(base_cfg.log_dir, name, 'slam_poses_%s.csv' % slam)
        eval_cfg.slam_poses_csv = slam_poses_csv(base_cfg.log_dir, name, slam)
        os.makedirs(os.path.dirname(eval_cfg.slam_poses_csv), exist_ok=True)

        if base_cfg.launch_prefix:
            # Save config and schedule batch job (via launch_prefix).
            # cfg_path = os.path.join(cfg.log_dir, name, 'slam_eval_%s.yaml' % slam)
            cfg_path = os.path.join(eval_cfg.log_dir, 'slam_eval_%s.yaml' % slam)
            if os.path.exists(cfg_path):
                print('Skipping existing config %s.' % cfg_path)
                continue
            eval_cfg.to_yaml(cfg_path)
            launch_prefix_parts = base_cfg.launch_prefix.format(log_dir=base_cfg.log_dir, name=name, slam=slam).split(' ')
            cmd = launch_prefix_parts + ['python', '-m', 'depth_correction.eval', '-c', cfg_path, 'slam']
            print('Command line:', cmd)
            print()
            out, err = cmd_out(cmd)
            print('Output:', out)
            print('Error:', err)
            print()

        else:
            # Avoid using ROS in global namespace to allow using scheduler.
            from .eval import eval_slam
            eval_slam(cfg=eval_cfg)


def train_and_eval_all(base_cfg: Config=None):

    splits = create_splits(dataset=base_cfg.dataset)

    num_exp = len(list(product(PoseProvider, Model, Loss, enumerate(splits))))
    print('Number of experiments: %i' % num_exp)
    print('Maximum number of jobs: %i' % base_cfg.num_jobs)
    assert num_exp < 100

    for i_exp, (pose_provider, model, loss, (i_split, (train_names, val_names, test_names))) \
            in enumerate(product(PoseProvider, Model, Loss, enumerate(splits))):

        if base_cfg.launch_prefix and i_exp >= base_cfg.num_jobs:
            print('Maximum number of jobs scheduled.')
            print()
            break

        port = base_cfg.ros_master_port + i_exp
        print('Generating config:')
        print('pose provider: %s' % pose_provider)
        print('dataset: %s' % base_cfg.dataset)
        print('model: %s' % model)
        print('loss: %s' % loss)
        print('split: %i' % i_split)
        print('port: %i' % port)

        cfg = base_cfg.copy()
        assert isinstance(cfg, Config)
        # TODO: Configure preprocessing.
        cfg.log_dir = cfg.get_log_dir()
        cfg.ros_master_port = port

        # Allow correction of individual pose if poses are provided by SLAM.
        if pose_provider != PoseProvider.ground_truth:
            cfg.train_poses_path = [slam_poses_csv(cfg.log_dir, name, pose_provider) for name in train_names]
            cfg.val_poses_path = [slam_poses_csv(cfg.log_dir, name, pose_provider) for name in val_names]
            cfg.test_poses_path = [slam_poses_csv(cfg.log_dir, name, pose_provider) for name in test_names]
            cfg.pose_correction = PoseCorrection.pose
        
        cfg.model_class = model
        cfg.loss = loss

        cfg.train_names = train_names
        cfg.val_names = val_names
        cfg.test_names = test_names
        cfg.log_dir = os.path.join(cfg.log_dir,
                                   '_'.join([pose_provider,
                                             cfg.model_class,
                                             cfg.get_nn_desc(),
                                             cfg.get_eigval_bounds_desc(),
                                             cfg.get_loss_desc()]),
                                   'split_%i' % i_split)
        print('Log dir: %s' % cfg.log_dir)
        if os.path.exists(cfg.log_dir):
            print('Log dir already exists. Skipping.')
            print()
            continue
        os.makedirs(cfg.log_dir, exist_ok=True)
        print()

        if base_cfg.launch_prefix:
            # Save config and schedule batch job (via launch_prefix).
            cfg_path = os.path.join(cfg.log_dir, 'config.yaml')
            if os.path.exists(cfg_path):
                print('Skipping existing config %s.' % cfg_path)
                continue
            cfg.to_yaml(cfg_path)
            launch_prefix_parts = base_cfg.launch_prefix.format(log_dir=cfg.log_dir).split(' ')
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


def eval_configs(base_cfg: Config=None, config=None, arg='all'):
    """Evaluate selected configs.

    Collect config paths using path template.
    Adjust log directory for each config to produce outputs in a separate directory.
    Evaluate SLAM pipelines using adjusted configs.

    :param launch_prefix:
    :param num_jobs:
    :param config:
    :param log_dir:
    :return:
    """
    assert isinstance(config, str)
    assert isinstance(base_cfg.log_dir, str)
    configs = glob(config)
    print('Configs to evaluate:')
    for c in configs:
        print(c)
    print()

    for i, config_path in enumerate(configs):

        if base_cfg.launch_prefix and i >= base_cfg.num_jobs:
            print('Maximum number of jobs scheduled.')
            print()
            break

        cfg = Config()
        cfg.from_yaml(config_path)
        dirname, basename = os.path.split(config_path)
        cfg.log_dir = base_cfg.log_dir.format(dirname=dirname, basename=basename)
        os.makedirs(cfg.log_dir, exist_ok=True)
        cfg.ros_master_port = base_cfg.ros_master_port + i
        if base_cfg.eigenvalue_bounds is not None:
            cfg.eigenvalue_bounds = base_cfg.eigenvalue_bounds
        if base_cfg.launch_prefix:
            # Save config and schedule batch job (via launch_prefix).
            new_path = os.path.join(cfg.log_dir, basename)
            if os.path.exists(new_path):
                print('Skipping existing config %s.' % new_path)
                continue
            cfg.to_yaml(new_path)
            launch_args = base_cfg.launch_prefix.format(log_dir=cfg.log_dir).split(' ')
            cmd = launch_args + ['python', '-m', 'depth_correction.eval', '-c', new_path, arg]
            print('Command line:', cmd)
            print()
            out, err = cmd_out(cmd)
            print('Output:', out)
            print('Error:', err)
            print()
        elif arg == 'all':
            print('Eval all')
            # Avoid using ROS in global namespace to allow using scheduler.
            from .eval import eval_loss_all, eval_slam_all
            eval_loss_all(cfg)
            eval_slam_all(cfg)
        elif arg == 'loss_all':
            from .eval import eval_loss_all
            eval_loss_all(cfg)
        elif arg == 'slam_all':
            from .eval import eval_slam_all
            eval_slam_all(cfg)


def run_all():
    eval_baselines()
    train_and_eval_all()


def run_from_cmdline():
    argv = sys.argv[1:]
    print('Command-line arguments:')
    for arg in argv:
        print(arg)
    print()
    cmd_cfg = Config()
    argv = cmd_cfg.from_args(argv)
    print('Config parsed from command line.')
    print('Non-default configuration:')
    for k, v in cmd_cfg.non_default().items():
        print('%s: %s' % (k, v))
    print()

    print('Remaining arguments:')
    for arg in argv:
        print(arg)
    print()

    parser = ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('args', type=str, nargs='+')
    args = parser.parse_args(argv)

    verb = args.args[0]
    arg = args.args[1] if len(args.args) >= 2 else None
    if verb == 'eval_baselines':
        eval_baselines(cmd_cfg)
    elif verb == 'train_and_eval_all':
        train_and_eval_all(cmd_cfg)
    elif verb == 'eval':
        print(verb, arg)
        print()
        eval_configs(cmd_cfg, config=args.config, arg=arg)
    elif verb == 'print_config':
        print(Config().to_yaml())
        print()


def main():
    run_from_cmdline()


if __name__ == '__main__':
    main()

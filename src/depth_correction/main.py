from __future__ import absolute_import, division, print_function
from .config import (
    cmd_out,
    Config,
    Loss,
    loss_eval_csv,
    Model,
    PoseCorrection,
    PoseProvider,
    SLAM,
    slam_eval_bag,
    slam_eval_csv,
    slam_poses_csv,
)
from argparse import ArgumentParser
from collections import deque
from glob import glob
from itertools import product
import os
import random
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


def eval_slam_baselines(base_cfg: Config=None):
    """Evaluate SLAM baselines (without depth correction).

    Generate config for each (sequence, slam) pair and evaluate given config.

    Log directory for both inputs and outputs:
        <out_dir>/<preproc>/<sequence>
    Inputs:
        <log_dir>/slam_eval_<slam>.yaml
    Outputs:
        <log_dir>/slam_poses_<slam>.csv
        <log_dir>/slam_eval_<slam>.csv

    :param base_cfg: Config with common parameters.
    """

    imported_module = importlib.import_module("data.%s" % base_cfg.dataset)
    dataset_names = getattr(imported_module, "dataset_names")
    ds = ['%s/%s' % (base_cfg.dataset, name) for name in dataset_names]

    # Generate SLAM poses as well.
    for i_exp, (name, slam) in enumerate(product(ds, SLAM)):

        if base_cfg.launch_prefix and i_exp >= base_cfg.num_jobs:
            print('Maximum number of jobs scheduled.')
            break

        print()
        print('Generating config...')
        print('Dataset: %s' % name)
        print('SLAM: %s' % slam)

        cfg = base_cfg.copy()
        assert isinstance(cfg, Config)
        cfg.log_dir = os.path.join(cfg.get_preproc_dir(), name)
        print('Log dir:', cfg.log_dir)
        os.makedirs(cfg.log_dir, exist_ok=True)
        cfg.model_class = 'BaseModel'
        cfg.model_args = []
        cfg.model_kwargs = {}
        cfg.model_state_dict = ''
        cfg.ros_master_port = base_cfg.ros_master_port + i_exp
        print('Port: %i' % cfg.ros_master_port)

        cfg.train_names = []
        cfg.val_names = []
        cfg.test_names = [name]
        cfg.slam = slam
        # CSV files will be generated (for all slams and losses).
        # eval_cfg.slam_eval_csv = slam_eval_csv(eval_cfg.log_dir, slam, 'test')
        cfg.slam_eval_csv = slam_eval_csv(cfg.log_dir, slam)
        os.makedirs(os.path.dirname(cfg.slam_eval_csv), exist_ok=True)

        # Output bag files from evaluation.
        cfg.slam_eval_bag = slam_eval_bag(cfg.log_dir, slam)
        os.makedirs(os.path.dirname(cfg.slam_eval_bag), exist_ok=True)

        # Output SLAM poses to structure within log dir.
        cfg.slam_poses_csv = slam_poses_csv(cfg.log_dir, None, slam)
        os.makedirs(os.path.dirname(cfg.slam_poses_csv), exist_ok=True)

        # Save config for debug and possibly for scheduling (via launch_prefix).
        cfg_path = os.path.join(cfg.log_dir, 'slam_eval_%s.yaml' % slam)
        if os.path.exists(cfg_path):
            print('Skipping existing config %s.' % cfg_path)
            continue
        cfg.to_yaml(cfg_path)

        if cfg.launch_prefix:
            out_path = os.path.join(cfg.log_dir, 'slam_eval_%s.out.txt' % slam)
            err_path = os.path.join(cfg.log_dir, 'slam_eval_%s.err.txt' % slam)
            launch_prefix = cfg.launch_prefix.format(log_dir=base_cfg.log_dir, name=name, slam=slam,
                                                     out=out_path, err=err_path)
            launch_prefix = launch_prefix.split(' ')
            cmd = launch_prefix + ['python', '-m', 'depth_correction.eval', '-c', cfg_path, 'eval_slam']
            print()
            print('Command line:', cmd)
            out, err = cmd_out(cmd)
            # with open(out_path, 'w') as out, open(err_path, 'w') as err:
            #     out, err = cmd_out(cmd, stdout=out, stderr=err)
            if out:
                print('Output:', out)
            if err:
                print('Error:', err)
        else:
            # Avoid using ROS in global namespace to allow using scheduler.
            from .eval import eval_slam
            eval_slam(cfg=cfg)


def eval_loss_baselines(base_cfg: Config=None):
    """Evaluate loss baselines (without depth correction).

    Generate config for each (sequence, loss) pair and evaluate given config.

    Log directory for both inputs and outputs:
        <out_dir>/<preproc>/<sequence>
    Inputs:
        <log_dir>/loss_eval_<loss>.yaml
    Outputs:
        <log_dir>/loss_eval_<loss>.csv

    :param base_cfg: Config with common parameters.
    """

    imported_module = importlib.import_module("data.%s" % base_cfg.dataset)
    dataset_names = getattr(imported_module, "dataset_names")
    ds = ['%s/%s' % (base_cfg.dataset, name) for name in dataset_names]

    # Generate SLAM poses as well.
    for i_exp, (name, loss) in enumerate(product(ds, Loss)):

        if base_cfg.launch_prefix and i_exp >= base_cfg.num_jobs:
            print('Maximum number of jobs scheduled.')
            break

        print()
        print('Generating config...')
        print('Dataset: %s' % name)
        print('Loss: %s' % loss)

        cfg = base_cfg.copy()
        assert isinstance(cfg, Config)
        cfg.log_dir = os.path.join(cfg.get_preproc_dir(), name)
        print('Log dir:', cfg.log_dir)
        os.makedirs(cfg.log_dir, exist_ok=True)
        cfg.model_class = 'BaseModel'
        cfg.model_args = []
        cfg.model_kwargs = {}
        cfg.model_state_dict = ''
        cfg.ros_master_port = base_cfg.ros_master_port + i_exp
        print('Port: %i' % cfg.ros_master_port)

        cfg.train_names = []
        cfg.val_names = []
        cfg.test_names = [name]
        cfg.loss = loss
        cfg.loss_eval_csv = loss_eval_csv(cfg.log_dir, loss)
        os.makedirs(os.path.dirname(cfg.loss_eval_csv), exist_ok=True)

        # Save config for debug and possibly for scheduling (via launch_prefix).
        cfg_path = os.path.join(cfg.log_dir, 'loss_eval_{loss}.yaml'.format(loss=loss))
        if os.path.exists(cfg_path):
            print('Skipping existing config %s.' % cfg_path)
            continue
        cfg.to_yaml(cfg_path)

        if cfg.launch_prefix:
            out_path = os.path.join(cfg.log_dir, 'loss_eval_%s.out.txt' % loss)
            err_path = os.path.join(cfg.log_dir, 'loss_eval_%s.err.txt' % loss)
            launch_prefix = cfg.launch_prefix.format(log_dir=base_cfg.log_dir, name=name, loss=loss,
                                                     out=out_path, err=err_path)
            launch_prefix = launch_prefix.split(' ')
            cmd = launch_prefix + ['python', '-m', 'depth_correction.eval', '-c', cfg_path, 'eval_loss']
            print()
            print('Command line:', cmd)
            out, err = cmd_out(cmd)
            # with open(out_path, 'w') as out, open(err_path, 'w') as err:
            #     out, err = cmd_out(cmd, stdout=out, stderr=err)
            if out:
                print('Output:', out)
            if err:
                print('Error:', err)

        else:
            # Avoid using ROS in global namespace to allow using scheduler.
            from .eval import eval_loss
            eval_loss(cfg=cfg)


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

        cfg = base_cfg.copy()
        assert isinstance(cfg, Config)

        print('Generating config...')
        print('Pose provider: %s' % pose_provider)
        print('Dataset: %s' % cfg.dataset)
        print('Model: %s' % model)
        print('Loss: %s' % loss)
        print('Split: %i' % i_split)

        cfg.ros_master_port = base_cfg.ros_master_port + i_exp
        print('Port: %i' % cfg.ros_master_port)

        cfg.pose_provider = pose_provider
        cfg.model_class = model
        if cfg.model_class in (Model.Polynomial, Model.ScaledPolynomial):
            cfg.model_args = []
            cfg.model_kwargs = {}
            cfg.model_kwargs['w'] = [0.0]
            cfg.model_kwargs['exponent'] = [4.0]
            cfg.model_kwargs['learnable_exponents'] = False
        cfg.loss = loss
        cfg.train_names = train_names
        cfg.val_names = val_names
        cfg.test_names = test_names
        cfg.log_dir = os.path.join(cfg.get_exp_dir(), 'split_%i' % i_split)
        print('Log dir: %s' % cfg.log_dir)
        os.makedirs(cfg.log_dir, exist_ok=True)

        # Allow correction of individual pose if poses are provided by SLAM.
        if cfg.pose_provider != PoseProvider.ground_truth:
            cfg.train_poses_path = [slam_poses_csv(base_cfg.log_dir, name, cfg.pose_provider) for name in cfg.train_names]
            cfg.val_poses_path = [slam_poses_csv(base_cfg.log_dir, name, cfg.pose_provider) for name in cfg.val_names]
            cfg.test_poses_path = [slam_poses_csv(base_cfg.log_dir, name, cfg.pose_provider) for name in cfg.test_names]
            cfg.pose_correction = PoseCorrection.pose

        if cfg.launch_prefix:
            # Save config and schedule batch job (via launch_prefix).
            out_path = os.path.join(cfg.log_dir, 'train_and_eval.out.txt')
            err_path = os.path.join(cfg.log_dir, 'train_and_eval.err.txt')
            cfg_path = os.path.join(cfg.log_dir, 'config.yaml')
            if os.path.exists(cfg_path):
                print('Skipping existing config %s.' % cfg_path)
                continue
            cfg.to_yaml(cfg_path)
            launch_prefix_parts = cfg.launch_prefix.format(log_dir=cfg.log_dir,
                                                           out=out_path, err=err_path).split(' ')
            cmd = launch_prefix_parts + ['python', '-m', 'depth_correction.train_and_eval', '-c', cfg_path]
            print()
            print('Command line:', cmd)
            out, err = cmd_out(cmd)
            # with open(out_path, 'w') as out, open(err_path, 'w') as err:
            #     out, err = cmd_out(cmd, stdout=out, stderr=err)
            if out:
                print('Output:', out)
            if err:
                print('Error:', err)
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
                # print('Skipping existing config %s.' % new_path)
                # continue
                new_cfg = Config()
                new_cfg.from_yaml(new_path)
                diff = cfg.diff(new_cfg)
                if diff and not base_cfg.force:
                    print('Skipping due to difference from existing config %s: %s.' % (new_path, diff))
                    continue
                elif diff:
                    print('Ignoring conflicting config %s: %s.' % (new_path, diff))
                else:
                    print('No difference to existing config %s.' % new_path)
            else:
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


def main():
    argv = sys.argv[1:]
    print('Command-line arguments:')
    for arg in argv:
        print(arg)
    print()
    cmd_cfg = Config()
    default_log_dir = cmd_cfg.log_dir
    argv = cmd_cfg.from_args(argv)
    if cmd_cfg.log_dir == default_log_dir:
        cmd_cfg.log_dir = cmd_cfg.get_preproc_dir()
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
    if verb == 'eval_loss_baselines':
        eval_loss_baselines(cmd_cfg)
    elif verb == 'eval_slam_baselines':
        eval_slam_baselines(cmd_cfg)
    elif verb == 'train_and_eval_all':
        train_and_eval_all(cmd_cfg)
    elif verb == 'eval':
        print(verb, arg)
        print()
        eval_configs(cmd_cfg, config=args.config, arg=arg)
    elif verb == 'print_config':
        print(Config().to_yaml())
        print()


if __name__ == '__main__':
    main()

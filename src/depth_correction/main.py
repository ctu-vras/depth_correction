from __future__ import absolute_import, division, print_function
from .config import Config, Loss, Model, PoseCorrection, PoseProvider, SLAM
from argparse import ArgumentParser
from collections import deque
from glob import glob
from itertools import product
import os
import random
from subprocess import DEVNULL, PIPE, run
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


def slam_poses_csv(cfg: Config, name, slam):
    # path = os.path.join(cfg.get_log_dir(), name, 'slam_poses_%s.csv' % slam)
    path = os.path.join(cfg.log_dir, name, 'slam_poses_%s.csv' % slam)
    return path


def eval_baselines(launch_prefix=None, dataset='asl_laser'):
    # Avoid using ROS in global namespace to allow using scheduler.
    from .eval import eval_loss, eval_slam
    # TODO: launch prefix
    # evaluate consistency loss on all sequences
    cfg = Config()
    # Adjust default config...
    cfg.dataset = dataset
    cfg.model_class = 'BaseModel'
    cfg.model_state_dict = ''
    cfg.log_dir = cfg.get_log_dir()
    os.makedirs(cfg.log_dir, exist_ok=True)

    imported_module = importlib.import_module("data.%s" % cfg.dataset)
    dataset_names = getattr(imported_module, "dataset_names")
    ds = ['%s/%s' % (dataset, name) for name in dataset_names]

    for test_loss in Loss:
        eval_cfg = cfg.copy()
        eval_cfg.test_names = ds
        eval_cfg.loss = test_loss
        eval_cfg.loss_eval_csv = os.path.join(cfg.log_dir, 'loss_eval_%s.csv' % test_loss)
        eval_loss(cfg=eval_cfg)

    # Generate SLAM poses as well.
    for slam in SLAM:
        for name in ds:
            eval_cfg = cfg.copy()
            eval_cfg.test_names = [name]
            eval_cfg.slam = slam
            eval_cfg.slam_eval_csv = os.path.join(cfg.log_dir, 'slam_eval_%s.csv' % slam)
            eval_cfg.slam_eval_bag = os.path.join(cfg.log_dir, 'slam_eval_%s.bag' % slam)
            # Output SLAM poses to structure within log dir.
            eval_cfg.slam_poses_csv = slam_poses_csv(cfg, name, slam)
            os.makedirs(os.path.dirname(eval_cfg.slam_poses_csv), exist_ok=True)
            eval_slam(cfg=eval_cfg)


def train_and_eval_all(launch_prefix=None, num_jobs=0, dataset='asl_laser'):

    splits = create_splits(dataset=dataset)

    num_exp = len(list(product(PoseProvider, Model, Loss, enumerate(splits))))
    print('Number of experiments: %i' % num_exp)
    print('Maximum number of jobs: %i' % num_jobs)
    assert num_exp < 100
    base_port = Config().ros_master_port

    for i_exp, (pose_provider, model, loss, (i_split, (train_names, val_names, test_names))) \
            in enumerate(product(PoseProvider, Model, Loss, enumerate(splits))):
        if launch_prefix and i_exp >= num_jobs:
            print('Maximum number of jobs scheduled.')
            print()
            break
        port = base_port + i_exp
        print('Generating config:')
        print('pose provider: %s' % pose_provider)
        print('dataset: %s' % dataset)
        print('model: %s' % model)
        print('loss: %s' % loss)
        print('split: %i' % i_split)
        print('port: %i' % port)

        cfg = Config()
        # TODO: Configure preprocessing.
        cfg.dataset = dataset
        cfg.log_dir = cfg.get_log_dir()
        cfg.ros_master_port = port

        # Allow correction of individual pose if poses are provided by SLAM.
        if pose_provider != PoseProvider.ground_truth:
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
                                   '%s_%s_%s' % (pose_provider, cfg.model_class, cfg.loss),
                                   'split_%i' % i_split)
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
            launch_prefix = launch_prefix.format(log_dir=cfg.log_dir)
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


def eval_configs(launch_prefix=None, num_jobs=0, config=None, log_dir=None, arg='all', eigenvalue_bounds=None):
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
    assert isinstance(log_dir, str)
    configs = glob(config)
    print(configs)
    base_port = Config().ros_master_port

    for i, config_path in enumerate(configs):

        if launch_prefix and i >= num_jobs:
            print('Maximum number of jobs scheduled.')
            print()
            break

        cfg = Config()
        cfg.from_yaml(config_path)
        dirname, basename = os.path.split(config_path)
        cfg.log_dir = log_dir.format(dirname=dirname, basename=basename)
        os.makedirs(cfg.log_dir, exist_ok=True)
        cfg.ros_master_port = base_port + i
        if eigenvalue_bounds is not None:
            cfg.eigenvalue_bounds = eigenvalue_bounds
        if launch_prefix:
            # Save config and schedule batch job (via launch_prefix).
            new_path = os.path.join(cfg.log_dir, basename)
            if os.path.exists(new_path):
                print('Skipping existing config %s.' % new_path)
                continue
            cfg.to_yaml(new_path)
            launch_args = launch_prefix.format(log_dir=cfg.log_dir).split(' ')
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
            # eval_all(cfg)
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
    parser = ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--log-dir', type=str)
    # parser.add_argument('--launch-prefix', type=str, nargs='+')
    parser.add_argument('--launch-prefix', type=str)
    parser.add_argument('--num-jobs', type=int, default=0)  # allows debug with fewer jobs
    parser.add_argument('--eigenvalue-bounds', type=str)
    parser.add_argument('--dataset', type=str, default='asl_laser')
    parser.add_argument('args', type=str, nargs='+')
    # parser.add_argument('arg', type=str, required=False)
    args = parser.parse_args()
    print(args)
    verb = args.args[0]
    arg = args.args[1] if len(args.args) >= 2 else None
    # return
    if verb == 'eval_baselines':
        eval_baselines(dataset=args.dataset)
    elif verb == 'train_and_eval_all':
        train_and_eval_all(launch_prefix=args.launch_prefix, num_jobs=args.num_jobs, dataset=args.dataset)
    elif verb == 'eval':
        print(verb, arg)
        kwargs = {}
        if args.eigenvalue_bounds:
            kwargs['eigenvalue_bounds'] = args.eigenvalue_bounds
        eval_configs(launch_prefix=args.launch_prefix, num_jobs=args.num_jobs,
                     config=args.config, log_dir=args.log_dir, arg=arg, **kwargs)
    elif verb == 'print_config':
        print(Config().to_yaml())


def main():
    run_from_cmdline()


if __name__ == '__main__':
    main()

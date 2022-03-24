from __future__ import absolute_import, division, print_function
from .config import Config, PoseCorrection
from .filters import filter_eigenvalues
from .io import append
from .loss import min_eigval_loss, trace_loss
from .model import *
from .preproc import *
from .ros import *
from argparse import ArgumentParser
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter


def eval_loss(cfg: Config):
    """Evaluate loss of model on test sequences.

    :param cfg:
    """
    os.makedirs(cfg.log_dir, exist_ok=True)
    cfg_path = os.path.join(cfg.log_dir, 'eval.yaml')
    if os.path.exists(cfg_path):
        print('Config %s already exists.' % cfg_path)
    else:
        cfg.to_yaml(cfg_path)

    assert cfg.dataset == 'asl_laser' or cfg.dataset == 'semantic_kitti'
    if cfg.dataset == 'asl_laser':
        from data.asl_laser import Dataset
    elif cfg.dataset == 'semantic_kitti':
        from data.semantic_kitti import Dataset

    model = load_model(cfg=cfg, eval_mode=True)

    if cfg.pose_correction != PoseCorrection.none:
        print('Pose deltas not used.')

    loss_fun = eval(cfg.loss)
    assert callable(loss_fun)

    # TODO: Process individual sequences separately.
    for i, name in enumerate(cfg.test_names):
        # Allow overriding poses paths, assume valid if non-empty.
        poses_path = cfg.test_poses_path[i] if cfg.test_poses_path else None
        clouds = []
        poses = []
        for cloud, pose in Dataset(name, poses_path=poses_path)[::cfg.data_step]:
            cloud = filtered_cloud(cloud, cfg)
            cloud = local_feature_cloud(cloud, cfg)
            clouds.append(cloud)
            poses.append(pose)
        poses = np.stack(poses).astype(dtype=cfg.numpy_float_type())
        poses = torch.as_tensor(poses, device=cfg.device)

        cloud = global_cloud(clouds, model, poses)
        cloud.update_all(k=cfg.nn_k, r=cfg.nn_r)
        mask = filter_eigenvalues(cloud, eig_bounds=cfg.eigenvalue_bounds,
                                  only_mask=True, log=cfg.log_filters)
        print('Testing on %.3f = %i / %i points from %s.'
              % (mask.float().mean().item(), mask.sum().item(), mask.numel(),
                 name))

        if cfg.enable_ros:
            publish_data([cloud], [poses], [name], cfg=cfg)

        test_loss, _ = loss_fun(cloud, mask=mask)
        print('Test loss on %s: %.9f' % (name, test_loss.item()))
        csv = cfg.loss_eval_csv
        assert csv
        append(csv, '%s %.9f\n' % (name, test_loss))


def eval_loss_all(cfg: Config):
    # Evaluate consistency loss and SLAM on all subsets.
    # Use ground-truth poses for evaluation.
    for names, suffix in zip([cfg.train_names, cfg.val_names, cfg.test_names],
                             ['train', 'val', 'test']):
        for loss in cfg.eval_losses:
            eval_cfg = cfg.copy()
            eval_cfg.test_names = names
            eval_cfg.train_poses_path = []
            eval_cfg.val_poses_path = []
            eval_cfg.test_poses_path = []
            eval_cfg.loss = loss
            eval_cfg.loss_eval_csv = os.path.join(cfg.log_dir,
                                                  'loss_eval_%s_%s.csv' % (loss, suffix))
            eval_loss(cfg=eval_cfg)


def eval_slam(cfg: Config):
    import roslaunch
    slam_eval_launch = os.path.join(Config().pkg_dir, 'launch', 'slam_eval.launch')

    cfg_path = os.path.join(cfg.log_dir, 'eval_slam.yaml')
    if os.path.exists(cfg_path):
        print('Config %s already exists.' % cfg_path)
    else:
        cfg.to_yaml(cfg_path)

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
        # Evaluate SLAM on whole kitti sequences if the dataset is semantic kitti
        if cfg.dataset == 'semantic_kitti':
            name = name[:17]  # semantic_kitti/XY
            # name += '_step_%i' % cfg.data_step
            name += '_end_1000_step_1'
        print('SLAM evaluation on %s started.' % name)
        # print(cfg.to_yaml())

        cli_args = [slam_eval_launch]
        cli_args.append('dataset:=%s' % name)
        if poses_path:
            cli_args.append('dataset_poses_path:=%s' % poses_path)
        cli_args.append('odom:=true')
        cli_args.append('rviz:=true' if cfg.rviz else 'rviz:=false')
        cli_args.append('slam_eval_csv:=%s' % cfg.slam_eval_csv)
        if cfg.slam_eval_bag:
            cli_args.append('record:=true')
            cli_args.append('bag:=%s' % cfg.slam_eval_bag.format(name=name.replace('/', '_')))
        if cfg.slam_poses_csv:
            cli_args.append('slam_poses_csv:=%s' % cfg.slam_poses_csv)
        cli_args.append('min_depth:=%.3f' % cfg.min_depth)
        cli_args.append('max_depth:=%.3f' % cfg.max_depth)
        cli_args.append('grid_res:=%.3f' % cfg.grid_res)
        if cfg.model_class != 'BaseModel':
            cli_args.append('depth_correction:=true')
        else:
            cli_args.append('depth_correction:=false')
        cli_args.append('nn_r:=%.3f' % cfg.nn_r)
        # TODO: Pass eigenvalue bounds to launch.
        # cli_args.append('eigenvalue_bounds:=[[0, -.inf, 0.0004], [1, 0.0025, .inf]]')
        cli_args.append('eigenvalue_bounds:=%s' % cfg.eigenvalue_bounds)
        cli_args.append('model_class:=%s' % cfg.model_class)
        cli_args.append('model_state_dict:=%s' % cfg.model_state_dict)
        # print(cli_args)
        roslaunch_args = cli_args[1:]
        roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
        launch_kwargs = {}
        launch_kwargs['force_log'] = True
        if cfg.ros_master_port:
            launch_kwargs['port'] = cfg.ros_master_port
        parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file, **launch_kwargs)
        # parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file, verbose=True, force_log=True)
        parent.start()
        parent.spin()
        print('SLAM evaluation on %s finished.' % name)


def eval_slam_all(cfg: Config):
    # Evaluate consistency loss and SLAM on all subsets.
    # Use ground-truth poses for evaluation.
    for names, suffix in zip([cfg.train_names, cfg.val_names, cfg.test_names],
                             ['train', 'val', 'test']):

        for slam in cfg.eval_slams:
            eval_cfg = cfg.copy()
            eval_cfg.test_names = names
            eval_cfg.train_poses_path = []
            eval_cfg.val_poses_path = []
            eval_cfg.test_poses_path = []
            eval_cfg.slam = slam
            eval_cfg.slam_eval_csv = os.path.join(cfg.log_dir,
                                                  'slam_eval_%s_%s.csv' % (slam, suffix))
            eval_cfg.slam_eval_bag = os.path.join(cfg.log_dir,
                                                  'slam_eval_%s_%s_{name}.bag' % (slam, suffix))
            eval_cfg.slam_poses_csv = ''
            eval_slam(cfg=eval_cfg)


def demo():
    cfg = Config()
    cfg.dataset = 'asl_laser'
    cfg.test_names = ['stairs']
    cfg.model_class = 'ScaledPolynomial'
    cfg.model_state_dict = '/home/petrito1/workspace/depth_correction/gen/2022-02-21_16-31-34/088_8.85347e-05_state_dict.pth'
    cfg.pose_correction = PoseCorrection.sequence
    eval_loss(cfg)


def run_from_cmdline():
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('arg', type=str)
    args = parser.parse_args()
    print('Arguments:')
    print(args)
    cfg = Config()
    cfg.from_yaml(args.config)
    print('Config:')
    print(cfg.to_yaml())
    print('Evaluating loss...')
    if args.arg == 'all':
        eval_loss(cfg)
        eval_slam(cfg)
    elif args.arg == 'loss':
        eval_loss(cfg)
    elif args.arg == 'loss_all':
        eval_loss_all(cfg)
    elif args.arg == 'slam':
        eval_slam(cfg)
    elif args.arg == 'slam_all':
        eval_slam_all(cfg)
    print('Evaluating loss finished.')


def main():
    run_from_cmdline()


if __name__ == '__main__':
    main()

from __future__ import absolute_import, division, print_function
from .config import Config, loss_eval_csv, PoseCorrection, slam_eval_bag, slam_eval_csv, slam_poses_csv
from .filters import filter_eigenvalues
from .io import append
from .loss import loss_by_name
from .model import load_model
from .preproc import filtered_cloud, local_feature_cloud, global_cloud
from .ros import publish_data
from argparse import ArgumentParser
import numpy as np
import os
import torch


def eval_loss(cfg: Config):
    """Evaluate loss on test sequences.

    :param cfg:
    """

    assert cfg.dataset in ('asl_laser', 'semantic_kitti')
    if cfg.dataset == 'asl_laser':
        from data.asl_laser import Dataset
    elif cfg.dataset == 'semantic_kitti':
        from data.semantic_kitti import Dataset

    model = load_model(cfg=cfg, eval_mode=True)

    if cfg.pose_correction != PoseCorrection.none:
        print('Pose deltas not used.')

    loss_fun = loss_by_name(cfg.loss)
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

        test_loss, _ = loss_fun(cloud, mask=mask, **cfg.loss_kwargs)
        print('Test loss on %s: %.9f' % (name, test_loss.item()))
        csv = cfg.loss_eval_csv
        assert csv
        append(csv, '%s %.9f\n' % (name, test_loss))


def eval_loss_all(cfg: Config):
    # Evaluate consistency loss and SLAM on all subsets.
    # Use ground-truth poses for evaluation.
    for names, suffix in zip([cfg.train_names, cfg.val_names, cfg.test_names],
                             ['train', 'val', 'test']):

        if not names:
            continue

        for loss in cfg.eval_losses:
            eval_cfg = cfg.copy()
            eval_cfg.test_names = names
            eval_cfg.train_poses_path = []
            eval_cfg.val_poses_path = []
            eval_cfg.test_poses_path = []
            eval_cfg.loss = loss
            eval_cfg.loss_eval_csv = loss_eval_csv(cfg.log_dir, loss, suffix)
            eval_loss(cfg=eval_cfg)


def eval_slam(cfg: Config):
    """Evaluate SLAM on test sequences.

    :param cfg:
    """
    import roslaunch
    slam_eval_launch = os.path.join(Config().pkg_dir, 'launch', 'slam_eval.launch')

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
            name += '_end_500_step_1'
        print('SLAM evaluation on %s started.' % name)

        cli_args = [slam_eval_launch]
        cli_args.append('dataset:=%s' % name)
        if poses_path:
            cli_args.append('dataset_poses_path:=%s' % poses_path)
        cli_args.append('odom:=true')
        if cfg.slam_eval_bag:
            cli_args.append('record:=true')
            cli_args.append('bag:=%s' % cfg.slam_eval_bag.format(name=name.replace('/', '_')))
        if cfg.model_class != 'BaseModel':
            cli_args.append('depth_correction:=true')
        else:
            cli_args.append('depth_correction:=false')

        keys_from_cfg = ['min_depth', 'max_depth', 'grid_res',
                         'nn_k', 'nn_r', 'shadow_neighborhood_angle', 'shadow_angle_bounds', 'eigenvalue_bounds',
                         'model_class', 'model_state_dict', 'slam', 'slam_eval_csv', 'slam_poses_csv', 'rviz']
        cfg_args = cfg.to_roslaunch_args(keys=keys_from_cfg)
        cli_args += cfg_args

        # print(cli_args)
        roslaunch_args = cli_args[1:]
        roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
        launch_kwargs = {}
        # launch_kwargs['force_log'] = True
        launch_kwargs['force_screen'] = True
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

        if not names:
            continue

        for slam in cfg.eval_slams:
            eval_cfg = cfg.copy()
            eval_cfg.test_names = names
            eval_cfg.train_poses_path = []
            eval_cfg.val_poses_path = []
            eval_cfg.test_poses_path = []
            eval_cfg.slam = slam
            # eval_cfg.slam_eval_bag = slam_eval_bag(cfg.log_dir, slam)
            eval_cfg.slam_eval_bag = None  # Don't record for now.
            eval_cfg.slam_eval_csv = slam_eval_csv(cfg.log_dir, slam, suffix)
            # if len(names) == 1:
            #     eval_cfg.slam_poses_csv = slam_poses_csv(cfg.log_dir, names[0], slam)
            # else:
            #     eval_cfg.slam_poses_csv = None
            eval_cfg.slam_poses_csv = None
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
    print('Evaluating...')
    if 'all' in args.arg:
        if 'loss' in args.arg:
            eval_loss_all(cfg)
        if 'slam' in args.arg:
            eval_slam_all(cfg)
    else:
        if 'loss' in args.arg:
            eval_loss(cfg)
        if 'slam' in args.arg:
            eval_slam(cfg)
    print('Evaluating finished.')


def main():
    run_from_cmdline()


if __name__ == '__main__':
    main()

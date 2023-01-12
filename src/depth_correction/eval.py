from __future__ import absolute_import, division, print_function
from .config import (
    Config,
    loss_eval_csv,
    NeighborhoodType,
    nonempty,
    PoseCorrection,
    slam_eval_csv,
)
from .dataset import create_dataset
from .depth_cloud import DepthCloud
from .io import append
from .loss import create_loss
from .model import load_model
from .preproc import (
    compute_neighborhood_features,
    establish_neighborhoods,
    global_cloud,
    global_cloud_mask,
    local_feature_cloud,
    offset_cloud,
)
from .transform import xyz_axis_angle_to_matrix
from argparse import ArgumentParser
import numpy as np
import os
import torch


def initialize_pose_corrections(datasets, cfg: Config):
    """Initialize pose correction for given datasets (sequence lengths).

    Reusing pose correction from training in validation / test must be done
    in caller.

    :param datasets: Datasets (lengths are used).
    :param cfg: Config with pose correction type, tensor data type and device.
    :return: Pose corrections.
    """
    pose_deltas = []
    kwargs = {
        'dtype': cfg.torch_float_type(),
        'device': cfg.device,
        'requires_grad': True,
    }
    for ds in datasets:
        if cfg.pose_correction == PoseCorrection.common:
            # Use a common correction for all sequences and poses, broadcast to all poses.
            if pose_deltas:
                pose_delta = pose_deltas[0]
            else:
                pose_delta = torch.zeros((1, 6), **kwargs)
        elif cfg.pose_correction == PoseCorrection.sequence:
            # Single correction per sequence (sensor rig calibration), broadcast to all poses.
            pose_delta = torch.zeros((1, 6), **kwargs)
        elif cfg.pose_correction == PoseCorrection.pose:
            # Correct every pose (e.g. from odometry or SLAM).
            pose_delta = torch.zeros((len(ds), 6), **kwargs)
        else:
            pose_delta = None

        pose_deltas.append(pose_delta)

    return pose_deltas


def create_corrected_poses(poses, pose_deltas, cfg: Config):

    if cfg.pose_correction == PoseCorrection.none:
        poses_upd = poses
    else:
        assert len(poses) == len(pose_deltas)
        # For common pose correction, there is a same correction for all sequences.
        if cfg.pose_correction == PoseCorrection.common:
            assert all(d is pose_deltas[0] for d in pose_deltas[1:])
        poses_upd = []
        for i in range(len(poses)):
            pose_deltas_mat = xyz_axis_angle_to_matrix(pose_deltas[i])
            poses_upd.append(torch.matmul(poses[i], pose_deltas_mat))

    return poses_upd


def eval_loss_clouds(clouds, poses, pose_deltas, masks, ns, model, loss_fun, cfg: Config):
    """Evaluate loss on given clouds, poses, deltas, etc."""

    offsets = [offset_cloud(c, model) for c in clouds] if cfg.loss_offset else None
    poses_upd = create_corrected_poses(poses, pose_deltas, cfg)
    global_clouds = [
        global_cloud(clouds=c, model=model if cfg.nn_type == NeighborhoodType.ball else None, poses=p)
        for c, p in zip(clouds, poses_upd)
    ]
    
    # TODO: Reset neighborhoods with the new global clouds.
    for cloud, nn in zip(global_clouds, ns):
        nn.cloud = len(nn) * [cloud]
    
    feat_clouds = [
        compute_neighborhood_features(cloud=cloud, model=model if cfg.nn_type == NeighborhoodType.plane else None,
                                      neighborhoods=nn, cfg=cfg)
        for cloud, nn in zip(global_clouds, ns)
    ]

    if cfg.loss == 'icp_loss':
        if clouds[0][0].normals is None:
            clouds = [[local_feature_cloud(cloud, cfg) for cloud in seq_clouds] for seq_clouds in clouds]

        loss, loss_cloud = loss_fun(clouds, poses_upd, model, masks=masks)

    else:
        # if (not masks or masks[0] is None) and isinstance(feat_clouds[0], DepthCloud):
        #     masks = [global_cloud_mask(cloud, cloud.mask if hasattr(cloud, 'mask') else None, cfg)
        #              for cloud in feat_clouds]

        # loss, loss_cloud = loss_fun(feat_clouds, mask=masks, offset=offsets)

        # Neighborhoods for ball and planar NN.
        loss, loss_cloud = loss_fun(feat_clouds, offset=offsets)

    return loss, loss_cloud, poses_upd, feat_clouds


def eval_loss(cfg: Config, test_datasets=None, test_ns=None, model=None, loss_fun=None,
              return_neighborhood=False):
    """Evaluate loss on test sequences.

    :param cfg: Config.
    :param test_datasets: Test datasets, created from config if not provided.
    :param test_ns: Test neighborhoods, created if needed.
    :param model: Model to use, created from config if not provided.
    :param loss_fun: Loss function, created from config if not provided.
    :param return_neighborhood: Whether to return neighborhood test_ns.
    """
    if test_datasets:
        test_names = [str(ds) for ds in test_datasets]
        print('Using provided test datasets: %s.' % ', '.join(test_names))
    else:
        print('Creating test datasets from config: %s.' % ', '.join(cfg.test_names))
        test_names = cfg.test_names
        test_datasets = []
        for i, name in enumerate(cfg.test_names):
            poses_path = cfg.test_poses_path[i] if cfg.test_poses_path else None
            ds = create_dataset(name, cfg, poses_path=poses_path)
            test_datasets.append(ds)

    if model is None:
        model = load_model(cfg=cfg, eval_mode=True)

    if loss_fun is None:
        loss_fun = create_loss(cfg)
    assert callable(loss_fun)

    test_clouds = []
    test_poses = []
    test_masks = [None] * len(test_datasets)
    for i, ds in enumerate(test_datasets):
        clouds = []
        poses = []
        for cloud, pose in ds:
            if cfg.nn_type == NeighborhoodType.ball:
                cloud = local_feature_cloud(cloud, cfg)
            else:
                cloud = DepthCloud.from_structured_array(cloud, dtype=cfg.numpy_float_type(), device=cfg.device)
            clouds.append(cloud)
            poses.append(pose)
        test_clouds.append(clouds)
        poses = np.stack(poses).astype(dtype=cfg.numpy_float_type())
        poses = torch.as_tensor(poses, device=cfg.device)
        test_poses.append(poses)

    # Initialize pose deltas, possibly from file.
    if cfg.test_poses_path and nonempty(cfg.test_poses_path):
        assert cfg.pose_correction != PoseCorrection.none
        test_pose_deltas = torch.load(cfg.test_pose_deltas, map_location=cfg.device)
    else:
        test_pose_deltas = initialize_pose_corrections(test_datasets, cfg)

    # Create test neighborhoods.
    if test_ns is None:
        test_ns = [establish_neighborhoods(clouds=clouds, poses=poses, cfg=cfg)
                   for clouds, poses in zip(test_clouds, test_poses)]

    # Compute loss.
    test_loss, _, test_poses_upd, test_feat_clouds \
            = eval_loss_clouds(test_clouds, test_poses, test_pose_deltas, test_masks, test_ns,
                               model, loss_fun, cfg)

    print('Test loss on %s: %.9f' % (', '.join(test_names), test_loss.item()))

    if cfg.loss_eval_csv:
        assert cfg.loss_eval_csv
        append(cfg.loss_eval_csv, '%s %.9f\n' % (','.join(test_names), test_loss))
        if len(test_names) > 1:
            print('Test loss on %s written to %s.' % (', '.join(test_names), cfg.loss_eval_csv))

    if return_neighborhood:
        return test_loss, test_ns
    else:
        return test_loss


def eval_loss_all(cfg: Config):
    # Evaluate consistency loss on all subsets.
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

        cli_args = [slam_eval_launch]
        cli_args.append('dataset:=%s' % name)
        if poses_path:
            cli_args.append('dataset_poses_path:=%s' % poses_path)
        cli_args.append('odom:=true')
        if cfg.slam_eval_bag:
            cli_args.append('record:=true')
            cli_args.append('bag:=%s' % cfg.slam_eval_bag.format(name=name.replace('/', '_')))
        cli_args.append('depth_correction:=true')

        keys_from_cfg = ['min_depth', 'max_depth', 'grid_res',
                         'nn_k', 'nn_r', 'shadow_neighborhood_angle', 'shadow_angle_bounds', 'eigenvalue_bounds',
                         'model_class', 'model_args', 'model_kwargs', 'model_state_dict',
                         'odom_cov', 'slam', 'slam_eval_bag', 'slam_eval_csv', 'slam_poses_csv',
                         'rviz']
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
        parent.start()
        print('SLAM evaluation on %s started.' % name)
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
            eval_cfg.slam_eval_bag = ''  # Don't record for now.
            eval_cfg.slam_eval_csv = slam_eval_csv(cfg.log_dir, slam, suffix)
            # if len(names) == 1:
            #     eval_cfg.slam_poses_csv = slam_poses_csv(cfg.log_dir, names[0], slam)
            # else:
            #     eval_cfg.slam_poses_csv = None
            eval_cfg.slam_poses_csv = ''
            eval_slam(cfg=eval_cfg)


def demo():
    cfg = Config()
    # cfg.dataset = 'asl_laser'
    # cfg.test_names = ['stairs']
    # cfg.model_class = 'ScaledPolynomial'
    # cfg.model_state_dict = '/home/petrito1/workspace/depth_correction/gen/2022-02-21_16-31-34/088_8.85347e-05_state_dict.pth'
    # cfg.pose_correction = PoseCorrection.sequence

    cfg.dataset = 'newer_college'
    cfg.test_names = [
        'newer_college/01_short_experiment/start_0_end_800_step_12',
        'newer_college/01_short_experiment/start_800_end_1600_step_12',
    ]
    cfg.test_poses_path = []
    cfg.pose_correction = PoseCorrection.common
    cfg.grid_res = 0.2
    cfg.min_depth = 1.0
    cfg.max_depth = 20.0
    cfg.nn_type = NeighborhoodType.plane
    cfg.ransac_dist_thresh = 0.03
    cfg.min_valid_neighbors = 250
    cfg.max_neighborhoods = 10
    cfg.model_class = 'ScaledPolynomial'
    cfg.log_filters = False

    eval_loss(cfg)


def run_from_cmdline():
    parser = ArgumentParser()
    # parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('arg', type=str)
    args = parser.parse_args()
    print('Arguments:')
    print(args)

    if 'demo' == args.arg:
        demo()
        return

    assert args.config
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

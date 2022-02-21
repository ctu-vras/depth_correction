#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from .config import Config, PoseCorrection
from .depth_cloud import DepthCloud
from .filters import filter_depth, filter_eigenvalues, filter_grid
from .loss import min_eigval_loss
from .model import *
from .utils import initialize_ros, timer, timing
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from nav_msgs.msg import Path
import numpy as np
from pytorch3d.transforms import (axis_angle_to_matrix,
                                  matrix_to_quaternion,
                                  quaternion_to_axis_angle,
                                  axis_angle_to_quaternion)
from ros_numpy import msgify
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import torch
from torch.utils.tensorboard import SummaryWriter


def filtered_cloud(cloud, cfg: Config):
    cloud = filter_depth(cloud, min=cfg.min_depth, max=cfg.max_depth, log=cfg.log_filters)
    cloud = filter_grid(cloud, grid_res=cfg.grid_res, keep='random', log=cfg.log_filters)
    return cloud


# @timing
def local_feature_cloud(cloud, cfg: Config):
    # Convert to depth cloud and transform.
    # dtype = eval('torch.%s' % cfg.float_type)
    # dtype = eval('np.%s' % cfg.float_type)
    cloud = DepthCloud.from_structured_array(cloud, dtype=cfg.numpy_float_type())
    cloud = cloud.to(device=cfg.device)
    # Find/update neighbors and estimate all features.
    cloud.update_all(k=cfg.nn_k, r=cfg.nn_r)
    # Select planar regions to correct in prediction phase.
    cloud.mask = filter_eigenvalues(cloud, cfg.eig_bounds, only_mask=True, log=cfg.log_filters)
    # mask = None
    # mask = filter_eigenvalue(cloud, 0, max=max_eig_0, only_mask=True, log=log_filters)
    # mask = mask & filter_eigenvalue(cloud, 1, min=min_eig_1, only_mask=True, log=log_filters)
    # cloud.mask = mask
    return cloud


def global_cloud(clouds: (list, tuple),
                 model: BaseModel,
                 poses: torch.Tensor):
    """Create global cloud with corrected depth.

    :param clouds: Filtered local features clouds.
    :param model: Depth correction model, directly applicable to clouds.
    :param poses: N-by-4-by-4 pose tensor to transform clouds to global frame.
    :return: Global cloud with corrected depth.
    """
    transformed_clouds = []
    for i, cloud in enumerate(clouds):
        cloud = model(cloud)
        cloud = cloud.transform(poses[i])
        transformed_clouds.append(cloud)
    cloud = DepthCloud.concatenate(transformed_clouds, True)
    # cloud.visualize(colors='z')
    # cloud.visualize(colors='inc_angles')
    return cloud


def global_clouds(clouds, model, poses):
    ret = []
    for c, p in zip(clouds, poses):
        cloud = global_cloud(c, model, p)
        ret.append(cloud)
    return ret


def cloud_to_ros_msg(dc, frame_id, stamp):
    pc_msg = msgify(PointCloud2, dc.to_structured_array())
    pc_msg.header.frame_id = frame_id
    pc_msg.header.stamp = stamp
    return pc_msg


def xyz_axis_angle_to_pose_msg(xyz_axis_angle):
    # assert isinstance(xyz_axis_angle, torch.Tensor)
    # assert isinstance(xyz_axis_angle, list)
    q = axis_angle_to_quaternion(xyz_axis_angle[3:])
    msg = Pose(Point(*xyz_axis_angle[:3]), Quaternion(w=q[0], x=q[1], y=q[2], z=q[3]))
    return msg


# @timing
def xyz_axis_angle_to_path_msg(xyz_axis_angle, frame_id, stamp):
    assert isinstance(xyz_axis_angle, torch.Tensor)
    assert xyz_axis_angle.dim() == 2
    assert xyz_axis_angle.shape[-1] == 6
    xyz_axis_angle = xyz_axis_angle.detach()
    msg = Path()
    msg.poses = [PoseStamped(Header(), xyz_axis_angle_to_pose_msg(p)) for p in xyz_axis_angle]
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    return msg


def xyz_axis_angle_to_matrix(xyz_axis_angle):
    assert isinstance(xyz_axis_angle, torch.Tensor)
    assert xyz_axis_angle.shape[-1] == 6

    mat = torch.zeros(xyz_axis_angle.shape[:-1] + (4, 4), dtype=xyz_axis_angle.dtype, device=xyz_axis_angle.device)
    mat[..., :3, :3] = axis_angle_to_matrix(xyz_axis_angle[..., 3:])
    mat[..., :3, 3] = xyz_axis_angle[..., :3]
    mat[..., 3, 3] = 1.
    assert mat.shape == xyz_axis_angle.shape[:-1] + (4, 4)
    # assert mat.shape[-2:] == (4, 4)
    return mat


def matrix_to_xyz_axis_angle(T):
    assert isinstance(T, torch.Tensor)
    assert T.dim() == 3
    assert T.shape[1:] == (4, 4)
    n_poses = len(T)
    q = matrix_to_quaternion(T[:, :3, :3])
    axis_angle = quaternion_to_axis_angle(q)
    xyz = T[:, :3, 3]
    poses = torch.concat([xyz, axis_angle], dim=1)
    assert poses.shape == (n_poses, 6)
    return poses


@timing
def publish_data(clouds: list, poses: list, cfg: Config):
    assert isinstance(clouds[0], DepthCloud)
    assert isinstance(poses[0], torch.Tensor)

    stamp = rospy.Time.now()
    for i, cloud in enumerate(clouds):
        poses_pub = rospy.Publisher('poses_%s' % cfg.val_names[i], Path, queue_size=2)
        dc_pub = rospy.Publisher('global_cloud_%s' % cfg.val_names[i], PointCloud2, queue_size=2)
        pc_opt_msg = cloud_to_ros_msg(cloud, frame_id=cfg.world_frame, stamp=stamp)
        path_opt_msg = xyz_axis_angle_to_path_msg(matrix_to_xyz_axis_angle(poses[i]),
                                                  frame_id=cfg.world_frame, stamp=stamp)
        dc_pub.publish(pc_opt_msg)
        poses_pub.publish(path_opt_msg)


def eval_loss(cfg: Config):
    """Evaluate model using particular loss on test sequences.

    :param cfg:
    """
    print(cfg.to_yaml())

    if cfg.enable_ros:
        initialize_ros()

    assert cfg.dataset == 'asl_laser'
    if cfg.dataset == 'asl_laser':
        from data.asl_laser import Dataset

    # TODO: Process individual sequences separately.
    val_clouds = []
    val_poses = []
    val_neighbors = [None] * len(cfg.test_names)
    val_masks = [None] * len(cfg.test_names)
    for name in cfg.test_names:
        clouds = []
        poses = []
        # for cloud, pose in cfg.Dataset(name)[::cfg.data_step]:
        for cloud, pose in Dataset(name)[::cfg.data_step]:
            cloud = filtered_cloud(cloud, cfg)
            cloud = local_feature_cloud(cloud, cfg)
            clouds.append(cloud)
            poses.append(pose)
        val_clouds.append(clouds)
        poses = np.stack(poses).astype(dtype=cfg.numpy_float_type())
        poses = torch.as_tensor(poses, device=cfg.device)
        val_poses.append(poses)

    model = load_model(cfg=cfg, eval_mode=True)

    # TODO: Load pose deltas
    # if cfg.pose_correction == PoseCorrection.none:
    #     val_poses_upd = val_poses
    print('Pose deltas not used.')
    val_poses_upd = val_poses

    clouds = global_clouds(val_clouds, model, val_poses_upd)
    for i in range(len(clouds)):
        cloud = clouds[i]
        if val_neighbors[i] is None:
            cloud.update_all(k=cfg.nn_k, r=cfg.nn_r)
            val_neighbors[i] = cloud.neighbors, cloud.weights
            val_masks[i] = filter_eigenvalues(cloud, cfg.eig_bounds, only_mask=True, log=cfg.log_filters)
            print('Validating on %.3f = %i / %i points.'
                  % (val_masks[i].float().mean().item(),
                     val_masks[i].sum().item(), val_masks[i].numel()))
        else:
            cloud.neighbors, cloud.weights = val_neighbors[i]
            cloud.update_all(k=cfg.nn_k, r=cfg.nn_r, keep_neighbors=True)
        clouds[i] = cloud

    val_loss, _ = min_eigval_loss(clouds, mask=val_masks)
    print('Loss: %.9f' % val_loss.item())


def main():
    cfg = Config()
    cfg.test_names = ['stairs']
    cfg.model_class = 'ScaledPolynomial'
    cfg.model_state_dict = '/home/petrito1/workspace/depth_correction/gen/2022-02-21_16-31-34/088_8.85347e-05_state_dict.pth'
    cfg.pose_correction = PoseCorrection.sequence
    eval_loss(cfg)


if __name__ == '__main__':
    main()

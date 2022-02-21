#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from .config import Config
from .depth_cloud import DepthCloud
from .filters import filter_depth, filter_eigenvalue, filter_eigenvalues, filter_grid
from .loss import min_eigval_loss
from .model import *
from .utils import initialize_ros, timer, timing
import importlib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.transforms import (axis_angle_to_matrix,
                                  matrix_to_quaternion,
                                  quaternion_to_axis_angle,
                                  axis_angle_to_quaternion)
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from ros_numpy import msgify
import os
pkg_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
print(pkg_dir)


def filtered_cloud(cloud, cfg: Config):
    cloud = filter_depth(cloud, min=cfg.min_depth, max=cfg.max_depth, log=cfg.log_filters)
    cloud = filter_grid(cloud, grid_res=cfg.grid_res, keep='random', log=cfg.log_filters)
    return cloud


# @timing
def local_feature_cloud(cloud, cfg: Config):
    # Convert to depth cloud and transform.
    cloud = DepthCloud.from_structured_array(cloud, dtype=cfg.dtype)
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


# def skew_matrix(x: torch.Tensor, dim=-1):
#     assert isinstance(x, torch.Tensor)
#     assert x.shape[-1] == 3
#
#     # [0, x[..., 2], x[..., 1]]
#     # torch.index_select(x, )
#     # x.index_select(dim=-1)
#     # torch.linalg.skw
#     shape = list(x.shape)
#     shape[dim] = 9
#     m = torch.zeros(shape, dtype=x.dtype, device=x.device)
#     # y.index_select(dim=dim, [1, 2, 3, 5, 6, 7]) = x.index_select(dim=dim, )
#     i = x.index_select(dim=dim, 0)
#     j = x.index_select(dim=dim, 1)
#     k = x.index_select(dim=dim, 2)
#     y.index_select(dim=dim, 1) = -x.index_select(dim=dim, 2)
#     y = y.reshape(shape[:dim] + [3, 3] + shape[dim + 1:])
#     return y
#
#
#
# def axis_angle_xyz_to_matrix(x: torch.Tensor):
#     assert isinstance(x, torch.Tensor)
#     assert x.shape[-1] == 3
#     angle = torch.linalg.norm(x, dim=-1, keepdim=True)
#     return y


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


def train(cfg: Config):

    if cfg.enable_ros:
        initialize_ros()

    # Cloud needs to retain neighbors, weights, and mask from previous
    # iterations.
    # Depth correction is applied based on local cloud statistics.
    # Loss is computed based on global cloud statistics.

    train_clouds = []
    train_poses = []
    # Pose corrections 3 translation, 3 elements axis-angle,.
    train_pose_deltas = []
    train_neighbors = [None] * len(cfg.train_names)
    train_masks = [None] * len(cfg.train_names)
    for name in cfg.train_names:
        clouds = []
        poses = []
        for cloud, pose in cfg.Dataset(name)[::cfg.data_step]:
            cloud = filtered_cloud(cloud, cfg)
            cloud = local_feature_cloud(cloud, cfg)
            # If poses are not optimized, depth can be corrected on global
            # feature clouds.
            # If poses are to be optimized, depth can be corrected on local
            # clouds and these can then be transformed to global cloud.
            clouds.append(cloud)
            poses.append(pose)
        train_clouds.append(clouds)
        poses = np.stack(poses).astype(dtype=cfg.dtype)
        poses = torch.as_tensor(poses, device=cfg.device)
        train_poses.append(poses)
        # pose_deltas = torch.zeros((poses.shape[0], 6), dtype=poses.dtype)
        pose_deltas = torch.zeros((1, 6), dtype=poses.dtype)
        pose_deltas.requires_grad = True
        train_pose_deltas.append(pose_deltas)

    val_clouds = []
    val_poses = []
    val_neighbors = [None] * len(cfg.val_names)
    val_masks = [None] * len(cfg.val_names)
    for name in cfg.val_names:
        clouds = []
        poses = []
        for cloud, pose in cfg.Dataset(name)[::cfg.data_step]:
            cloud = filtered_cloud(cloud, cfg)
            cloud = local_feature_cloud(cloud, cfg)
            clouds.append(cloud)
            poses.append(pose)
        val_clouds.append(clouds)
        poses = np.stack(poses).astype(dtype=cfg.dtype)
        poses = torch.as_tensor(poses, device=cfg.device)
        val_poses.append(poses)

    # Create model
    model = eval(cfg.model_class)(device=cfg.device)
    print(model)

    # Initialize optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, nesterov=True)
    # optimizer = torch.optim.SGD(train_pose_deltas, lr=LR, momentum=0.9, nesterov=True)
    optimizer = torch.optim.SGD([{'params': model.parameters(), 'lr': cfg.lr},
                                 {'params': train_pose_deltas, 'lr': cfg.lr}], momentum=0.9, nesterov=True)

    writer = SummaryWriter('%s/config/tb_runs/model_%s_lr_%f_%s_%f'
                           % (pkg_dir, cfg.model_class, cfg.lr, cfg.Dataset, timer()))

    min_loss = np.inf
    for it in range(cfg.n_opt_iters):
        if rospy.is_shutdown():
            break

        optimizer.zero_grad()

        # Training

        # Allow optimizing pose deltas.
        if train_pose_deltas is None:
            train_poses_upd = train_poses
        else:
            # Convert pose deltas to matrices
            train_poses_upd = []
            for i in range(len(train_poses)):
                pose_deltas_mat = xyz_axis_angle_to_matrix(train_pose_deltas[i])
                train_poses_upd.append(torch.matmul(train_poses[i], pose_deltas_mat))

        clouds = global_clouds(train_clouds, model, train_poses_upd)

        for i in range(len(clouds)):
            cloud = clouds[i]
            if train_neighbors[i] is None:
                cloud.update_all(k=cfg.nn_k, r=cfg.nn_r)
                train_neighbors[i] = cloud.neighbors, cloud.weights
                # mask = filter_eigenvalue(cloud, 0, max=max_eig_0, only_mask=True, log=log_filters)
                # mask = mask & filter_eigenvalue(cloud, 1, min=min_eig_1, only_mask=True, log=log_filters)
                # mask = None
                # for eig, min, max in cfg.eig_bounds:
                #     eig_mask = filter_eigenvalue(cloud, eig, min=min, max=max, only_mask=True, log=cfg.log_filters)
                #     mask = eig_mask if mask is None else mask & eig_mask
                # train_masks[i] = mask
                train_masks[i] = filter_eigenvalues(cloud, eig_bounds=cfg.eig_bounds, only_mask=True,
                                                    log=cfg.log_filters)
                print('Training on %.3f = %i / %i points.'
                      % (train_masks[i].float().mean().item(),
                         train_masks[i].sum().item(), train_masks[i].numel()))
            else:
                cloud.neighbors, cloud.weights = train_neighbors[i]
                cloud.update_all(k=cfg.nn_k, r=cfg.nn_r, keep_neighbors=True)
            clouds[i] = cloud

        train_loss, _ = min_eigval_loss(clouds, mask=train_masks)

        # Validation

        if train_pose_deltas is None:
            val_poses_upd = val_poses
        else:
            # Convert pose deltas to matrices
            val_poses_upd = []
            for i in range(len(train_poses)):
                pose_deltas_mat = xyz_axis_angle_to_matrix(train_pose_deltas[i])
                val_poses_upd.append(torch.matmul(val_poses[i], pose_deltas_mat))

        clouds = global_clouds(val_clouds, model, val_poses_upd)
        for i in range(len(clouds)):
            cloud = clouds[i]
            if val_neighbors[i] is None:
                cloud.update_all(k=cfg.nn_k, r=cfg.nn_r)
                val_neighbors[i] = cloud.neighbors, cloud.weights
                # mask = filter_eigenvalue(cloud, 0, max=max_eig_0, only_mask=True, log=log_filters)
                # mask = mask & filter_eigenvalue(cloud, 1, min=min_eig_1, only_mask=True, log=log_filters)
                # val_masks[i] = mask
                val_masks[i] = filter_eigenvalues(cloud, cfg.eig_bounds, only_mask=True, log=cfg.log_filters)
                print('Validating on %.3f = %i / %i points.'
                      % (val_masks[i].float().mean().item(),
                         val_masks[i].sum().item(), val_masks[i].numel()))
            else:
                cloud.neighbors, cloud.weights = val_neighbors[i]
                cloud.update_all(k=cfg.nn_k, r=cfg.nn_r, keep_neighbors=True)
            clouds[i] = cloud

        val_loss, _ = min_eigval_loss(clouds, mask=val_masks)

        # if SHOW_RESULTS and it % plot_period == 0:
        #     for dc in clouds:
        #         dc.visualize(colors='inc_angles')
        #         dc.visualize(colors='min_eigval')

        if val_loss.item() < min_loss:
            saved = True
            min_loss = val_loss.item()
            torch.save(model.state_dict(),
                       '%s/config/weights/%s_train_%s_val_%s_r%.2f_eig_%.4f_%.4f_min_eigval_loss_%.9f.pth'
                       % (pkg_dir, cfg.model_class, ','.join(cfg.train_names), ','.join(cfg.val_names),
                          cfg.nn_r, cfg.eig_bounds[0][2], cfg.eig_bounds[1][1], val_loss.item()))
        else:
            saved = False

        print('It. %i: training loss: %.9f, validation: %.9f. Model %s %s.'
              % (it, train_loss.item(), val_loss.item(), model, 'saved' if saved else 'not saved'))

        # publish (validation) poses and clouds
        # publish_data(clouds, val_poses_upd, frame_id='world')

        writer.add_scalar("min_eigval_loss/train", train_loss, it)
        writer.add_scalar("min_eigval_loss/val", val_loss, it)
        for i in range(len(cfg.val_names)):
            pose_deltas = train_pose_deltas[i].squeeze(0)
            writer.add_scalar("pose_correction_%s/dx" % cfg.val_names[i], pose_deltas[0], it)
            writer.add_scalar("pose_correction_%s/dy" % cfg.val_names[i], pose_deltas[1], it)
            writer.add_scalar("pose_correction_%s/dz" % cfg.val_names[i], pose_deltas[2], it)
            writer.add_scalar("pose_correction_%s/dax" % cfg.val_names[i], pose_deltas[3], it)
            writer.add_scalar("pose_correction_%s/day" % cfg.val_names[i], pose_deltas[4], it)
            writer.add_scalar("pose_correction_%s/daz" % cfg.val_names[i], pose_deltas[5], it)

        # Optimization step
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    writer.flush()
    writer.close()


def main():
    cfg = Config()
    # Debug
    cfg.data_step = 10
    cfg.max_depth = 10.0
    cfg.grid_res = 0.1
    cfg.nn_r = .2
    train(cfg)


if __name__ == '__main__':
    main()

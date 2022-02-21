#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from .config import Config, PoseCorrection
from .depth_cloud import DepthCloud
from .filters import filter_depth, filter_eigenvalue, filter_eigenvalues, filter_grid
from .loss import min_eigval_loss
from .model import *
from .utils import initialize_ros, timer, timing
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
import importlib
from nav_msgs.msg import Path
import numpy as np
import os
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


pkg_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
print(pkg_dir)


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


def train(cfg: Config):
    """Train and return the depth correction model model.
    Validation datasets are used to select the best model.

    :param cfg:
    :return: Trained and validated model.
    """
    print(cfg.to_yaml())

    if cfg.enable_ros:
        initialize_ros()

    # Cloud needs to retain neighbors, weights, and mask from previous
    # iterations.
    # Depth correction is applied based on local cloud statistics.
    # Loss is computed based on global cloud statistics.

    if cfg.dataset == 'asl_laser':
        from data.asl_laser import Dataset
    # Dataset = eval(cfg.dataset)

    train_clouds = []
    train_poses = []
    # Pose corrections 3 translation, 3 elements axis-angle,.
    train_pose_deltas = []
    train_neighbors = [None] * len(cfg.train_names)
    train_masks = [None] * len(cfg.train_names)
    for name in cfg.train_names:
        clouds = []
        poses = []
        # for cloud, pose in cfg.Dataset(name)[::cfg.data_step]:
        for cloud, pose in Dataset(name)[::cfg.data_step]:
            cloud = filtered_cloud(cloud, cfg)
            cloud = local_feature_cloud(cloud, cfg)
            # If poses are not optimized, depth can be corrected on global
            # feature clouds.
            # If poses are to be optimized, depth can be corrected on local
            # clouds and these can then be transformed to global cloud.
            clouds.append(cloud)
            poses.append(pose)
        train_clouds.append(clouds)
        # poses = np.stack(poses).astype(dtype=cfg.dtype)
        poses = np.stack(poses).astype(dtype=cfg.numpy_float_type())
        poses = torch.as_tensor(poses, device=cfg.device)
        train_poses.append(poses)
        pose_deltas = None
        if cfg.pose_correction == PoseCorrection.none:
            # pose_deltas = None
            continue
        elif cfg.pose_correction == PoseCorrection.common:
            # Use same tensor if possible.
            if train_pose_deltas:
                pose_deltas = train_pose_deltas[0]
            else:
                pose_deltas = torch.zeros((1, 6), dtype=poses.dtype, requires_grad=True)
        elif cfg.pose_correction == PoseCorrection.sequence:
            pose_deltas = torch.zeros((1, 6), dtype=poses.dtype, requires_grad=True)
        elif cfg.pose_correction == PoseCorrection.pose:
            pose_deltas = torch.zeros((poses.shape[0], 6), dtype=poses.dtype)
        # pose_deltas = torch.zeros((1, 6), dtype=poses.dtype, requires_grad=True)
        # pose_deltas.requires_grad = True
        train_pose_deltas.append(pose_deltas)

    val_clouds = []
    val_poses = []
    val_neighbors = [None] * len(cfg.val_names)
    val_masks = [None] * len(cfg.val_names)
    for name in cfg.val_names:
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

    model = load_model(cfg=cfg, eval_mode=False)
    print(model)

    # Initialize optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, nesterov=True)
    # optimizer = torch.optim.SGD(train_pose_deltas, lr=LR, momentum=0.9, nesterov=True)
    params = [{'params': model.parameters(), 'lr': cfg.lr}]
    if cfg.pose_correction != PoseCorrection.none:
        params.append({'params': train_pose_deltas, 'lr': cfg.lr})
    optimizer = torch.optim.SGD(params, momentum=0.9, nesterov=True)

    # writer = SummaryWriter('%s/config/tb_runs/model_%s_lr_%f_%s_%f'
    #                        % (pkg_dir, cfg.model_class, cfg.lr, cfg.Dataset, timer()))
    writer = SummaryWriter(cfg.log_dir)

    min_loss = np.inf
    # best_model = None
    # best_pose_deltas = None
    best_cfg = None
    for it in range(cfg.n_opt_iters):
        if rospy.is_shutdown():
            break

        # Training

        # Allow optimizing pose deltas.
        # if train_pose_deltas is None:
        if cfg.pose_correction == PoseCorrection.none:
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
        # if train_pose_deltas is None:
        if cfg.pose_correction == PoseCorrection.none:
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
                val_masks[i] = filter_eigenvalues(cloud, cfg.eig_bounds, only_mask=True, log=cfg.log_filters)
                print('Validating on %.3f = %i / %i points.'
                      % (val_masks[i].float().mean().item(),
                         val_masks[i].sum().item(), val_masks[i].numel()))
            else:
                cloud.neighbors, cloud.weights = val_neighbors[i]
                cloud.update_all(k=cfg.nn_k, r=cfg.nn_r, keep_neighbors=True)
            clouds[i] = cloud

        val_loss, _ = min_eigval_loss(clouds, mask=val_masks)

        # if cfg.show_results and it % cfg.plot_period == 0:
        #     for dc in clouds:
        #         dc.visualize(colors='inc_angles')
        #         dc.visualize(colors='min_eigval')

        if val_loss.item() < min_loss:
            saved = True
            min_loss = val_loss.item()
            # torch.save(model.state_dict(),
            #            '%s/%s_train_%s_val_%s_r%.2f_eig_%.4f_%.4f_min_eigval_loss_it_%03i_loss_%.6g.pth'
            #            % (cfg.log_dir, cfg.model_class, ','.join(cfg.train_names), ','.join(cfg.val_names),
            #               cfg.nn_r, cfg.eig_bounds[0][2], cfg.eig_bounds[1][1], it, val_loss.item()))
            state_dict_path = '%s/%03i_%.6g_state_dict.pth' % (cfg.log_dir, it, min_loss)
            torch.save(model.state_dict(), state_dict_path)
            # best_model = model.detach().clone()
            pose_deltas = [p.detach().clone() for p in train_pose_deltas if p is not None]
            pose_deltas_path = '%s/%03i_%.6g_pose_deltas.pth' % (cfg.log_dir, it, min_loss)
            torch.save(pose_deltas, pose_deltas_path)

            best_cfg = cfg.copy()
            best_cfg.model_state_dict = state_dict_path
            best_cfg.train_pose_deltas = pose_deltas_path

        else:
            saved = False

        print('It. %03i: train loss: %.9f, val.: %.9f. Model %s %s.'
              % (it, train_loss.item(), val_loss.item(), model, 'saved' if saved else 'not saved'))

        # publish (validation) poses and clouds
        # publish_data(clouds, val_poses_upd, frame_id='world')

        writer.add_scalar("min_eigval_loss/train", train_loss, it)
        writer.add_scalar("min_eigval_loss/val", val_loss, it)
        if train_pose_deltas:
            # TODO: Add summary histogram for all sequences.
            for i in range(len(cfg.train_names)):
                writer.add_histogram("pose_correction/train/%s/dx" % cfg.train_names[i], train_pose_deltas[i][:, 0], it)
                writer.add_histogram("pose_correction/train/%s/dy" % cfg.train_names[i], train_pose_deltas[i][:, 1], it)
                writer.add_histogram("pose_correction/train/%s/dz" % cfg.train_names[i], train_pose_deltas[i][:, 2], it)
                writer.add_histogram("pose_correction/train/%s/dax" % cfg.train_names[i], train_pose_deltas[i][:, 3], it)
                writer.add_histogram("pose_correction/train/%s/day" % cfg.train_names[i], train_pose_deltas[i][:, 4], it)
                writer.add_histogram("pose_correction/train/%s/daz" % cfg.train_names[i], train_pose_deltas[i][:, 5], it)

        # Optimization step
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    writer.flush()
    writer.close()

    # return best_model, best_pose_deltas
    return best_cfg


def main():
    cfg = Config()
    # Debug
    cfg.data_step = 10
    cfg.max_depth = 10.0
    cfg.grid_res = 0.1
    cfg.nn_r = .2
    cfg.pose_correction = PoseCorrection.sequence
    train(cfg)


if __name__ == '__main__':
    main()

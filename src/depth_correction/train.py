from __future__ import absolute_import, division, print_function
from .config import Config, PoseCorrection
from .filters import filter_eigenvalues
from .loss import min_eigval_loss, trace_loss
from .model import *
from .preproc import *
from .ros import *
from .transform import *
import numpy as np
import os
import rospy
import torch
from torch.utils.tensorboard import SummaryWriter


def train(cfg: Config):
    """Train depth correction model, validate it, and return best config.

    :param cfg:
    :return: Config of the best model.
    """
    cfg_path = os.path.join(cfg.log_dir, 'train.yaml')
    cfg.to_yaml(cfg_path)

    assert cfg.dataset == 'asl_laser'
    if cfg.dataset == 'asl_laser':
        from data.asl_laser import Dataset
    assert cfg.loss in ('min_eigval_loss', 'trace_loss')
    # if cfg.loss == 'min_eigval_loss':
    #     loss = min_eigval_loss
    loss_fun = eval(cfg.loss)
    print(cfg.loss, loss_fun.__name__)

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

    # Collect optimizable parameters.
    params = [{'params': model.parameters(), 'lr': cfg.lr}]
    if cfg.pose_correction != PoseCorrection.none:
        params.append({'params': train_pose_deltas, 'lr': cfg.lr})
    # Initialize optimizer.
    # optimizer = torch.optim.Adam(params)
    optimizer = torch.optim.SGD(params, momentum=0.9, nesterov=True)

    writer = SummaryWriter(cfg.log_dir)

    min_loss = np.inf
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

        train_loss, _ = loss_fun(clouds, mask=train_masks)

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

        val_loss, _ = loss_fun(clouds, mask=val_masks)

        # if cfg.show_results and it % cfg.plot_period == 0:
        #     for dc in clouds:
        #         dc.visualize(colors='inc_angles')
        #         dc.visualize(colors='min_eigval')

        if val_loss.item() < min_loss:
            saved = True
            min_loss = val_loss.item()
            state_dict_path = '%s/%03i_%.6g_state_dict.pth' % (cfg.log_dir, it, min_loss)
            torch.save(model.state_dict(), state_dict_path)
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
        if cfg.enable_ros:
            publish_data(clouds, val_poses_upd, cfg.val_names, cfg=cfg)

        writer.add_scalar("%s/train" % cfg.loss, train_loss, it)
        writer.add_scalar("%s/val" % cfg.loss, val_loss, it)
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

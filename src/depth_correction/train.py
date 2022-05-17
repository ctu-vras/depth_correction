from __future__ import absolute_import, division, print_function
from .config import Config, PoseCorrection
from .dataset import create_dataset
from .loss import create_loss
from .model import load_model
from .preproc import *
from .ros import *
from .transform import *
from argparse import ArgumentParser
import numpy as np
import os
import rospy
import torch
from torch.optim import Adam, SGD  # Needed for eval
from torch.utils.tensorboard import SummaryWriter


class TrainCallbacks(object):

    def __init__(self, cfg: Config=None):
        self.cfg = cfg

    def iteration_started(self, iter):
        pass

    def train_inputs(self, iter, clouds, poses):
        pass

    def val_inputs(self, iter, clouds, poses):
        pass

    def train_loss(self, iter, model, clouds, pose_deltas, poses, masks, loss):
        pass

    def val_loss(self, iter, model, clouds, pose_deltas, poses, masks, loss):
        pass


def initialize_pose_corrections(datasets, cfg: Config):
    pose_deltas = []
    for ds in enumerate(datasets):
        if cfg.pose_correction == PoseCorrection.common:
            # Use a common correction for all sequences and poses.
            if pose_deltas:
                pose_delta = pose_deltas[0]
            else:
                pose_delta = torch.zeros((1, 6), dtype=cfg.torch_float_type(), requires_grad=True)
        elif cfg.pose_correction == PoseCorrection.sequence:
            # Single correction per sequence (sensor rig calibration).
            pose_delta = torch.zeros((1, 6), dtype=cfg.torch_float_type(), requires_grad=True)
        elif cfg.pose_correction == PoseCorrection.pose:
            # Correct every pose (e.g. from odometry or SLAM).
            pose_delta = torch.zeros((len(ds), 6), dtype=cfg.torch_float_type(), requires_grad=True)
        else:
            pose_delta = None
        if pose_delta is not None:
            pose_deltas.append(pose_delta)
        # pose_deltas.append(pose_delta)

    return pose_deltas


def create_corrected_poses(train_poses, train_pose_deltas, cfg: Config):
    if cfg.pose_correction == PoseCorrection.none:
        train_poses_upd = train_poses
    else:
        assert len(train_poses) == len(train_pose_deltas)
        # For common pose correction, there is a same correction for all sequences.
        if cfg.pose_correction == PoseCorrection.common:
            assert all(d is train_pose_deltas[0] for d in train_pose_deltas[1:])
        train_poses_upd = []
        for i in range(len(train_poses)):
            pose_deltas_mat = xyz_axis_angle_to_matrix(train_pose_deltas[i])
            train_poses_upd.append(torch.matmul(train_poses[i], pose_deltas_mat))

    return train_poses_upd


def train(cfg: Config, callbacks=None, train_datasets=None, val_datasets=None):
    """Train depth correction model, validate it, and return best config.

    :param cfg: Training config.
    :param callbacks: Dictionary of callbacks to call during training.
    :param train_datasets: Use these datasets instead of those created from config.
    :return: Config of the best model.
    """
    if not callbacks:
        callbacks = TrainCallbacks(cfg)

    cfg_path = os.path.join(cfg.log_dir, 'train.yaml')
    if os.path.exists(cfg_path):
        print('Config %s already exists.' % cfg_path)
    else:
        cfg.to_yaml(cfg_path)

    if train_datasets:
        print('Using provided training datasets.')
    else:
        print('Creating datasets from config: %s.' % ', '.join(cfg.train_names))
        train_datasets = []
        for i, name in enumerate(cfg.train_names):
            # Allow overriding poses paths, assume valid if non-empty.
            poses_path = cfg.train_poses_path[i] if cfg.train_poses_path else None
            ds = create_dataset(name, cfg, poses_path=poses_path)
            train_datasets.append(ds)

    if val_datasets:
        print('Using provided training datasets.')
    else:
        print('Creating datasets from config: %s.' % ', '.join(cfg.val_names))
        val_datasets = []
        for i, name in enumerate(cfg.val_names):
            poses_path = cfg.val_poses_path[i] if cfg.val_poses_path else None
            ds = create_dataset(name, cfg, poses_path=poses_path)
            val_datasets.append(ds)

    loss_fun = create_loss(cfg)

    # Cloud needs to retain neighbors, weights, and mask from previous
    # iterations.
    # Depth correction is applied based on local cloud statistics.
    # Loss is computed based on global cloud statistics.
    train_clouds = []
    train_poses = []
    # Pose corrections 3 translation, 3 elements axis-angle,.
    train_neighbors = [None] * len(cfg.train_names)
    train_masks = [None] * len(cfg.train_names)
    for i, ds in enumerate(train_datasets):
        # Allow overriding poses paths, assume valid if non-empty.
        clouds = []
        poses = []
        for cloud, pose in ds:
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
    train_pose_deltas = initialize_pose_corrections(train_datasets, cfg)

    val_clouds = []
    val_poses = []
    # Pose corrections 3 translation, 3 elements axis-angle,.
    val_neighbors = [None] * len(cfg.val_names)
    val_masks = [None] * len(cfg.val_names)
    for i, ds in enumerate(val_datasets):
        clouds = []
        poses = []
        for cloud, pose in ds:
            cloud = local_feature_cloud(cloud, cfg)
            clouds.append(cloud)
            poses.append(pose)
        val_clouds.append(clouds)
        poses = np.stack(poses).astype(dtype=cfg.numpy_float_type())
        poses = torch.as_tensor(poses, device=cfg.device)
        val_poses.append(poses)

    if cfg.pose_correction == PoseCorrection.common:
        # Reuse pose correction from training.
        val_pose_deltas = len(val_datasets) * [train_pose_deltas[0]]
    else:
        val_pose_deltas = initialize_pose_corrections(val_datasets, cfg)

    model = load_model(cfg=cfg, eval_mode=False)
    print(model)

    # Optimizable parameters and optimizer for train sequences.
    params = [{'params': model.parameters(), 'lr': cfg.lr}]
    if cfg.pose_correction != PoseCorrection.none:
        params.append({'params': train_pose_deltas, 'lr': cfg.lr})
    # Initialize optimizer.
    args = cfg.optimizer_args if cfg.optimizer_args else []
    kwargs = cfg.optimizer_kwargs if cfg.optimizer_kwargs else {}
    optimizer = eval(cfg.optimizer)(params, *args, **kwargs)
    print('Optimizer: %s' % optimizer)

    # Optimizable parameters and optimizer for validation sequences,
    # only for sequence- or pose-wise corrections.
    if (cfg.pose_correction == PoseCorrection.sequence
            or cfg.pose_correction == PoseCorrection.pose):
        val_params = [{'params': val_pose_deltas, 'lr': cfg.lr}]
        args = cfg.optimizer_args if cfg.optimizer_args else []
        kwargs = cfg.optimizer_kwargs if cfg.optimizer_kwargs else {}
        val_optimizer = eval(cfg.optimizer)(val_params, *args, **kwargs)
        print('Validation optimizer: %s' % val_optimizer)
    else:
        val_optimizer = None

    writer = SummaryWriter(cfg.log_dir)

    min_loss = np.inf
    best_cfg = None
    for it in range(cfg.n_opt_iters):
        if rospy.is_shutdown():
            break

        # Training
        # Allow optimizing pose deltas.
        train_poses_upd = create_corrected_poses(train_poses, train_pose_deltas, cfg)

        clouds = [global_cloud(c, model, p) for c, p in zip(train_clouds, train_poses_upd)]
        offsets = [offset_cloud(c, model) for c in train_clouds] if cfg.loss_offset else None

        for i in range(len(clouds)):
            cloud = clouds[i]
            if train_neighbors[i] is None:
                cloud.update_all(k=cfg.nn_k, r=cfg.nn_r)
                train_neighbors[i] = cloud.neighbors, cloud.weights
                train_masks[i] = global_cloud_mask(cloud, cloud.mask, cfg)
                print('Training on %.3f = %i / %i points.'
                      % (train_masks[i].float().mean().item(),
                         train_masks[i].sum().item(), train_masks[i].numel()))
            else:
                cloud.neighbors, cloud.weights = train_neighbors[i]
                cloud.update_all(k=cfg.nn_k, r=cfg.nn_r, keep_neighbors=True)
            clouds[i] = cloud

        train_loss, _ = loss_fun(clouds, mask=train_masks, offset=offsets)
        callbacks.train_loss(it, model, clouds, train_pose_deltas, train_poses_upd, train_masks, train_loss)

        # Validation
        val_poses_upd = create_corrected_poses(val_poses, val_pose_deltas, cfg)

        clouds = [global_cloud(c, model, p) for c, p in zip(val_clouds, val_poses_upd)]
        offsets = [offset_cloud(c, model) for c in val_clouds] if cfg.loss_offset else None

        for i in range(len(clouds)):
            cloud = clouds[i]
            if val_neighbors[i] is None:
                cloud.update_all(k=cfg.nn_k, r=cfg.nn_r)
                val_neighbors[i] = cloud.neighbors, cloud.weights
                val_masks[i] = global_cloud_mask(cloud, cloud.mask, cfg)
                print('Validating on %.3f = %i / %i points.'
                      % (val_masks[i].float().mean().item(),
                         val_masks[i].sum().item(), val_masks[i].numel()))
            else:
                cloud.neighbors, cloud.weights = val_neighbors[i]
                cloud.update_all(k=cfg.nn_k, r=cfg.nn_r, keep_neighbors=True)
            clouds[i] = cloud

        val_loss, _ = loss_fun(clouds, mask=val_masks, offset=offsets)
        callbacks.val_loss(it, model, clouds, val_pose_deltas, val_poses_upd, val_masks, val_loss)

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
            poses_upd = [p.detach().clone() for p in train_poses_upd if p is not None]
            poses_upd_path = '%s/%03i_%.6g_poses_upd.pth' % (cfg.log_dir, it, min_loss)
            torch.save(poses_upd, poses_upd_path)

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

        # Optimize validation pose updates.
        if val_optimizer is not None:
            val_optimizer.zero_grad()
            val_loss.backward()
            val_optimizer.step()

    writer.flush()
    writer.close()

    best_cfg_path = os.path.join(cfg.log_dir, 'best.yaml')
    best_cfg.to_yaml(best_cfg_path)

    return best_cfg


def run_from_cmdline():
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    args = parser.parse_args()
    print('Arguments:')
    print(args)
    cfg = Config()
    cfg.from_yaml(args.config)
    print('Config:')
    print(cfg.to_yaml())
    print('Training...')
    train(cfg)
    print('Training finished.')


def main():
    run_from_cmdline()


if __name__ == '__main__':
    main()

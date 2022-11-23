from __future__ import absolute_import, division, print_function
from .config import Config, NeighborhoodType, PoseCorrection
from .dataset import create_dataset
from .depth_cloud import DepthCloud
from .eval import eval_loss_clouds, initialize_pose_corrections
from .loss import create_loss
from .model import load_model
from .preproc import establish_neighborhoods, local_feature_cloud
from .ros import publish_data
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
        print('Using provided training datasets: %s.' % ', '.join([str(ds) for ds in train_datasets]))
    else:
        print('Creating training datasets from config: %s.' % ', '.join(cfg.train_names))
        train_datasets = []
        for i, name in enumerate(cfg.train_names):
            # Allow overriding poses paths, assume valid if non-empty.
            poses_path = cfg.train_poses_path[i] if cfg.train_poses_path else None
            ds = create_dataset(name, cfg, poses_path=poses_path)
            train_datasets.append(ds)

    if val_datasets:
        print('Using provided validation datasets: %s.' % ', '.join([str(ds) for ds in val_datasets]))
    else:
        print('Creating validation datasets from config: %s.' % ', '.join(cfg.val_names))
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
    train_masks = [None] * len(train_datasets)
    for i, ds in enumerate(train_datasets):
        clouds = []
        poses = []
        for cloud, pose in ds:
            if cfg.nn_type == NeighborhoodType.ball:
                cloud = local_feature_cloud(cloud, cfg)
            else:
                cloud = DepthCloud.from_structured_array(cloud, dtype=cfg.numpy_float_type(), device=cfg.device)
            clouds.append(cloud)
            poses.append(pose)
        train_clouds.append(clouds)
        poses = np.stack(poses).astype(dtype=cfg.numpy_float_type())
        poses = torch.as_tensor(poses, device=cfg.device)
        train_poses.append(poses)
    train_pose_deltas = initialize_pose_corrections(train_datasets, cfg)

    val_clouds = []
    val_poses = []
    # Pose corrections 3 translation, 3 elements axis-angle,.
    val_masks = [None] * len(val_datasets)
    for i, ds in enumerate(val_datasets):
        clouds = []
        poses = []
        for cloud, pose in ds:
            if cfg.nn_type == NeighborhoodType.ball:
                cloud = local_feature_cloud(cloud, cfg)
            else:
                cloud = DepthCloud.from_structured_array(cloud, dtype=cfg.numpy_float_type(), device=cfg.device)
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
    params = []
    if cfg.optimize_model and len(list(model.parameters())) > 0:
        params.append({'params': model.parameters(), 'lr': cfg.lr})
    if cfg.pose_correction != PoseCorrection.none:
        # params.append({'params': train_pose_deltas, 'lr': cfg.lr, 'weight_decay': 0.0})
        params.append({'params': train_pose_deltas, 'lr': cfg.lr})
    # Initialize optimizer.
    args = cfg.optimizer_args[:] if cfg.optimizer_args else []
    kwargs = cfg.optimizer_kwargs.copy() if cfg.optimizer_kwargs else {}
    optimizer = eval(cfg.optimizer)(params, *args, **kwargs)
    print('Optimizer: %s' % optimizer)

    # Optimizable parameters and optimizer for validation sequences,
    # only for sequence- or pose-wise corrections.
    if (cfg.pose_correction == PoseCorrection.sequence
            or cfg.pose_correction == PoseCorrection.pose):
        val_params = [{'params': val_pose_deltas, 'lr': cfg.lr}]
        args = cfg.optimizer_args[:] if cfg.optimizer_args else []
        kwargs = cfg.optimizer_kwargs.copy() if cfg.optimizer_kwargs else {}
        val_optimizer = eval(cfg.optimizer)(val_params, *args, **kwargs)
        print('Validation optimizer: %s' % val_optimizer)
    else:
        val_optimizer = None

    writer = SummaryWriter(cfg.log_dir)

    # Create training and validation neighborhoods.
    train_ns = [establish_neighborhoods(clouds=clouds, poses=poses, cfg=cfg)
                for clouds, poses in zip(train_clouds, train_poses)]
    val_ns = [establish_neighborhoods(clouds=clouds, poses=poses, cfg=cfg)
              for clouds, poses in zip(val_clouds, val_poses)]

    min_loss = np.inf
    best_cfg = None
    for it in range(cfg.n_opt_iters):
        if rospy.is_shutdown():
            break

        # Training
        train_loss, _, train_poses_upd, train_feat_clouds \
                = eval_loss_clouds(train_clouds, train_poses, train_pose_deltas, train_masks, train_ns,
                                   model, loss_fun, cfg)
        callbacks.train_loss(it, model, train_feat_clouds, train_pose_deltas, train_poses_upd, train_masks, train_loss)

        # Validation
        val_loss, _, val_poses_upd, val_feat_clouds \
                = eval_loss_clouds(val_clouds, val_poses, val_pose_deltas, val_masks, val_ns,
                               model, loss_fun, cfg)
        callbacks.val_loss(it, model, val_feat_clouds, val_pose_deltas, val_poses_upd, val_masks, val_loss)

        if cfg.enable_ros and cfg.val_names:
            publish_data(clouds, poses_upd, cfg.val_names, cfg=cfg)

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
            best_cfg.to_yaml(os.path.join(cfg.log_dir, 'best.yaml'))

        else:
            saved = False

        print('It. %03i: train loss: %.9f, val.: %.9f. Model %s %s.'
              % (it, train_loss.item(), val_loss.item(), model, 'saved' if saved else 'not saved'))

        # publish (validation) poses and clouds
        if cfg.enable_ros:
            publish_data(clouds, val_poses_upd, cfg.val_names, cfg=cfg)

        writer.add_scalar("%s/train" % cfg.loss, train_loss, it)
        writer.add_scalar("%s/val" % cfg.loss, val_loss, it)

        if hasattr(model, 'w'):
            assert model.w.shape[0] == 1
            for i in range(model.w.numel()):
                writer.add_scalar('model/w_%i' % i, model.w[i], it)
                if model.w.grad is not None:
                    writer.add_scalar('model/w_%i/grad' % i, model.w.grad[i], it)

        if hasattr(model, 'exponent'):
            assert model.exponent.shape[0] == 1
            for i in range(model.exponent.numel()):
                writer.add_scalar('model/exponent_%i' % i, model.exponent[i], it)
                if model.exponent.grad is not None:
                    writer.add_scalar('model/exponent_%i/grad' % i, model.exponent.grad[i], it)

        if train_pose_deltas and train_pose_deltas[0] is not None:
            # TODO: Add summary histogram for all sequences.
            for i in range(len(train_datasets)):
                name = str(train_datasets[i])
                for j, key in enumerate(['tx', 'ty', 'tz', 'rx', 'ry', 'rz']):
                    writer.add_histogram("pose_correction/train/%s/%s"
                                         % (name, key), train_pose_deltas[i][:, j], it)
                    if train_pose_deltas[i].grad is not None:
                        writer.add_histogram("pose_correction/train/%s/%s/grad"
                                             % (name, key), train_pose_deltas[i].grad[:, j], it)

        # Optimization step
        optimizer.zero_grad()
        train_loss.backward()
        # Keep the first pose fixed.
        if cfg.pose_correction == PoseCorrection.pose:
            for i in range(len(train_pose_deltas)):
                train_pose_deltas[i].grad[0].zero_()
        optimizer.step()

        # Optimize validation pose updates.
        if val_optimizer is not None:
            val_optimizer.zero_grad()
            val_loss.backward()
            # Keep the first pose fixed.
            if cfg.pose_correction == PoseCorrection.pose:
                for i in range(len(val_pose_deltas)):
                    val_pose_deltas[i].grad[0].zero_()
            val_optimizer.step()

    writer.flush()
    writer.close()

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

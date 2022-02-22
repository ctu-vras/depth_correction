from __future__ import absolute_import, division, print_function
from .config import Config, PoseCorrection
from .filters import filter_eigenvalues
from .io import append
from .loss import min_eigval_loss
from .model import *
from .preproc import *
from .ros import *
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter


def eval_loss(cfg: Config):
    """Evaluate loss of model on test sequences.

    :param cfg:
    """
    assert cfg.dataset == 'asl_laser'
    if cfg.dataset == 'asl_laser':
        from data.asl_laser import Dataset

    model = load_model(cfg=cfg, eval_mode=True)

    if cfg.pose_correction != PoseCorrection.none:
        print('Pose deltas not used.')

    # TODO: Process individual sequences separately.
    for name in cfg.test_names:
        clouds = []
        poses = []
        for cloud, pose in Dataset(name)[::cfg.data_step]:
            cloud = filtered_cloud(cloud, cfg)
            cloud = local_feature_cloud(cloud, cfg)
            clouds.append(cloud)
            poses.append(pose)
        poses = np.stack(poses).astype(dtype=cfg.numpy_float_type())
        poses = torch.as_tensor(poses, device=cfg.device)

        cloud = global_cloud(clouds, model, poses)
        cloud.update_all(k=cfg.nn_k, r=cfg.nn_r)
        mask = filter_eigenvalues(cloud, eig_bounds=cfg.eig_bounds,
                                  only_mask=True, log=cfg.log_filters)
        print('Testing on %.3f = %i / %i points from %s.'
              % (mask.float().mean().item(), mask.sum().item(), mask.numel(),
                 name))

        if cfg.enable_ros:
            publish_data([cloud], [poses], [name], cfg=cfg)

        test_loss, _ = min_eigval_loss(cloud, mask=mask)
        print('Test loss on %s: %.9f' % (name, test_loss.item()))
        eval_csv = os.path.join(cfg.log_dir, 'eval_loss.csv')
        append(eval_csv, '%s %.9f\n' % (name, test_loss))


def main():
    cfg = Config()
    cfg.test_names = ['stairs']
    cfg.model_class = 'ScaledPolynomial'
    cfg.model_state_dict = '/home/petrito1/workspace/depth_correction/gen/2022-02-21_16-31-34/088_8.85347e-05_state_dict.pth'
    cfg.pose_correction = PoseCorrection.sequence
    eval_loss(cfg)


if __name__ == '__main__':
    main()

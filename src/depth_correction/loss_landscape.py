from __future__ import absolute_import, division, print_function
from .config import Config, Loss, Model, PoseCorrection
from .dataset import create_dataset, DepthBiasDataset, NoisyDepthDataset, NoisyPoseDataset
from .eval import eval_loss, eval_loss_single
from .model import load_model, model_by_name
from datetime import datetime
from itertools import product
import matplotlib
matplotlib.rcParams['axes.formatter.useoffset'] = False
import matplotlib.pyplot as plt
import numpy as np
import os
import rospy


cfg = Config()
# Synthetic
# cfg.train_names = ['ground_plane/10']
# cfg.train_names = ['open_box/n_6_density_25.0']
# cfg.data_step = 1

# Meshes
# cfg.train_names = ['simple_cave_01.obj']
# cfg.train_names = ['simple_tunnel_01.obj']
# cfg.train_names = ['gazebo_cave_world.ply']
# cfg.train_names = ['burning_building_rubble.ply']
# cfg.train_names = ['newer_college.ply']

# Real
cfg.train_names = []
# cfg.train_names = ['asl_laser/apartment']
# cfg.train_names = ['asl_laser/eth']
cfg.val_names = []
# cfg.test_names = []
cfg.test_names = ['asl_laser/apartment']
cfg.data_step = 4

cfg.dataset = (cfg.train_names + cfg.val_names + cfg.test_names)[0].split('/')[0]

# Depth and grid filters prevent computing L2 loss since the points don't match.
# cfg.min_depth = 0.0
# cfg.max_depth = float('inf')
cfg.min_depth = 1.0
cfg.max_depth = 15.0
# cfg.max_depth = 10.0
# cfg.grid_res = 0.0
# cfg.grid_res = 0.1
cfg.grid_res = 0.1
cfg.nn_k = 0
# cfg.nn_r = 0.2
cfg.nn_r = 0.25
# cfg.nn_r = 0.4
cfg.shadow_angle_bounds = []
# cfg.eigenvalue_bounds = []
# cfg.eigenvalue_bounds = [[0, -float('inf'), (cfg.nn_r / 8)**2],
#                               [1, (cfg.nn_r / 4)**2, float('inf')]]
# cfg.dir_dispersion_bounds = []
# cfg.vp_dispersion_bounds = []
# cfg.log_filters = True
cfg.log_filters = False

# Artificial noise
cfg.depth_bias_model_class = Model.ScaledPolynomial
# cfg.depth_bias_model_kwargs['w'] = [-0.002]
# cfg.depth_bias_model_kwargs['w'] = [0.0]
cfg.depth_bias_model_kwargs['w'] = [0.002]
cfg.depth_bias_model_kwargs['exponent'] = [6.0]
cfg.depth_bias_model_kwargs['learnable_exponents'] = False

cfg.depth_noise = 0.0
# cfg.depth_noise = 0.03
cfg.pose_noise = 0.0
# cfg.pose_noise = 0.005
# cfg.pose_noise = 0.050
cfg.pose_noise_mode = NoisyPoseDataset.Mode.common
# cfg.pose_correction = PoseCorrection.common
cfg.pose_correction = PoseCorrection.none

cfg.model_class = Model.ScaledPolynomial
cfg.model_kwargs['w'] = [0.0]
# cfg.model_kwargs['w'] = [-0.002]
cfg.model_kwargs['exponent'] = [6.0]
cfg.model_kwargs['learnable_exponents'] = False
cfg.model_state_dict = None

cfg.loss = Loss.min_eigval_loss
# cfg.loss = Loss.trace_loss
cfg.loss_offset = False
cfg.loss_kwargs['sqrt'] = False
cfg.loss_kwargs['normalization'] = True

cfg.log_dir = os.path.join(cfg.out_dir, 'loss_landscape', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

def noisy_dataset(ds, cfg: Config):
    if cfg.depth_bias_model_class:
        gt_model = model_by_name(cfg.depth_bias_model_class)(**cfg.depth_bias_model_kwargs)
        if (gt_model.w != 0.0).any():
            print('Adding bias from %s.' % gt_model)
            ds = DepthBiasDataset(ds, gt_model, cfg=cfg)
        else:
            print('No bias added.')
    else:
        print('No bias added.')

    if cfg.depth_noise:
        print('Adding depth noise %.3g.', cfg.depth_noise)
        ds = NoisyDepthDataset(ds, noise=cfg.depth_noise)
    else:
        print('No depth noise added.')

    if cfg.pose_noise:
        print('Adding pose noise %.3g, %s.', cfg.pose_noise, cfg.pose_noise_mode)
        ds = NoisyPoseDataset(ds, noise=cfg.pose_noise, mode=cfg.pose_noise_mode)
    else:
        print('No pose noise added.')
    return ds


def loss_landscape_configs(cfg: Config):
    base_cfg = cfg
    gt_model = model_by_name(cfg.depth_bias_model_class)(**cfg.depth_bias_model_kwargs)
    from depth_correction.model import Polynomial, ScaledPolynomial
    assert isinstance(gt_model, (Polynomial, ScaledPolynomial))
    assert gt_model.w.numel() == 1
    configs = []
    grid_nn_all = [
        # [0.1, 0.2],
        [0.2, 0.4],
        # [0.4, 0.8],
    ]
    eigenvalue_ratio_bounds_all = [
        # [[0, 1, 0.0, 0.25], [1, 2, 0.25, 1.0]],
        # [[0, 1, 0.0, 0.16], [1, 2, 0.25, 1.0]],
        # [[0, 1, 0.0, 0.09], [1, 2, 0.25, 1.0]],
        # [[0, 1, 0.0, 0.04], [1, 2, 0.25, 1.0]],
        # [[0, 1, 0.0, 0.01], [1, 2, 0.25, 1.0]],
        # [[0, 1, 0.0, 0.0025], [1, 2, 0.25, 1.0]],

        # [[0, 1, 0.0, 0.05], [1, 2, 0.25, 1.0]],
        # [[0, 1, 0.0, 0.04], [1, 2, 0.25, 1.0]],
        # [[0, 1, 0.0, 0.03], [1, 2, 0.25, 1.0]],
        # [[0, 1, 0.0, 0.02], [1, 2, 0.25, 1.0]],
        [[0, 1, 0.0, 0.01], [1, 2, 0.25, 1.0]],
    ]
    # w_all = np.linspace(-0.005, 0.005, 21)
    # w_all = np.linspace(-0.003, 0.003, 25)
    # w_all = np.linspace(-0.003, 0.000, 13)
    w_all = np.linspace(-0.004, 0.004, 9)
    # w_all = np.linspace(-0.002, 0.002, 3)
    for (grid_res, nn_r), eigenvalue_ratio_bounds, w in product(grid_nn_all, eigenvalue_ratio_bounds_all, w_all):
        cfg = base_cfg.copy()
        cfg.grid_res = grid_res
        cfg.nn_r = nn_r
        cfg.loss_eval_csv = None  # Don't write eval results.
        cfg.model_kwargs['w'] = [w]
        cfg.eigenvalue_ratio_bounds = eigenvalue_ratio_bounds
        configs.append(cfg)
    return configs

def loss_landscape(cfg: Config):
    """Compute and plot loss landscape for varying parameter values."""
    gt_model = model_by_name(cfg.depth_bias_model_class)(**cfg.depth_bias_model_kwargs)
    # ds = [noisy_dataset(create_dataset(name, cfg), cfg) for name in cfg.train_names]

    results = {}
    # for ds in ds:
    cfgs = loss_landscape_configs(cfg=cfg)
    for i, cfg in enumerate(cfgs):
        # assert len(cfg.train_names) == 1
        print('Computing loss %i / %i...' % (i + 1, len(cfgs)))
        if rospy.is_shutdown():
            raise Exception('Shutdown.')
        model = load_model(cfg=cfg)
        print('Using model: %s.' % model)
        # loss = eval_loss_single(cfg, ds)
        loss = eval_loss(cfg)
        # name = cfg.get_preproc_desc() + ' ' + cfg.get_exp_desc()
        # results.setdefault(name, []).append([model.w.detach(), str(model), str(ds), loss.detach().item()])
        name = ', '.join([cfg.get_preproc_desc(), cfg.get_eigval_ratio_bounds_desc()])
        results.setdefault(name, []).append([model.w.detach(), str(model), loss.detach().item()])

    for name, res in results.items():
        # _, model, _, loss = zip(*res)
        _, model, loss = zip(*res)
        print(name, model, loss)

    # fig, axes = plt.subplots(1, 1, figsize=(12.0, 12.0), constrained_layout=True, squeeze=False)
    fig, axes = plt.subplots(1, 1, figsize=(12.0, 12.0), squeeze=False)
    ax = axes[0, 0]
    ax.cla()
    # print(results)
    ax.axvline(x=gt_model.w[0], color='k', label='ground truth')
    for name, res in results.items():
        # w, model, _, loss = zip(*res)
        w, model, loss = zip(*res)
        p = ax.plot(w, loss, label=name)
        # print(p)
        i_min = min(range(len(loss)), key=loss.__getitem__)
        ax.plot(w[i_min], loss[i_min], 'o', color=p[0].get_color(), mfc='none', label='_nolegend_')
    ax.set_xlabel('Weights')
    ax.set_ylabel('Loss')
    title = ''
    # if len(ds) == 1:
    #     title += str(ds[0])
    if len(cfg.test_names) == 1:
        title += str(cfg.test_names[0])
    if title:
        ax.set_title(title)
    # desc = [cfg.get_exp_desc() for cfg in self.loss_landscape_configs()]
    # ax.set_title(cfg.get_exp_desc())
    ax.grid()
    # ax.legend()
    ax.legend(bbox_to_anchor=(0.0, 1.0), loc="upper left")
    fig.tight_layout()
    plt.pause(10.0)
    # plt.show(block=True)
    # print('loss_landscape end')

    path = os.path.join(cfg.log_dir, 'loss_landscape.png')
    print('Loss landscape written to %s.' % path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300)


def main():
    loss_landscape(cfg)


if __name__ == '__main__':
    main()

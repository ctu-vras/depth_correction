from __future__ import absolute_import, division, print_function
from .config import Config, Loss, Model, NeighborhoodType, PoseCorrection
from .dataset import create_dataset, DepthBiasDataset, NoisyDepthDataset, NoisyPoseDataset
from .depth_cloud import DepthCloud
from .eval import eval_loss, eval_loss_single, eval_loss_planes
from .loss import create_loss
from .model import load_model, model_by_name, Polynomial, ScaledPolynomial
from .preproc import global_cloud
from .segmentation import Planes
from .utils import covs
from .visualization import visualize_incidence_angles
from data.newer_college import dataset_names as newer_college_datasets
from datetime import datetime
from itertools import product
import matplotlib
matplotlib.rcParams['axes.formatter.useoffset'] = False
import matplotlib.pyplot as plt
import numpy as np
import torch
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
cfg.val_names = []
# cfg.test_names = [
#     'asl_laser/apartment',
#     'asl_laser/eth',
#     'asl_laser/stairs',
#     'asl_laser/gazebo_summer',
#     'asl_laser/gazebo_winter',
# ]
# cfg.test_poses_path = len(cfg.test_names) * [None]
# cfg.data_step = 4
cfg.test_names = ['newer_college/%s' % name for name in newer_college_datasets]
# cfg.test_names = cfg.test_names[:2]
cfg.test_poses_path = len(cfg.test_names) * [None]
cfg.data_step = 1
# cfg.test_names = ['open_box/n_6_density_25.0']
# ds = RenderedMeshDataset('simple_cave_01.obj', poses_path='poses_gt_simple_cave_01.csv')
# cfg.test_names = ['rendered_mesh/simple_cave_01.obj']
# cfg.test_poses_path = ['poses_gt_simple_cave_01.csv']
# cfg.data_step = 1

cfg.dataset = (cfg.train_names + cfg.val_names + cfg.test_names)[0].split('/')[0]

# Depth and grid filters prevent computing L2 loss since the points don't match.
cfg.min_depth = 1.0
cfg.max_depth = 15.0
cfg.grid_res = 0.2
# cfg.nn_type = NeighborhoodType.ball
# cfg.nn_k = 0
# cfg.nn_r = 0.2
# cfg.nn_r = 0.25
# cfg.nn_r = 0.4
cfg.nn_type = NeighborhoodType.plane
cfg.nn_r = 0.03
cfg.min_valid_neighbors = 1000
cfg.shadow_angle_bounds = []
# cfg.eigenvalue_bounds = []
# cfg.eigenvalue_bounds = [[0, -float('inf'), (cfg.nn_r / 8)**2],
#                               [1, (cfg.nn_r / 4)**2, float('inf')]]
# cfg.dir_dispersion_bounds = []
# cfg.vp_dispersion_bounds = []
# cfg.log_filters = True
cfg.log_filters = False

# Artificial noise
# cfg.depth_bias_model_class = Model.ScaledPolynomial
cfg.depth_bias_model_class = None
# cfg.depth_bias_model_kwargs['w'] = [-0.002]
cfg.depth_bias_model_kwargs['w'] = [0.0]
# cfg.depth_bias_model_kwargs['w'] = [0.002]
cfg.depth_bias_model_kwargs['exponent'] = [4.0]
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
cfg.model_kwargs['exponent'] = [4.0]
cfg.model_kwargs['learnable_exponents'] = False
cfg.model_state_dict = None

cfg.loss = Loss.min_eigval_loss
# cfg.loss = Loss.trace_loss
cfg.loss_offset = False
cfg.loss_kwargs['sqrt'] = False
cfg.loss_kwargs['normalization'] = True

cfg.log_dir = os.path.join(cfg.out_dir, 'loss_landscape', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


def eval_loss_planes(cfg: Config, model=None, loss_fun=None, planes=None):
    if model is None:
        model = load_model(cfg=cfg, eval_mode=True)

    if loss_fun is None:
        loss_fun = create_loss(cfg)
    assert callable(loss_fun)

    assert len(cfg.test_names) == 1
    kwargs = {}
    if cfg.test_poses_path:
        assert len(cfg.test_poses_path) == 1
        kwargs['poses_path'] = cfg.test_poses_path[0]
    ds = create_dataset(cfg.test_names[0], cfg, **kwargs)
    name = str(ds)
    print('Using dataset %s with %i clouds.' % (name, len(ds)))
    clouds = []
    poses = []
    for cloud, pose in ds:
        cloud = DepthCloud.from_structured_array(cloud, cfg.numpy_float_type())
        clouds.append(cloud)
        poses.append(pose)
    poses = np.stack(poses).astype(dtype=cfg.numpy_float_type())
    poses = torch.as_tensor(poses, device=cfg.device)
    cloud = global_cloud(clouds, None, poses)
    print('Global cloud %s contains %i points.' % (name, len(cloud)))
    if planes is None:
        planes = Planes.fit(cloud, cfg.ransac_dist_thresh, min_support=cfg.min_valid_neighbors,
                            max_iterations=cfg.num_ransac_iters, max_models=cfg.max_neighborhoods,
                            eps=2.0 * np.sqrt(3) * cfg.grid_res,
                            visualize_progress=False, visualize_final=cfg.log_filters, verbose=0)
        # planes.visualize(cloud)
    n_used = sum(len(idx) for idx in planes.indices)
    print('Testing on %.3f = %i / %i points from %s.' % (n_used / cloud.size(), n_used, cloud.size(), name))
    # Update cloud incidence angles from normals and ray directions.
    # segmented = cloud.clone()
    plane_clouds = []
    covs_all = []
    eigvals_all = []
    for i in range(planes.size()):
        plane_cloud = cloud[planes.indices[i]]
        plane_cloud.normals = planes.params[i:i + 1, :-1].expand((len(planes.indices[i]), -1))
        plane_cloud.update_incidence_angles()
        plane_cloud = model(plane_cloud)
        plane_clouds.append(plane_cloud)
        x = plane_cloud.to_points()
        cov = covs(x)
        covs_all.append(cov)
        eigvals_all.append(torch.linalg.eigh(cov)[0])
    # if planes is None:
    visualize_incidence_angles(plane_clouds)
    planes.cov = torch.stack(covs_all)
    planes.eigvals = torch.stack(eigvals_all)
    test_loss, _ = loss_fun(planes)
    return test_loss, planes


def loss_landscape_configs(cfg: Config):
    base_cfg = cfg
    # gt_model = model_by_name(cfg.depth_bias_model_class)(**cfg.depth_bias_model_kwargs)
    # from depth_correction.model import Polynomial, ScaledPolynomial
    # assert isinstance(gt_model, (Polynomial, ScaledPolynomial))
    # assert gt_model.w.numel() == 1
    configs = []
    # (grid_res, nn_r = ransac thresh, min_valid_neighbors = support)
    grid_nn_all = [
        # ball NN
        # [0.1, 0.2, 5],
        # [0.2, 0.4, 5],
        # [0.4, 0.8, 5],
        # planes NN
        # [0.1, 0.03, 1000],
        # [0.1, 0.02, 1000],
        [0.2, 0.03, 250],
        # [0.2, 0.02, 250],
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
        # [[0, 1, 0.0, 0.01], [1, 2, 0.25, 1.0]],
        [],
    ]
    # w_all = np.linspace(-0.004, 0.004, 9)
    w_all = np.linspace(-0.005, 0.005, 21)
    # w_all = np.linspace(-0.005, 0.005, 7)
    # w_all = np.linspace(-0.016, 0.016, 17)
    for (name, poses_path), (grid_res, nn_r, min_valid_neighbors), eigenvalue_ratio_bounds, w in product(
            zip(base_cfg.test_names, base_cfg.test_poses_path), grid_nn_all, eigenvalue_ratio_bounds_all, w_all):
        cfg = base_cfg.copy()
        cfg.test_names = [name]
        cfg.test_poses_path = [poses_path]
        cfg.grid_res = grid_res
        cfg.nn_r = nn_r
        cfg.loss_eval_csv = None  # Don't write eval results.
        cfg.model_kwargs['w'] = [w]
        cfg.eigenvalue_ratio_bounds = eigenvalue_ratio_bounds
        cfg.min_valid_neighbors = min_valid_neighbors

        configs.append(cfg)
    return configs

def loss_landscape(cfg: Config):
    """Compute and plot loss landscape for varying parameter values."""
    base_cfg = cfg
    if cfg.depth_bias_model_class:
        gt_model = model_by_name(cfg.depth_bias_model_class)(**cfg.depth_bias_model_kwargs)
    else:
        gt_model = None

    results = {}
    # for ds in ds:
    cfgs = loss_landscape_configs(cfg=base_cfg)
    planes = {}
    for i, cfg in enumerate(cfgs):
        print('Computing loss %i / %i on %s...'
              % (i + 1, len(cfgs), ', '.join(cfg.test_names)))
        if rospy.is_shutdown():
            raise Exception('Shutdown.')
        model = load_model(cfg=cfg)
        # print('Using model: %s.' % model)
        if cfg.nn_type == NeighborhoodType.ball:
            loss = eval_loss(cfg)
        elif cfg.nn_type == NeighborhoodType.plane:
            assert len(cfg.test_names) == 1
            key = cfg.get_preproc_desc() + '/' + cfg.test_names[0]
            if key in planes:
                loss, _ = eval_loss_planes(cfg, planes=planes[key])
            else:
                loss, planes[key] = eval_loss_planes(cfg)
        else:
            assert False
        if cfg.nn_type == NeighborhoodType.ball:
            # name = ', '.join([cfg.get_preproc_desc(), cfg.get_eigval_ratio_bounds_desc()])
            name = ', '.join([', '.join(cfg.test_names),
                              cfg.get_grid_filter_desc(),
                              cfg.get_depth_filter_desc(),
                              cfg.get_eigval_ratio_bounds_desc()])
        elif cfg.nn_type == NeighborhoodType.plane:
            name = ', '.join([', '.join(cfg.test_names),
                              cfg.get_grid_filter_desc(),
                              cfg.get_depth_filter_desc(),
                              cfg.get_nn_desc(),
                              'mvn%i' % cfg.min_valid_neighbors])
        results.setdefault(name, []).append([model.w.detach().item(), str(model), loss.detach().item()])

    for name, res in results.items():
        # _, model, _, loss = zip(*res)
        _, model, loss = zip(*res)
        print(name, model, loss)

    # fig, axes = plt.subplots(1, 1, figsize=(12.0, 12.0), constrained_layout=True, squeeze=False)
    fig, axes = plt.subplots(1, 1, figsize=(8.0, 8.0), squeeze=False)
    ax = axes[0, 0]
    ax.cla()
    gt_w = gt_model.w.detach()[0] if isinstance(gt_model, (Polynomial, ScaledPolynomial)) else 0.0
    ax.axvline(x=gt_w, color='k', label='ground truth')
    for name, res in results.items():
        w, model, loss = zip(*res)
        p = ax.plot(w, loss, label=name)
        i_min = min(range(len(loss)), key=loss.__getitem__)
        ax.plot(w[i_min], loss[i_min], 'o', color=p[0].get_color(), mfc='none', label='_nolegend_')
    ax.set_xlabel('Weights')
    ax.set_ylabel('Loss')
    title = ''
    # if len(ds) == 1:
    #     title += str(ds[0])
    if len(base_cfg.test_names) == 1:
        title += str(base_cfg.test_names[0])
    if title:
        ax.set_title(title)
    # desc = [cfg.get_exp_desc() for cfg in self.loss_landscape_configs()]
    # ax.set_title(cfg.get_exp_desc())
    ax.grid()
    # ax.legend()
    # ax.legend(bbox_to_anchor=(0.0, 1.0), loc="upper left")
    # ax.legend(loc=(1.04, 1.0))
    ax.legend(loc=(0.0, 1.04))
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

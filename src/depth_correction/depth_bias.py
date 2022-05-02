from __future__ import absolute_import, division, print_function
from .config import Config
from .depth_cloud import DepthCloud
from .model import load_model
from .preproc import *
from glob import glob
# import matplotlib
# matplotlib.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
import open3d as o3d
import torch


__all__ = [
    'demo',
    'main',
    'plot_depth_bias',
    'plot_fit',
]


def lims(x):
    return np.nanquantile(x, [0.001, 0.999])


def domain(model, n=100):
    if isinstance(model, Polynomial):
        return np.linspace(model.domain[0], model.domain[1], n)
    if isinstance(model, np.ndarray):
        return np.linspace(np.nanmin(model), np.nanmax(model), n)
    raise ValueError('Invalid domain input, only polynomial or data sample is supported.')


def plot_fit(x, y, y_corr=None, x_label='x', y_label='y', x_lims=None, y_lims=None):
    if x_lims is None:
        x_lims = lims(x)
    if y_lims is None:
        y_lims = lims(y)

    poly1 = Polynomial.fit(x, y, 1).convert()
    print()
    print('%s to %s (deg. 1 fit): %s' % (y_label, x_label, poly1))
    if y_corr is not None:
        poly1_corr = Polynomial.fit(x, y_corr, 1).convert()
        print('%s to %s (deg. 1 fit): %s' % (y_label + ' corr.', x_label, poly1_corr))
    # poly2 = Polynomial.fit(x, y, 2).convert()
    # print('%s to %s (deg. 2 fit): %s' % (y_label, x_label, poly2))
    # xs = domain(poly1)
    xs = domain(x)
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4))
    ax.plot(x, y, 'r.', markersize=0.5, alpha=0.2, label='data')
    ax.plot(xs, poly1(xs), 'r--', linewidth=2, label='fit deg. 1')
    if y_corr is not None:
        ax.plot(x, y_corr, 'b.', markersize=0.5, alpha=0.2, label='data corr.')
        ax.plot(xs, poly1_corr(xs), 'b--', linewidth=2, label='fit deg. 1 corr.')
    # ax.plot(xs, poly2(xs), 'g--', linewidth=2, label='fit deg. 2')
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.show()


def plot_depth_bias(ds, cfg: Config):

    print(ds.path)
    if cfg.model_state_dict:
        model = load_model(cfg=cfg, eval_mode=True)
    else:
        model = None

    clouds = []
    poses = []
    for cloud, pose in ds[::cfg.data_step]:
        cloud = filtered_cloud(cloud, cfg)
        cloud = local_feature_cloud(cloud, cfg)
        clouds.append(cloud)
        poses.append(torch.as_tensor(pose))

    # offset_cloud = DepthCloud.concatenate(clouds, fields=DepthCloud.source_fields + ['cov', 'eigvals'])

    cloud = global_cloud(clouds, None, poses)
    cloud.update_all(k=cfg.nn_k, r=cfg.nn_r)
    cloud.mask = global_cloud_mask(cloud, cloud.mask, cfg)
    print('%.3f = %i / %i points kept (all filters).'
          % (cloud.mask.float().mean(), cloud.mask.sum(), cloud.mask.numel()))
    cloud.visualize(window_name='Point Mask', colors=cloud.mask, colormap=cm.viridis)

    # extracted = cloud[cloud.mask]
    # extracted.update_all(k=cfg.nn_k, r=cfg.nn_r)
    # extracted.mask = global_cloud_mask(extracted, extracted.mask, cfg)
    # print('%.3f = %i / %i extracted points kept (all filters).'
    #       % (extracted.mask.float().mean(), extracted.mask.sum(), extracted.mask.numel()))
    # extracted.visualize(window_name='Extracted Point Mask', colors=extracted.mask, colormap=cm.viridis)

    return

    if model is not None:
        cloud_corr = global_cloud(clouds, model, poses)
        cloud_corr.update_all(k=cfg.nn_k, r=cfg.nn_r)
        # cloud.visualize(colors=torch.sqrt(cloud_corr.eigvals[:, 0]))

    # Visualize plane distance to incidence angle.
    # TODO: Compare using plane fit for low incidence angle?
    depth = cloud.depth.detach().numpy().ravel()
    inc = cloud.inc_angles.detach().numpy().ravel()
    inv_cos = 1.0 / np.cos(inc)
    dist = (cloud.normals * (cloud.points - cloud.mean)).sum(dim=1).detach().numpy().ravel()
    norm_dist = dist / depth
    print('Dist: %.6f +- %.6f m, norm dist: %.6f +- %.6f'
          % (dist.mean(), dist.std(), norm_dist.mean(), norm_dist.std()))

    if cfg.model_state_dict:
        dist_corr = (cloud_corr.normals * (cloud_corr.points - cloud_corr.mean)).sum(dim=1).detach().numpy().ravel()
        norm_dist_corr = dist_corr / depth
        print('Dist. corr.: %.6f +- %.6f m, norm. dist. corr.: %.6f +- %.6f'
              % (dist_corr.mean(), dist_corr.std(), norm_dist_corr.mean(), norm_dist_corr.std()))

    # Fit models dependent on incidence angle
    plot_fit(depth, dist, y_corr=dist_corr,
             x_label='Depth [m]', y_label='Distance to Plane [m]')
    plot_fit(inc, dist, y_corr=dist_corr,
             x_label='Incidence Angle', y_label='Distance to Plane [m]',
             x_lims=[0.0, np.pi / 2])
    plot_fit(inc, norm_dist, y_corr=norm_dist_corr,
             x_label='Incidence Angle', y_label='Distance to Plane / Depth',
             x_lims=[0.0, np.pi / 2])
    # plot_fit(inv_cos, norm_dist,
    #          x_label='1 / Incidence Angle Cosine', y_label='Distance to Plane / Depth',
    #          x_lims=[1.0, 11.47])


def demo():
    cfg = Config()
    # cfg.dataset = 'asl_laser'
    cfg.dataset = 'semantic_kitti'
    # Load config from file.
    cfg_path = None
    # cfg_path = glob('%s/gen/%s_*/ground_truth*ScaledPolynomial*/split_*/best.yaml' % (cfg.pkg_dir, cfg.dataset))[0]
    if cfg_path:
        print(cfg_path)
        cfg.from_yaml(cfg_path)
    elif cfg.dataset == 'asl_laser':
        cfg.grid_res = 0.1
        cfg.nn_r = 0.2
    elif cfg.dataset == 'semantic_kitti':
        cfg.grid_res = 0.2
        cfg.nn_r = 0.4
    else:
        raise ValueError()

    if cfg.dataset == 'asl_laser':
        from data.asl_laser import Dataset, dataset_names
    elif cfg.dataset == 'semantic_kitti':
        from data.semantic_kitti import Dataset, dataset_names
        cfg.min_depth = 3.0

    # cfg.eigenvalue_bounds = []

    for name in dataset_names:
        print()
        print(name)
        ds = Dataset(name)
        plot_depth_bias(ds, cfg)


def main():
    demo()


if __name__ == '__main__':
    main()

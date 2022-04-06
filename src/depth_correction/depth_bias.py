from __future__ import absolute_import, division, print_function
from .config import Config
from .depth_cloud import DepthCloud
from .filters import filter_eigenvalue, filter_eigenvalues, filter_depth, filter_grid
from .preproc import *
from .utils import timer, timing
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
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


def plot_fit(x, y, x_label='x', y_label='y', x_lims=None, y_lims=None):
    if x_lims is None:
        x_lims = lims(x)
    if y_lims is None:
        y_lims = lims(y)

    poly1 = Polynomial.fit(x, y, 1).convert()
    poly2 = Polynomial.fit(x, y, 2).convert()
    print('%s to %s (deg. 1 fit): %s' % (y_label, x_label, poly1))
    print('%s to %s (deg. 2 fit): %s' % (y_label, x_label, poly2))
    # xs = domain(poly1)
    xs = domain(x)
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4))
    ax.plot(x, y, '.', markersize=0.5, label='data')
    ax.plot(xs, poly1(xs), 'r-', linewidth=2, label='fit deg. 1')
    ax.plot(xs, poly2(xs), 'g--', linewidth=2, label='fit deg. 2')
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.show()


def plot_depth_bias(ds, cfg: Config):
    clouds = []
    poses = []
    for cloud, pose in ds:
        cloud = filtered_cloud(cloud, cfg)
        cloud = local_feature_cloud(cloud, cfg)
        clouds.append(cloud)
        poses.append(torch.as_tensor(pose))

    cloud = global_cloud(clouds, None, poses)
    # cloud.visualize(colors='inc_angles')
    # cloud.visualize(colors='z')
    cloud.update_all(k=cfg.nn_k, r=cfg.nn_r)
    # cloud.visualize(colors='inc_angles')
    # cloud.visualize(colors='min_eigval')
    cloud.visualize(colors=torch.sqrt(cloud.eigvals[:, 0]))

    # Select planar regions to estimate bias.
    mask = filter_eigenvalues(cloud, eig_bounds=cfg.eigenvalue_bounds, only_mask=True, log=True)
    if cfg.eigenvalue_bounds:
        cloud.visualize(colors=mask)

    return

    # Visualize plane distance to incidence angle.
    # TODO: Compare using plane fit for low incidence angle?
    depth = cloud.depth.detach().numpy().ravel()
    inc = cloud.inc_angles.detach().numpy().ravel()
    inv_cos = 1.0 / np.cos(inc)
    dist = (cloud.normals * (cloud.points - cloud.mean)).sum(dim=1).detach().numpy().ravel()
    norm_dist = dist / depth

    # Fit models dependent on incidence angle
    plot_fit(depth, dist,
             x_label='Depth [m]', y_label='Distance to Plane [m]')
    plot_fit(inc, dist,
             x_label='Incidence Angle', y_label='Distance to Plane [m]',
             x_lims=[0.0, np.pi / 2])
    plot_fit(inc, norm_dist,
             x_label='Incidence Angle', y_label='Distance to Plane / Depth',
             x_lims=[0.0, np.pi / 2])
    plot_fit(inv_cos, norm_dist,
             x_label='1 / Incidence Angle Cosine', y_label='Distance to Plane / Depth',
             x_lims=[1.0, 11.47])


def demo():
    cfg = Config()
    cfg.grid_res = 0.1
    cfg.nn_k = 15
    # cfg.nn_r = 0.4
    # cfg.nn_r = None
    # cfg.eigenvalue_bounds = [[0, None, (cfg.nn_r / 8)**2],
    #                          [1, (cfg.nn_r / 4)**2, None]]
    cfg.eigenvalue_bounds = []
    for k, v in cfg.non_default().items():
        print('%s: %s' % (k, v))

    # from data.asl_laser import Dataset, dataset_names
    # dataset_names = ['eth']
    # dataset_names = ['apartment', 'eth', 'gazebo_summer', 'gazebo_winter', 'stairs']
    from data.semantic_kitti import Dataset, dataset_names
    for name in dataset_names:
        print(name)
        ds = Dataset(name)
        ds = ds[::cfg.data_step]
        plot_depth_bias(ds, cfg)


def main():
    demo()


if __name__ == '__main__':
    main()

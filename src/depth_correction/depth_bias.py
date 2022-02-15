from __future__ import absolute_import, division, print_function
from .depth_cloud import DepthCloud
from .filters import filter_eigenvalue, filter_depth, filter_grid
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
    # print(np.nanquantile(x, np.linspace(0.0, 1.0, 10)))
    # print(np.nanquantile(y, np.linspace(0.0, 1.0, 10)))


def plot_depth_bias(ds):
    min_depth = 1.0
    max_depth = 15.0
    grid_res = 0.05
    k = None
    r = 3 * grid_res
    device = torch.device('cpu')
    # device = torch.device('cuda')

    clouds = []
    poses = []
    for cloud, pose in ds:
        # Filter and subsample input cloud.
        cloud = filter_depth(cloud, min=min_depth, max=max_depth)
        cloud = filter_grid(cloud, grid_res=grid_res, keep='last')

        # Convert to depth cloud and transform.
        cloud = DepthCloud.from_points(cloud)
        cloud = cloud.type(dtype=torch.float64)
        cloud = cloud.to(device=device)
        pose = torch.tensor(pose, device=device)
        cloud = cloud.transform(pose)

        # Find/update neighbors and estimate all features.
        # cloud.update_all(k=k, r=r)
        # Select planar regions to estimate bias.
        # mask = filter_eigenvalue(cloud, 0, max=(grid_res / 5)**2, only_mask=True)
        # mask = mask & filter_eigenvalue(cloud, 1, min=grid_res**2, only_mask=True)
        # cloud.mask = mask

        # cloud.update_all(k=k, r=r)
        clouds.append(cloud)
        poses.append(pose)

    cloud = DepthCloud.concatenate(clouds, True)
    cloud.visualize(colors='z')
    # cloud.visualize(colors='inc_angles')
    cloud.update_all(k=k, r=r)
    cloud.visualize(colors='inc_angles')

    # Select planar regions to estimate bias.
    mask = filter_eigenvalue(cloud, 0, max=(grid_res / 5)**2, only_mask=True, log=True)
    mask = mask & filter_eigenvalue(cloud, 1, min=grid_res**2, only_mask=True, log=True)
    # cloud.mask = mask
    cloud = cloud[mask]

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
    from data.asl_laser import Dataset, dataset_names
    dataset_names = ['eth']
    # dataset_names = ['apartment', 'eth', 'gazebo_summer', 'gazebo_winter', 'stairs']
    for name in dataset_names:
        ds = Dataset(name)
        ds = ds[::4]
        plot_depth_bias(ds)


def main():
    demo()


if __name__ == '__main__':
    main()

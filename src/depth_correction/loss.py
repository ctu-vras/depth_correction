from __future__ import absolute_import, division, print_function
from .depth_cloud import DepthCloud
from .filters import filter_eigenvalue, filter_depth, filter_grid
from .nearest_neighbors import nearest_neighbors
import numpy as np
from numpy.polynomial import Polynomial
from random import shuffle
import torch
from timeit import default_timer as timer


refractive_index_vacuum = 1.0
refractive_index_air = 1.000293

# waist_hokuyo =
# waist_ouster = 5e-3  # FWHM?

wavelength_hokuyo = 905e-9
wavelength_ouster = 865e-9


__all__ = [
    'beam_width',
    'min_eigval_loss',
    'neighbor_cov',
    'neighbor_fun',
    'rayleight_length',
    'reduce',
    'trace_loss',
]


def rayleight_length(w0, n=refractive_index_air, l=wavelength_hokuyo):
    """Rayleight lenght (range) for given beam waist.

    :param w0: beam waist [m].
    :param n: index of refraction of the propagation medium, n=1.0 for vacuum, n=1.000293 for air (default).
    :param
    """
    assert isinstance(w0, torch.Tensor)
    z_r = torch.pi * w0**2 * n / l
    return z_r


def beam_width(z, w0, n=refractive_index_air, l=wavelength_hokuyo, m2=1.0):
    """Beam width at given depth.

    :param z: depth [m].
    :param w0: beam waist [m].
    :param n: index of refraction of the propagation medium.
    :param l: wavelength [m].
    :param m2: M2, "M squared", beam quality factor, or beam propagation factor. m2=1 for ideal Gaussian beam (default).
    :return: Beam width [m].
    """
    assert isinstance(z, torch.Tensor)
    assert isinstance(w0, torch.Tensor)
    w = w0 * m2 * torch.sqrt(1.0 + (z / rayleight_length(w0, n, l))**2)
    return w


def reduce(x, reduction='mean', weights=None, only_finite=False, skip_nans=False):
    assert reduction in ('none', 'mean', 'sum')

    keep = None
    if only_finite:
        keep = ~x.isfinite()
    elif skip_nans:
        keep = ~x.isnan()
    if keep is not None:
        if weights:
            weights = weights[keep]
        x = x[keep]

    if reduction == 'mean':
        if weights is None:
            x = x.mean()
        else:
            x = (weights * x).sum() / weights.sum()
    elif reduction == 'sum':
        if weights is None:
            x = x.sum()
        else:
            x = (weights * x).sum()

    return x


def neighbor_fun(points, fun, query=None, k=None, r=None):
    assert isinstance(points, torch.Tensor)
    assert isinstance(query, torch.Tensor)
    assert callable(fun)
    assert k is None or (isinstance(k, int) and k >= 1)
    assert r is None or (isinstance(r, float) and r > 0.0)

    dist, ind = nearest_neighbors(points, query, k=k, r=r)

    # TODO: Allow batch dimension.
    # n = query.shape[-1]
    n = query.shape[0]
    result = []
    for i in range(n):
        nn = torch.index_select(points, 0, torch.tensor(ind[i]))
        q = query[i:i + 1]
        result.append(fun(nn, q))

    return result


def neighbor_cov(points, query=None, k=None, r=None, correction=1):
    fun = lambda p, q: torch.cov(p.transpose(-1, -2), correction=correction)
    cov = neighbor_fun(points, fun, query=query, k=k, r=r)
    cov = torch.stack(cov)
    return cov


def min_eigval_loss(cloud, k=None, r=None,
                    max_angle=None,
                    eigenvalue_bounds=None,
                    offset=False,
                    reduction='mean',
                    invalid=0.):
    assert isinstance(cloud, DepthCloud)
    assert k is None or isinstance(k, int)
    assert r is None or isinstance(r, float)
    assert eigenvalue_bounds is None or len(eigenvalue_bounds) == 2

    # dc = cloud.copy()
    dc = cloud.deepcopy()
    # dc.update_all(k=k, r=r)
    dc.update_points()
    dc.update_neighbors(k=k, r=r)
    if max_angle is not None:
        dc.filter_neighbors_normal_angle(max_angle)
    dc.update_features()
    dc.loss = dc.eigvals[:, 0]

    if offset:
        assert cloud.eigvals is not None
        dc.loss = dc.loss - cloud.eigvals[:, 0]

    if eigenvalue_bounds is not None:
        dc = filter_eigenvalue(dc, 0, min=eigenvalue_bounds[0], max=eigenvalue_bounds[1])

    dc.loss = torch.relu(dc.loss)
    loss = reduce(dc.loss, reduction=reduction)
    return loss, dc


def trace_loss(points, query=None, k=None, r=None, reduction='mean', invalid=0.):
    invalid = torch.tensor(invalid)
    fun = lambda p, q: torch.cov(p.transpose(-1, -2)).trace() if p.shape[0] >= 3 else invalid
    loss = neighbor_fun(points, fun, query=query, k=k, r=r)
    loss = torch.stack(loss)
    loss = reduce(loss, reduction=reduction)
    return loss


def show_cloud(cloud, colormap=None):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    o3d.visualization.draw_geometries([pcd])


def demo():
    from data.asl_laser import Dataset, dataset_names

    clouds = []
    poses = []
    # ds = Dataset('apartment')
    # ids = ds.ids[0:10:2]
    ds = Dataset('eth')
    # ids = ds.ids[::6]
    # ids = [0, 10]
    # ds = Dataset('gazebo_summer')
    # ds = Dataset('gazebo_winter')
    # ds = Dataset('stairs')
    # ids = ds.ids[0:10:2]
    # step = 10
    # start = 10
    # stop = start + step
    # ids = ds.ids[start:stop:step]
    # ids = ds.ids[::10]
    ids = ds.ids[10:21:10]

    min_depth = 1.0
    max_depth = 15.0
    grid_res = 0.05
    k = None
    # k = 9
    # r = None
    # r = 0.15
    r = 3 * grid_res
    for id in ids:
        t = timer()
        cloud = ds.local_cloud(id)
        pose = torch.tensor(ds.cloud_pose(id))
        dc = DepthCloud.from_points(cloud)
        # print('%i points read from dataset %s, cloud %i (%.3f s).'
        #       % (dc.size(), ds.name, id, timer() - t))

        dc = filter_depth(dc, min=min_depth, max=max_depth, log=False)

        t = timer()
        dc = filter_grid(dc, grid_res, keep='last')
        # print('%i points kept by grid filter with res. %.2f m (%.3f s).'
        #       % (dc.size(), grid_res, timer() - t))

        dc = dc.transform(pose)
        dc.update_all(k=k, r=r)
        # keep = filter_eigenvalue(dc, 0, max=(grid_res / 5)**2, only_mask=True)
        # keep = keep & filter_eigenvalue(dc, 1, min=grid_res**2, only_mask=True)
        # dc = dc[keep]
        # dc.update_all(r=r)

        clouds.append(dc)
        poses.append(pose)

    dc = DepthCloud.concatenate(clouds, True)
    # dc.visualize(colors='inc_angles')
    # dc.visualize(colors='z')

    dc.update_all(k=k, r=r)

    # Visualize incidence angle to plane distance.
    # TODO: Compare using plane fit for low incidence angle.
    depth = dc.depth.detach().numpy().ravel()
    inc = dc.inc_angles.detach().numpy().ravel()
    # scaled_inc = depth * inc
    inv_cos = 1.0 / np.cos(inc)
    # scaled_inv_cos = depth * inv_cos
    # dist = dc.normals.inner(dc.points - dc.mean)
    dist = (dc.normals * (dc.points - dc.mean)).sum(dim=1).detach().numpy().ravel()
    norm_dist = dist / depth

    # Fit models dependent on incidence angle
    def domain(model, n=100):
        if isinstance(model, Polynomial):
            return np.linspace(model.domain[0], model.domain[1], n)
        if isinstance(model, np.ndarray):
            return np.linspace(np.nanmin(model), np.nanmax(model), n)
        raise ValueError('Invalid domain input, only polynomial or data sample is supported.')

    def lims(x):
        return np.nanquantile(x, [0.001, 0.999])

    import matplotlib.pyplot as plt
    # figsize = 8.27, 8.27
    figsize = 6.4, 6.4

    def plot_fit(x, y, x_label='x', y_label='y', x_lims=None, y_lims=None):
        if x_lims is None:
            x_lims = lims(x)
        if y_lims is None:
            y_lims = lims(y)
        poly1 = Polynomial.fit(x, y, 1)
        poly2 = Polynomial.fit(x, y, 2)
        print('%s to %s (deg. 1 fit): %s' % (y_label, x_label, poly1))
        print('%s to %s (deg. 2 fit): %s' % (y_label, x_label, poly2))
        # xs = domain(poly1)
        xs = domain(x)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(x, y, '.', markersize=1, label='data')
        ax.plot(xs, poly1(xs), 'r-', linewidth=1, label='fit deg. 1')
        ax.plot(xs, poly2(xs), 'g--', linewidth=1, label='fit deg. 2')
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

    plot_fit(inc, dist,
             'Incidence Angle', 'Distance to Plane [m]')
    plot_fit(inc, norm_dist,
             'Incidence Angle', 'Distance to Plane / Depth')
    plot_fit(inv_cos, norm_dist,
             '1 / Incidence Angle Cosine', 'Distance to Plane / Depth',
             x_lims=[1.0, 11.47])

    return

    # combined.filter_neighbors_normal_angle(np.radians(30.))
    eigval_bounds = (0.0, 0.05**2)
    # max_angle = None
    max_angle = np.radians(30.)
    loss, loss_dc = min_eigval_loss(combined, r=r, offset=True,
                                    eigenvalue_bounds=eigval_bounds,
                                    max_angle=max_angle)

    print('Loss: %.6g' % loss.item())
    loss_dc.visualize(colors='loss')


def main():
    demo()


if __name__ == '__main__':
    main()

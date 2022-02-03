from __future__ import absolute_import, division, print_function
from .depth_cloud import DepthCloud
from .filters import filter_eigenvalue, filter_depth, filter_grid
from .nearest_neighbors import nearest_neighbors
import numpy as np
from numpy.polynomial import Polynomial
from random import shuffle
import torch
from timeit import default_timer as timer


__all__ = [
    'min_eigval_loss',
    'neighbor_cov',
    'neighbor_fun',
    'reduce',
    'trace_loss',
]


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
    # TODO: segmentation fault in update_neighbors
    dc.update_neighbors(k=k, r=r)
    if max_angle is not None:
        dc.filter_neighbors_normal_angle(max_angle)
    dc.update_features()
    print('Updated features')
    dc.loss = dc.eigvals[:, 0]

    if offset:
        assert cloud.eigvals is not None
        dc.loss = dc.loss - cloud.eigvals[:, 0]

    if eigenvalue_bounds is not None:
        keep = filter_eigenvalue(dc, 0, min=eigenvalue_bounds[0], max=eigenvalue_bounds[1], only_mask=True)
        dc.eigvals = dc.eigvals[keep]
        # assert len(eigenvalue_bounds) == 2
        # assert eigenvalue_bounds[0] <= eigenvalue_bounds[1]
        # out_of_bounds = ((dc.eigvals[:, 0] < eigenvalue_bounds[0])
        #                  | (dc.eigvals[:, 0] > eigenvalue_bounds[1]))
        # dc.loss[out_of_bounds] = 0.0
        #
        # n_out = out_of_bounds.sum().item()
        # n_total = out_of_bounds.numel()
        # print('%i / %i = %.1f %% eigenvalue out of bounds (new).'
        #       % (n_out, n_total, 100 * n_out / n_total))

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
    # ds = Dataset('eth')
    # ids = ds.ids[::6]
    # ids = [0, 10]
    # ds = Dataset('gazebo_summer')
    # ds = Dataset('gazebo_winter')
    ds = Dataset('stairs')
    # ids = ds.ids[0:10:2]
    ids = ds.ids[::2]

    min_depth = 1.0
    max_depth = 10.0
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
        print('%i points read from dataset %s, cloud %i (%.3f s).'
              % (dc.size(), ds.name, id, timer() - t))

        dc = filter_depth(dc, min=min_depth, max=max_depth)

        t = timer()
        dc = filter_grid(dc, grid_res, keep='last')
        print('%i points kept by grid filter with res. %.2f m (%.3f s).'
              % (dc.size(), grid_res, timer() - t))

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
    dc.visualize(colors='z')

    dc.update_all(k=k, r=r)

    # Visualize incidence angle to plane distance.
    # TODO: Compare using plane fit for low incidence angle.
    inc_angles = (180.0 / np.pi) * dc.inc_angles.detach().numpy().ravel()
    # dist = dc.normals.inner(dc.points - dc.mean)
    dist = (dc.normals * (dc.points - dc.mean)).sum(dim=1).detach().numpy().ravel()
    poly1 = Polynomial.fit(inc_angles, dist, 1)
    print(poly1)
    # Negative slope: distance to plane decreases with incidence angle.
    poly2 = Polynomial.fit(inc_angles, dist, 2)
    print(poly2)
    xs = np.linspace(poly1.domain[0], poly1.domain[1], 100)

    import matplotlib.pyplot as plt
    plt.plot(inc_angles, dist, '.', markersize=1, label='data')
    plt.plot(xs, poly1(xs), 'r-', linewidth=1, label='fit deg. 1')
    plt.plot(xs, poly2(xs), 'g--', linewidth=1, label='fit deg. 2')
    plt.xticks(np.linspace(0., 90., 10))
    plt.xlabel('Incidence Angle [deg]')
    plt.ylabel('Distance to Plane [m]')
    plt.legend()
    plt.show()

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

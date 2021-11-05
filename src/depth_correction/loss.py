from __future__ import absolute_import, division, print_function
from .depth_cloud import DepthCloud
from .filters import filter_grid
from .nearest_neighbors import nearest_neighbors
import numpy as np
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


def min_eigval_loss(cloud, query=None, k=None, r=None, offset=False,
                    input_eigval_bounds=None, updated_eigval_bounds=None,
                    max_angle=None, depth_bounds=None, reduction='mean',
                    invalid=0.):
    if isinstance(cloud, DepthCloud):
        dc = cloud.copy()
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

        if input_eigval_bounds is not None:
            assert len(input_eigval_bounds) == 2
            assert input_eigval_bounds[0] <= input_eigval_bounds[1]
            out_of_bounds = ((cloud.eigvals[:, 0] < input_eigval_bounds[0])
                             | (cloud.eigvals[:, 0] > input_eigval_bounds[1]))
            dc.loss[out_of_bounds] = 0.0

            n_out = out_of_bounds.sum().item()
            n_total = out_of_bounds.numel()
            print('%i / %i = %.1f %% out of bounds (input).'
                  % (n_out, n_total, 100 * n_out / n_total))

        if updated_eigval_bounds is not None:
            assert len(updated_eigval_bounds) == 2
            assert updated_eigval_bounds[0] <= updated_eigval_bounds[1]
            out_of_bounds = ((dc.eigvals[:, 0] < updated_eigval_bounds[0])
                             | (dc.eigvals[:, 0] > updated_eigval_bounds[1]))
            dc.loss[out_of_bounds] = 0.0

            n_out = out_of_bounds.sum().item()
            n_total = out_of_bounds.numel()
            print('%i / %i = %.1f %% out of bounds (new).'
                  % (n_out, n_total, 100 * n_out / n_total))

        if depth_bounds is not None:
            assert len(depth_bounds) == 2
            assert depth_bounds[0] <= depth_bounds[1]
            out_of_bounds = ((dc.depth[:, 0] < depth_bounds[0])
                             | (dc.depth[:, 0] > depth_bounds[1]))
            dc.loss[out_of_bounds] = 0.0

            n_out = out_of_bounds.sum().item()
            n_total = out_of_bounds.numel()
            print('%i / %i = %.1f %% depth out of bounds.'
                  % (n_out, n_total, 100 * n_out / n_total))

        dc.loss = torch.relu(dc.loss)
        loss = reduce(dc.loss, reduction=reduction)
        return loss, dc

    invalid = torch.tensor(invalid)
    # Serial eigvals.
    fun = lambda p, q: torch.linalg.eigvalsh(torch.cov(p.transpose(-1, -2)))[0] if p.shape[0] >= 3 else invalid
    loss = neighbor_fun(cloud, fun, query=query, k=k, r=r)
    loss = torch.stack(loss)
    loss = reduce(loss, reduction=reduction)

    # Parallelize eigvals.
    # cov = neighbor_cov(points, query=query, k=k, r=r)
    # loss = torch.linalg.eigvalsh(cov)[..., 0]
    # loss = reduce(loss, reduction=reduction)
    return loss


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
    # ds = Dataset(dataset_names[0])
    ds = Dataset('apartment')
    # ds = Dataset('eth')
    # ds = Dataset('gazebo_summer')
    r = 0.15
    for id in ds.ids[::10]:
        t = timer()
        cloud = ds.local_cloud(id)
        print('min:', cloud.min(axis=0))
        print('max:', cloud.max(axis=0))
        pose = torch.tensor(ds.cloud_pose(id))
        dc = DepthCloud.from_points(cloud)
        print('%i points read from dataset %s, cloud %i (%.3f s).'
              % (dc.size(), ds.name, id, timer() - t))

        t = timer()
        grid_res = 0.05
        dc = filter_grid(dc, grid_res, keep='last')
        print('%i points kept by grid filter with res. %.2f m (%.3f s).'
              % (dc.size(), grid_res, timer() - t))

        dc = dc.transform(pose)
        dc.update_all(r=r)
        dc.visualize(colors='inc_angles')
        # dc.visualize(colors='min_eigval')

        clouds.append(dc)
        poses.append(pose)

    combined = DepthCloud.concatenate(clouds, True)
    # combined.update_neighbors(r=r)
    # combined.filter_neighbors_normal_angle(0.5)
    # combined.visualize(colors='inc_angles')
    # combined.visualize(colors='min_eigval')

    # copy = combined.copy()
    # copy.update_neighbors(r=r)
    # copy.filter_neighbors_normal_angle(1.0)
    # copy.visualize()
    eigval_bounds = (0.0, 0.05**2)
    depth_bounds = (1.0, 20.0)
    max_angle = np.radians(30.)
    loss, loss_dc = min_eigval_loss(combined, r=r, offset=True,
                                    input_eigval_bounds=eigval_bounds,
                                    updated_eigval_bounds=eigval_bounds,
                                    max_angle=max_angle,
                                    depth_bounds=depth_bounds)
    print('Loss: %.6g' % loss.item())
    loss_dc.visualize(colors='loss')


def main():
    demo()


if __name__ == '__main__':
    main()

from __future__ import absolute_import, division, print_function
from .depth_cloud import DepthCloud
from .filters import filter_grid
from .nearest_neighbors import nearest_neighbors
import torch
from timeit import default_timer as timer


__all__ = [
    'min_eigval_loss',
    'neighbor_cov',
    'neighbor_fun',
    'reduce',
    'trace_loss',
]


def reduce(x, reduction='mean'):
    assert reduction in ('none', 'mean', 'sum')

    if reduction == 'mean':
        x = x.mean()
    elif reduction == 'sum':
        x = x.sum()

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


def min_eigval_loss(cloud, query=None, k=None, r=None, offset=False, bounds=None, reduction='mean', invalid=0.):
    if isinstance(cloud, DepthCloud):
        dc = cloud.copy()
        dc.update_all(k=k, r=r)
        dc.loss = dc.eigvals[:, 0]

        if offset:
            assert cloud.eigvals is not None
            dc.loss = dc.loss - cloud.eigvals[:, 0]

        if bounds is not None:
            assert len(bounds) == 2
            assert bounds[0] <= bounds[1]
            out_of_bounds = (dc.eigvals[:, 0] < bounds[0]) | (dc.eigvals[:, 0] > bounds[1])
            dc.loss[out_of_bounds] = 0.0

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
    ds = Dataset('eth')
    for id in ds.ids[::10]:
        t = timer()
        cloud = ds.local_cloud(id)
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
        dc.update_all(r=0.15)
        # dc.visualize(colors='inc_angles')
        # dc.visualize(colors='min_eigval')

        clouds.append(dc)
        poses.append(pose)

    combined = DepthCloud.concatenate(clouds, True)
    combined.visualize(colors='inc_angles')
    combined.visualize(colors='min_eigval')

    loss, loss_dc = min_eigval_loss(combined, r=0.15, offset=True, bounds=(0.0, 0.05**2))
    print(loss)
    loss_dc.visualize(colors='loss')


def main():
    demo()


if __name__ == '__main__':
    main()

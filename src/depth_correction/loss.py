from __future__ import absolute_import, division, print_function
from .depth_cloud import DepthCloud
from .filters import filter_grid
from .nearest_neighbors import nearest_neighbors
import torch
from timeit import default_timer as timer


__all__ = [
    'min_eig_loss',
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


def min_eig_loss(points, query=None, k=None, r=None, reduction='mean'):
    # fun = lambda x: torch.linalg.eigvalsh(torch.cov(x.transpose(-1, -2)))[0]
    # loss = neighbor_fun(points, fun, query=query, k=k, r=r)
    # loss = torch.stack(loss)
    # loss = reduce(loss, reduction=reduction)

    # Parallelize eigvals.
    cov = neighbor_cov(points, query=query, k=k, r=r)
    loss = torch.linalg.eigvalsh(cov)[..., 0]
    loss = reduce(loss, reduction=reduction)
    return loss


def trace_loss(points, query=None, k=None, r=None, reduction='mean'):
    fun = lambda p, q: torch.cov(p.transpose(-1, -2)).trace()
    loss = neighbor_fun(points, fun, query=query, k=k, r=r)
    loss = torch.stack(loss)
    loss = reduce(loss, reduction=reduction)
    return loss


def map_colors(values, colormap, min_value=None, max_value=None):
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)
    if not isinstance(colormap, torch.Tensor):
        colormap = torch.tensor(colormap, dtype=torch.float64)
    assert tuple(colormap.shape) == (2, 3)
    if min_value is None:
        min_value = values.min()
    if max_value is None:
        max_value = values.max()
    a = (values - min_value) / (max_value - min_value)
    # TODO: Allow full colormap with multiple colors.
    # num_colors = colormap.shape[0]
    # i0 = torch.floor(a * (num_colors - 1))
    # i0 = i1 + 1
    a = a.reshape([-1, 1])
    colors = (1 - a) * colormap[0:1] + a * colormap[1:]
    return colors


def show_cloud(cloud, colormap=None):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    o3d.visualization.draw_geometries([pcd])


def demo():
    from data.asl_laser import Dataset, dataset_names
    import open3d as o3d

    clouds = []
    poses = []
    # ds = Dataset('wood_summer')
    # ds = Dataset('eth')
    ds = Dataset('apartment')
    for id in ds.ids[::10]:
        t = timer()
        cloud = ds.local_cloud(id)
        n = cloud.shape[0]
        pose = ds.cloud_pose(id)
        print('%i points read from dataset %s, cloud %i (%.3f s).'
              % (n, ds.name, id, timer() - t))
        # show_cloud(cloud)

        t = timer()
        grid_res = 0.1
        cloud = filter_grid(cloud, grid_res, keep='last')
        print(cloud.shape)
        n = cloud.shape[0]
        print('%i points kept by grid filter with res. %.2f m (%.3f s).'
              % (n, grid_res, timer() - t))
        # show_cloud(cloud)

        cloud = torch.tensor(cloud, dtype=torch.float64)
        pose = torch.tensor(pose, dtype=torch.float64)

        clouds.append(cloud)
        poses.append(pose)

        # query = torch.index_select(cloud[::100], dim=0)
        # query = cloud[::100]
        query = cloud

        t = timer()
        loss = min_eig_loss(cloud, query, k=9, reduction='none')
        # loss = min_eig_loss(cloud, query, r=.15, reduction='none')
        # print(loss.shape)
        # print('Dataset %s, cloud %i: min eig loss (10 neighbors): %.6g (%.3f s).'
        #       % (ds.name, id, loss**.5, timer() - t))
        print('Dataset %s, cloud %i: min eig loss (10 neighbors): %.6g (%.3f s).'
              % (ds.name, id, loss.mean()**.5, timer() - t))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        colormap = torch.tensor([[0., 1., 0.], [1., 0., 0.]], dtype=torch.float64)
        min_value, max_value = torch.quantile(loss, torch.tensor([0., 0.99], dtype=torch.float64))
        print('min, max: %.6g, %.6g' % (min_value, max_value))
        colors = map_colors(loss, colormap, min_value=min_value, max_value=max_value)
        print(colors.shape)
        pcd.colors = o3d.utility.Vector3dVector(colors.detach().numpy())
        o3d.visualization.draw_geometries([pcd])

        # depth_cloud = DepthCloud.from_points(cloud)


def main():
    demo()


if __name__ == '__main__':
    main()

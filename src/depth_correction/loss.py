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


def min_eigval_loss(cloud, query=None, k=None, r=None, offset=False, reduction='mean', invalid=0.):
    if isinstance(cloud, DepthCloud):
        dc = cloud.copy()
        dc.update_all(k=k, r=r)
        dc.loss = dc.eigvals[:, 0]
        if offset:
            assert cloud.eigvals is not None
            dc.loss = dc.loss - cloud.eigvals[:, 0]
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
    ds = Dataset(dataset_names[0])
    for id in ds.ids[::10]:
        t = timer()
        cloud = ds.local_cloud(id)
        pose = ds.cloud_pose(id)

        dc = DepthCloud.from_points(cloud)
        print('%i points read from dataset %s, cloud %i (%.3f s).'
              % (dc.size(), ds.name, id, timer() - t))

        t = timer()
        grid_res = 0.1
        dc = filter_grid(dc, grid_res, keep='last')
        print('%i points kept by grid filter with res. %.2f m (%.3f s).'
              % (dc.size(), grid_res, timer() - t))
        dc.visualize()

        dc.update_points()
        dc.update_neighbors(r=0.25)
        dc.update_cov()
        dc.update_eigvals()

        dc.visualize(colors='min_eigval')

        loss, loss_dc = min_eigval_loss(dc, r=0.25)
        print(loss)
        loss_dc.visualize(colors='loss')

        loss, loss_dc = min_eigval_loss(dc, r=0.25, offset=True)
        print(loss)
        loss_dc.visualize(colors='loss')

        # t = timer()
        # loss = min_eig_loss(cloud, query, k=9, reduction='none')
        # loss = min_eig_loss(cloud, query, r=.25, reduction='none')
        # loss = trace_loss(cloud, query, r=.25, reduction='none')
        # print(loss.shape)
        # print('Dataset %s, cloud %i: min eig loss (10 neighbors): %.6g (%.3f s).'
        #       % (ds.name, id, loss**.5, timer() - t))
        # print('Dataset %s, cloud %i: min eig loss (10 neighbors): %.6g (%.3f s).'
        #       % (ds.name, id, loss.mean()**.5, timer() - t))

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(cloud)
        # colormap = torch.tensor([[0., 1., 0.], [1., 0., 0.]], dtype=torch.float64)
        # min_value, max_value = torch.quantile(loss, torch.tensor([0., 0.99], dtype=torch.float64))
        # print('min, max: %.6g, %.6g' % (min_value, max_value))
        # colors = map_colors(loss, colormap, min_value=min_value, max_value=max_value)
        # print(colors.shape)
        # pcd.colors = o3d.utility.Vector3dVector(colors.detach().numpy())
        # o3d.visualization.draw_geometries([pcd])

        # depth_cloud = DepthCloud.from_points(cloud)


def main():
    demo()


if __name__ == '__main__':
    main()

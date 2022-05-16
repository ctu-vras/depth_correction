from __future__ import absolute_import, division, print_function
from .config import Config
from .depth_cloud import DepthCloud
from .filters import filter_eigenvalue, filter_depth, filter_grid
from .nearest_neighbors import nearest_neighbors
from .utils import timing, trace
from enum import Enum
import numpy as np
from numpy.polynomial import Polynomial
import torch

__all__ = [
    'batch_loss',
    'create_loss',
    'loss_by_name',
    'min_eigval_loss',
    'neighbor_cov',
    'neighbor_fun',
    'reduce',
    'trace_loss',
]


class Reduction(Enum):
    NONE = 'none'
    MEAN = 'mean'
    SUM = 'sum'


def reduce(x, reduction=Reduction.MEAN, weights=None, only_finite=False, skip_nans=False):
    # assert reduction in ('none', 'mean', 'sum')
    assert reduction in Reduction

    keep = None
    if only_finite:
        keep = ~x.isfinite()
    elif skip_nans:
        keep = ~x.isnan()
    if keep is not None:
        if weights:
            weights = weights[keep]
        x = x[keep]

    if reduction == Reduction.MEAN:
        if weights is None:
            x = x.mean()
        else:
            x = (weights * x).sum() / weights.sum()
    elif reduction == Reduction.SUM:
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


def batch_loss(loss_fun, clouds, mask=None, offset=None, sqrt=False, reduction=Reduction.MEAN):
    """General batch loss of a sequence of clouds.

    :param loss_fun: Loss function.
    :param clouds: Sequence of clouds, optional.
    :param mask: Sequence of masks, optional.
    :param offset: Source cloud to offset point-wise loss values, optional.
    :param sqrt: Whether to use square root of point-wise losses.
    :param reduction: Loss reduction mode.
    :return: Reduced loss and loss clouds.
    """
    assert callable(loss_fun)
    assert isinstance(clouds, (list, tuple))
    assert mask is None or isinstance(mask, (list, tuple))
    assert offset is None or isinstance(offset, (list, tuple))

    losses, loss_clouds = [], []
    for i in range(len(clouds)):
        c = clouds[i]
        m = None if mask is None else mask[i]
        o = None if offset is None else offset[i]
        loss, loss_cloud = loss_fun(c, mask=m, offset=o, sqrt=sqrt, reduction=Reduction.NONE)
        losses.append(loss)
        loss_clouds.append(loss_cloud)
    # Double reduction (average of averages)
    # loss = reduce(torch.cat(losses), reduction=reduction)
    # Single point-wise reduction (average)
    loss = reduce(torch.cat(losses), reduction=reduction)
    return loss, loss_clouds


def min_eigval_loss(cloud, mask=None, offset=None, sqrt=False, reduction=Reduction.MEAN):
    """Map consistency loss based on the smallest eigenvalue.

    Pre-filter cloud before, or set the mask to select points to be used in
    loss reduction. In general, surfaces for which incidence angles can be
    reliably estimated should be selected, typically planar regions.

    :param cloud:
    :param mask:
    :param offset: Source cloud to offset point-wise loss values, optional.
    :param sqrt: Whether to use square root of eigenvalue.
    :param reduction:
    :return:
    """
    # If a batch of clouds is (as a list), process them separately,
    # and reduce point-wise loss in the end by delegating to batch_loss.
    if isinstance(cloud, (list, tuple)):
        return batch_loss(min_eigval_loss, cloud, mask=mask, offset=offset, sqrt=sqrt, reduction=reduction)

    assert isinstance(cloud, (DepthCloud, list, tuple))

    assert offset is None or isinstance(offset, DepthCloud)
    # assert eigenvalue_bounds is None or len(eigenvalue_bounds) == 2

    assert isinstance(cloud, DepthCloud)
    assert cloud.eigvals is not None

    if mask is None:
        loss = cloud.eigvals[:, 0]
    else:
        assert isinstance(mask, torch.Tensor)
        loss = cloud.eigvals[mask, 0]

    if offset is not None:
        assert isinstance(offset, DepthCloud)
        assert offset.eigvals is not None
        # Offset the loss using trace from the offset cloud.
        if mask is None:
            loss = loss - offset.eigvals[:, 0]
        else:
            loss = loss - offset.eigvals[mask, 0]

    # Ensure positive loss.
    loss = torch.relu(loss)

    if sqrt:
        loss = torch.sqrt(loss)

    if mask is None:
        cloud = cloud.copy()
    else:
        cloud = cloud[mask]
    cloud.loss = loss

    loss = reduce(loss, reduction=reduction)
    return loss, cloud


def trace_loss(cloud, mask=None, offset=None, sqrt=None, reduction=Reduction.MEAN):
    """Map consistency loss based on the trace of covariance matrix.

    Pre-filter cloud before, or set the mask to select points to be used in
    loss reduction. In general, surfaces for which incidence angles can be
    reliably estimated should be selected, typically planar regions.

    :param cloud:
    :param mask:
    :param offset: Source cloud to offset point-wise loss values, optional.
    :param sqrt: Whether to use square root of trace.
    :param reduction:
    :return:
    """
    # If a batch of clouds is (as a list), process them separately,
    # and reduce point-wise loss in the end by delegating to batch_loss.
    if isinstance(cloud, (list, tuple)):
        return batch_loss(trace_loss, cloud, mask=mask, offset=offset, sqrt=sqrt, reduction=reduction)

    assert isinstance(cloud, DepthCloud)
    assert cloud.cov is not None

    if mask is None:
        loss = trace(cloud.cov)
    else:
        assert isinstance(mask, torch.Tensor)
        loss = trace(cloud.cov[mask])

    if offset is not None:
        assert isinstance(offset, DepthCloud)
        assert offset.cov is not None
        # Offset the loss using trace from the offset cloud.
        if mask is None:
            loss = loss - trace(offset.cov)
        else:
            loss = loss - trace(offset.cov[mask])

    # Ensure positive loss.
    loss = torch.relu(loss)

    if sqrt:
        loss = torch.sqrt(loss)

    if mask is None:
        cloud = cloud.copy()
    else:
        cloud = cloud[mask]
    cloud.loss = loss

    loss = reduce(loss, reduction=reduction)
    return loss, cloud


def loss_by_name(name):
    assert name in ('min_eigval_loss', 'trace_loss')
    return globals()[name]


def create_loss(cfg: Config):
    loss = loss_by_name(cfg.loss)

    def loss_fun(*args, **kwargs):
        return loss(*args, **kwargs, **cfg.loss_kwargs)

    return loss_fun


def preprocess_cloud(cloud, min_depth=None, max_depth=None, grid_res=None, k=None, r=None):
    cloud = filter_depth(cloud, min=min_depth, max=max_depth, log=False)
    cloud = filter_grid(cloud, grid_res, keep='last')
    cloud.update_all(k=k, r=r)
    keep = filter_eigenvalue(cloud, 0, max=(grid_res / 5)**2, only_mask=True, log=False)
    keep = keep & filter_eigenvalue(cloud, 1, min=grid_res**2, only_mask=True, log=False)
    cloud = cloud[keep]
    cloud.update_all(k=k, r=r)
    return cloud


def dataset_to_cloud(ds, min_depth=None, max_depth=None, grid_res=None, k=None, r=None, device='cpu'):
    if isinstance(device, str):
        device = torch.device(device)
    clouds = []
    poses = []

    for cloud, pose in ds:
        cloud = DepthCloud.from_points(cloud)
        cloud.to(device)
        pose = torch.tensor(pose, device=device)
        cloud = cloud.transform(pose)
        cloud = preprocess_cloud(cloud, min_depth=min_depth, max_depth=max_depth, grid_res=grid_res, k=k, r=r)
        clouds.append(cloud)
        poses.append(pose)

    cloud = DepthCloud.concatenate(clouds, True)
    # cloud.visualize(colors='inc_angles')
    cloud.visualize(colors='z')
    cloud.update_all(k=k, r=r)
    return cloud


def l2_loss(dc1, dc2):
    assert dc1.points is not None
    assert dc2.points is not None
    assert len(dc1.points) == len(dc2.points)
    return torch.linalg.norm(dc1.points - dc2.points) / len(dc1.points)


def demo():
    from data.asl_laser import Dataset
    # ds = Dataset('apartment')
    ds = Dataset('eth')
    # ds = Dataset('gazebo_summer')
    # ds = Dataset('gazebo_winter')
    # ds = Dataset('stairs')
    # ds = ds[10:21:10]
    ds = ds[::5]

    min_depth = 1.0
    max_depth = 15.0
    grid_res = 0.05
    k = None
    r = 3 * grid_res
    device = torch.device('cpu')
    # device = torch.device('cuda')

    dc = dataset_to_cloud(ds, min_depth=min_depth, max_depth=max_depth, grid_res=grid_res, k=k, r=r,
                          device=device)

    # Visualize incidence angle to plane distance.
    # TODO: Compare using plane fit for low incidence angle.
    depth = dc.depth.detach().numpy().ravel()
    inc = dc.inc_angles.detach().numpy().ravel()
    inv_cos = 1.0 / np.cos(inc)
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
        poly1 = Polynomial.fit(x, y, 1).convert()
        poly2 = Polynomial.fit(x, y, 2).convert()
        print('%s to %s (deg. 1 fit): %s' % (y_label, x_label, poly1))
        print('%s to %s (deg. 2 fit): %s' % (y_label, x_label, poly2))
        # xs = domain(poly1)
        xs = domain(x)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
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

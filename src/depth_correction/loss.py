from __future__ import absolute_import, division, print_function
from .config import Config
from .depth_cloud import DepthCloud
from .filters import filter_eigenvalue, filter_depth, filter_grid
from .nearest_neighbors import nearest_neighbors
from .point_cloud import PointCloud
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


def eigh3_deledalle(mat, normalize=True, sort=True):
    # https://hal.archives-ouvertes.fr/hal-01501221/document
    # (a, d, f), (_, b, e), (_, _, c) = mat
    # a, b, ..., f are N-dimensional tensors.
    a = mat[..., 0, 0]
    b = mat[..., 1, 1]
    c = mat[..., 2, 2]
    d = mat[..., 0, 1]
    e = mat[..., 1, 2]
    f = mat[..., 0, 2]

    # (8)
    # N-dimensional tensors
    x1 = a**2 + b**2 + c**2 - a * b - a * c - b * c + 3 * (d**2 + f**2 + e**2)
    x2 = -(2 * a - b - c) * (2 * b - a - c) * (2 * c - a - b) \
            + 9 * ((2 * c - a - b) * d**2 + (2 * b - a - c) * f**2 + (2 * a - b - c) * e**2) \
            - 54 * (d * e * f)
    
    # (9)
    # TODO: Convert to mask for individual indices.
    # if x2 > 0:
    #     phi = torch.atan(torch.sqrt(4 * x1**3 - x2**2) / x2)
    # elif x2 == 0:
    #     phi = torch.pi / 2
    # elif x2 < 0:
    #     phi = torch.atan(torch.sqrt(4 * x1**3 - x2**2) / x2) + torch.pi
    # # Use 4-quadrant arctan2 to avoid switch.
    # N-dimensional tensor
    phi = torch.atan2(torch.sqrt(4 * x1**3 - x2**2), x2)
    
    # (7) Eigenvalues
    # N-dimensional tensors
    eigval1 = (a + b + c - 2 * torch.sqrt(x1) * torch.cos(phi / 3)) / 3
    eigval2 = (a + b + c + 2 * torch.sqrt(x1) * torch.cos((phi - torch.pi) / 3)) / 3
    eigval3 = (a + b + c + 2 * torch.sqrt(x1) * torch.cos((phi + torch.pi) / 3)) / 3
    
    # (11)
    # TODO: Handle zero denominators.
    # N-dimensional tensors
    m1 = (d * (c - eigval1) - e * f) / (f * (b - eigval1) - d * e)
    m2 = (d * (c - eigval2) - e * f) / (f * (b - eigval2) - d * e)
    m3 = (d * (c - eigval3) - e * f) / (f * (b - eigval3) - d * e)

    # (10) Eigenvectors
    # N-by-3 tensors
    eigvec1 = torch.stack([(eigval1 - c - e * m1) / f, m1, torch.ones_like(m1)], dim=-1)
    eigvec2 = torch.stack([(eigval2 - c - e * m2) / f, m2, torch.ones_like(m2)], dim=-1)
    eigvec3 = torch.stack([(eigval3 - c - e * m3) / f, m3, torch.ones_like(m3)], dim=-1)
    
    # Stack eigenvalues and vectors to tensors.
    # N-by-3
    eigvals = torch.stack([eigval1, eigval2, eigval3], dim=-1)
     # N-by-3-by-3
    eigvecs = torch.stack([eigvec1, eigvec2, eigvec3], dim=-1)

    # Normalize eigenvectors.
    if normalize:
        eigvecs = eigvecs / torch.linalg.norm(eigvecs, dim=1, keepdim=True)

    # Sort eigenvalues and eigenvectors.
    if sort:
        eigvals, ind = torch.sort(eigvals, dim=-1)
        eigvecs = torch.gather(eigvecs, 2, ind.unsqueeze(1).expand(eigvecs.shape))
    
    return eigvals, eigvecs


def eigh3(mat):
    """Analytic eigenvalue decomposition of 3-by-3 symmetric matrices.

    Symmetric matrices have real eigenvalues and orthogonal eigenvectors.
    For real symmetric matrices, the eigenvectors are also real.

    Matrix A = U * diag(L) * U.T,
    where U is the matrix of eigenvectors and L is the vector of eigenvalues.

    :param mat: ...-by-3-by-3 3D covariance matrices.
    :return: Eigenvalues and eigenvectors.
    """
    assert isinstance(mat, torch.Tensor)
    assert mat.shape[-2] == mat.shape[-1]
    assert mat.shape[-1] == 3

    return eigh3_deledalle(mat)


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


def batch_loss(loss_fun, clouds, masks=None, offsets=None, reduction=Reduction.MEAN, **kwargs):
    """General batch loss of a sequence of clouds.

    :param loss_fun: Loss function.
    :param clouds: Sequence of clouds.
    :param masks: Sequence of masks, optional.
    :param offsets: Sequences of offset clouds, optional.
    :param reduction: Loss reduction mode.
    :param kwargs: Other key-value loss arguments.
    :return: Reduced loss and loss clouds.
    """
    assert callable(loss_fun)
    assert isinstance(clouds, (list, tuple))
    if masks is None:
        masks = len(clouds) * [None]
    if offsets is None:
        offsets = len(clouds) * [None]
    assert isinstance(masks, (list, tuple))
    assert len(masks) == len(clouds)
    assert isinstance(offsets, (list, tuple))
    assert len(offsets) == len(clouds)

    losses, loss_clouds = [], []
    for cloud, mask, offset in zip(clouds, masks, offsets):
        loss, loss_cloud = loss_fun(cloud, mask=mask, offset=offset, reduction=Reduction.NONE, **kwargs)
        losses.append(loss)
        loss_clouds.append(loss_cloud)
    # Double reduction (average of averages)
    # loss = reduce(torch.cat(losses), reduction=reduction)
    # Single point-wise reduction (average)
    loss = reduce(torch.cat(losses), reduction=reduction)
    return loss, loss_clouds


def min_eigval_loss(cloud, mask=None, offset=None, sqrt=False, normalization=False, reduction=Reduction.MEAN,
                    inlier_max_loss=None, inlier_ratio=1.0, inlier_loss_mult=1.0):
    """Map consistency loss based on the smallest eigenvalue.

    Pre-filter cloud before, or set the mask to select points to be used in
    loss reduction. In general, surfaces for which incidence angles can be
    reliably estimated should be selected, typically planar regions.

    :param cloud:
    :param mask:
    :param offset: Offset point-wise loss values, optional.
    :param sqrt: Whether to use square root of eigenvalue.
    :param normalization: Whether to normalize minimum eigenvalue by total variance.
    :param reduction:
    :return:
    """
    # If a batch of clouds is (as a list), process them separately,
    # and reduce point-wise loss in the end by delegating to batch_loss.
    if isinstance(cloud, (list, tuple)):
        return batch_loss(min_eigval_loss, cloud, masks=mask, offsets=offset, sqrt=sqrt, normalization=normalization,
                          reduction=reduction,
                          inlier_max_loss=inlier_max_loss, inlier_ratio=inlier_ratio, inlier_loss_mult=inlier_loss_mult)

    assert isinstance(cloud, (DepthCloud, PointCloud))
    assert cloud.eigvals is not None
    assert offset is None or isinstance(offset, (DepthCloud, PointCloud))

    if mask is not None:
        print('Using %.3f valid entries from input cloud.' % mask.float().mean())
        cloud = cloud[mask]
        mask = None

    eigvals = cloud.eigvals
    loss = eigvals[:, 0]

    if normalization:
        loss = loss / eigvals.sum(dim=-1).clamp(min=1e-6)

    if inlier_ratio < 1.0:
        assert offset is None
        # Sort loss values and select inlier_ratio of them.
        loss_quantile = torch.quantile(loss, inlier_ratio, dim=0)
        print('Loss %.3g-quantile: %.3g.' % (inlier_ratio, loss_quantile.item()))
        if inlier_loss_mult != 1.0:
            loss_quantile = inlier_loss_mult * loss_quantile
        print('Multiplied %.3g-quantile: %.3g.' % (inlier_ratio, loss_quantile.item()))
        if inlier_max_loss is None:
            inlier_max_loss = loss_quantile
        else:
            inlier_max_loss = torch.min(inlier_max_loss, loss_quantile)

    if inlier_max_loss is not None:
        assert offset is None
        mask = (loss <= inlier_max_loss)
        print('Using %i (%.3g) inliers with loss <= %.3g.'
              % (mask.sum().item(), mask.float().mean().item(), inlier_max_loss.item()))

    if mask is not None:
        cloud = cloud[mask]
        loss = loss[mask]

    # Offset the loss using loss computed on local clouds.
    if offset is not None:
        loss = loss - offset

    # Ensure positive loss.
    loss = torch.relu(loss)

    if sqrt:
        loss = torch.sqrt(loss)

    cloud = cloud.copy()
    cloud.loss = loss

    loss = reduce(loss, reduction=reduction)
    return loss, cloud


def trace_loss(cloud, mask=None, offset=None, sqrt=None, reduction=Reduction.MEAN,
               inlier_max_loss=None, inlier_ratio=1.0, inlier_loss_mult=1.0,
               **kwargs):
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
        return batch_loss(trace_loss, cloud, masks=mask, offsets=offset, sqrt=sqrt, reduction=reduction,
                          inlier_max_loss=inlier_max_loss, inlier_ratio=inlier_ratio, inlier_loss_mult=inlier_loss_mult)

    assert isinstance(cloud, (DepthCloud, PointCloud))
    assert cloud.cov is not None
    assert offset is None or isinstance(offset, (DepthCloud, PointCloud))

    if mask is not None:
        print('Using %.3f valid entries from input cloud.' % mask.float().mean())
        cloud = cloud[mask]
        mask = None

    cov = cloud.cov
    loss = trace(cov)

    if inlier_ratio < 1.0:
        assert offset is None
        # Sort loss values and select inlier_ratio of them.
        loss_quantile = torch.quantile(loss, inlier_ratio, dim=0)
        print('Loss %.3g-quantile: %.3g.' % (inlier_ratio, loss_quantile.item()))
        if inlier_loss_mult != 1.0:
            loss_quantile = inlier_loss_mult * loss_quantile
        print('Multiplied %.3g-quantile: %.3g.' % (inlier_ratio, loss_quantile.item()))
        if inlier_max_loss is None:
            inlier_max_loss = loss_quantile
        else:
            inlier_max_loss = torch.min(inlier_max_loss, loss_quantile)

    if inlier_max_loss is not None:
        assert offset is None
        mask = (loss <= inlier_max_loss)
        print('Using %i (%.3g) inliers with loss <= %.3g.'
              % (mask.sum().item(), mask.float().mean().item(), inlier_max_loss.item()))

    if mask is not None:
        cloud = cloud[mask]
        loss = loss[mask]

    # Offset the loss using loss computed on local clouds.
    if offset is not None:
        loss = loss - offset

    # Ensure positive loss.
    loss = torch.relu(loss)

    if sqrt:
        loss = torch.sqrt(loss)

    cloud = cloud.copy()
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


def test_eigh3():
    
    def rand_C3():
        x = torch.randn(3, 3)
        return x @ x.t()
    
    n = 2
    C = torch.stack([rand_C3() for _ in range(n)])

    eigvals_torch, eigvecs_torch = torch.linalg.eigh(C)
    # print('eigvals_torch:\n', eigvals_torch)
    # print('eigvecs_torch:\n', eigvecs_torch)
        
    eigvals, eigvecs = eigh3(C)
    # print('eigvals:\n', eigvals)
    # print('eigvecs:\n', eigvecs)

    assert torch.allclose(eigvals, eigvals_torch, atol=1e-6), \
            (eigvals, eigvals_torch, eigvals - eigvals_torch)
    assert torch.all(torch.isclose(eigvecs, eigvecs_torch, atol=1e-5)
                     | torch.isclose(-eigvecs, eigvecs_torch, atol=1e-5)), \
            (eigvecs, eigvecs_torch, eigvecs - eigvecs_torch)
        

def test():
    test_eigh3()


def main():
    # test()
    demo()


if __name__ == '__main__':
    main()

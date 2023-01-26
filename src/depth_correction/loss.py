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
from scipy.spatial import cKDTree
from pytorch3d.ops.knn import knn_points
import warnings


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
        keep = x.isfinite()
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


def batch_loss(loss_fun, clouds, masks=None, offsets=None, reduction=Reduction.MEAN,
               only_finite=False, skip_nans=False,
               **kwargs):
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

    loss = reduce(torch.cat(losses), reduction=reduction,
                  only_finite=only_finite, skip_nans=skip_nans)
    return loss, loss_clouds


def min_eigval_loss(cloud, mask=None, offset=None, sqrt=False, normalization=False, reduction=Reduction.MEAN,
                    inlier_max_loss=None, inlier_ratio=1.0, inlier_loss_mult=1.0,
                    only_finite=False, skip_nans=False, **kwargs):
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
                          inlier_max_loss=inlier_max_loss, inlier_ratio=inlier_ratio, inlier_loss_mult=inlier_loss_mult,
                          only_finite=only_finite, skip_nans=skip_nans)

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

    loss = reduce(loss, reduction=reduction,
                  only_finite=only_finite, skip_nans=skip_nans)
    return loss, cloud


def trace_loss(cloud, mask=None, offset=None, sqrt=None, reduction=Reduction.MEAN,
               inlier_max_loss=None, inlier_ratio=1.0, inlier_loss_mult=1.0,
               only_finite=False, skip_nans=False,
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
                          inlier_max_loss=inlier_max_loss, inlier_ratio=inlier_ratio, inlier_loss_mult=inlier_loss_mult,
                          only_finite=only_finite, skip_nans=skip_nans)

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

    loss = reduce(loss, reduction=reduction, only_finite=only_finite, skip_nans=skip_nans)
    return loss, cloud


def icp_loss(clouds, poses=None, model=None, masks=None, **kwargs):
    """ICP-like point to plane loss.

    :param clouds: List of lists of clouds :) Individual scans from different data sequences.
    :param poses: List od lists of poses for each point cloud scan.
    :param masks:
    :return:
    """
    transformed_clouds = clouds
    if model is not None:
        transformed_clouds = [[model(c) for c in seq_clouds] for seq_clouds in transformed_clouds]
    if poses is not None:
        transformed_clouds = [[c.transform(p) for c, p in zip(seq_clouds, seq_poses)]
                              for seq_clouds, seq_poses in zip(transformed_clouds, poses)]
    loss = 0.
    loss_cloud = []
    loss_fun = point_to_plane_dist if kwargs['icp_point_to_plane'] else point_to_point_dist

    for i in range(len(transformed_clouds)):
        seq_trans_clouds = transformed_clouds[i]
        seq_masks = None if masks is None else masks[i]
        loss_seq = loss_fun(seq_trans_clouds, masks=seq_masks, inlier_ratio=kwargs['icp_inlier_ratio'])
        loss = loss + loss_seq

        cloud = DepthCloud.concatenate(seq_trans_clouds)
        cloud.loss = loss
        loss_cloud.append(cloud)

    loss = loss / len(transformed_clouds)

    return loss, loss_cloud


def point_to_plane_dist(clouds: list, inlier_ratio=0.5, masks=None, differentiable=True, verbose=False):
    """ICP-like point to plane distance.

    Computes point to plane distances for consecutive pairs of point cloud scans, and returns the average value.

    :param clouds: List of clouds. Individual scans from a data sequences.
    :param masks: List of tuples masks[i] = (mask1, mask2) where mask1 defines indices of points from 1st point cloud
                  in a pair that intersect (close enough) with points from 2nd cloud in the pair,
                  mask2 is list of indices of intersection points from the 2nd point cloud in a pair.
    :param inlier_ratio: Ratio of inlier points between a two pairs of neighboring clouds.
    :param differentiable: Whether to use differentiable method of finding neighboring points (from Pytorch3d: slow on CPU)
                           or from scipy (faster but not differentiable).
    :param verbose:
    :return:
    """
    assert 0.0 <= inlier_ratio <= 1.0
    if masks is not None:
        assert len(clouds) == len(masks) + 1
        # print('Using precomputed intersection masks for point to plane loss')
    point2plane_dist = 0.0
    n_pairs = len(clouds) - 1
    for i in range(n_pairs):
        cloud1 = clouds[i]
        assert cloud1.normals is not None, "Cloud must have normals computed to estimate point to plane distance"
        cloud2 = clouds[i + 1]

        points1 = cloud1.to_points() if cloud1.points is None else cloud1.points
        points2 = cloud2.to_points() if cloud2.points is None else cloud2.points
        assert not torch.all(torch.isnan(points1))
        assert not torch.all(torch.isnan(points2))
        points1 = torch.as_tensor(points1, dtype=torch.float)
        points2 = torch.as_tensor(points2, dtype=torch.float)

        # find intersections between neighboring point clouds (1 and 2)
        if masks is None:
            if not differentiable:
                tree = cKDTree(points2)
                dists, ids = tree.query(points1, k=1)
            else:
                dists, ids, _ = knn_points(points1[None], points2[None], K=1)
                dists = torch.sqrt(dists).squeeze()
                ids = ids.squeeze()
            dist_th = torch.quantile(dists[~torch.isnan(dists)], inlier_ratio)
            mask1 = dists <= dist_th
            mask2 = ids[mask1]
            inl_err = dists[mask1].mean()
        else:
            mask1, mask2 = masks[i]
            inl_err = torch.tensor(-1.0)

        points1_inters = points1[mask1]
        assert len(points1_inters) > 0, "Point clouds do not intersect. Try to sample lidar scans more frequently"
        points2_inters = points2[mask2]

        # point to plane distance 1 -> 2
        normals1_inters = cloud1.normals[mask1]
        # assert np.allclose(np.linalg.norm(normals1_inters, axis=1), np.ones(len(normals1_inters)))
        vectors = points2_inters - points1_inters
        normals = normals1_inters
        dists_to_plane = torch.multiply(vectors, normals).sum(dim=1).abs()
        dist12 = dists_to_plane.mean()

        # point to plane distance 2 -> 1
        normals2_inters = cloud2.normals[mask2]
        # assert np.allclose(np.linalg.norm(normals2_inters, axis=1), np.ones(len(normals2_inters)))
        vectors = points1_inters - points2_inters
        normals = normals2_inters
        dists_to_plane = torch.multiply(vectors, normals).sum(dim=1).abs()
        dist21 = dists_to_plane.mean()

        point2plane_dist += 0.5 * (dist12 + dist21)

        if inl_err > 0.3:
            warnings.warn('ICP inliers error is too big: %.3f (> 0.3) [m] for pairs (%i, %i)' % (inl_err, i, i + 1))

        if verbose:
            print('Mean point to plane distance: %.3f [m] for scans: (%i, %i), inliers error: %.6f' %
                  (point2plane_dist.item(), i, i+1, inl_err.item()))

    point2plane_dist = torch.as_tensor(point2plane_dist / n_pairs)

    return point2plane_dist


def point_to_point_dist(clouds: list, inlier_ratio=0.5, masks=None, differentiable=True, verbose=False):
    """ICP-like point to point distance.

    Computes point to point distances for consecutive pairs of point cloud scans, and returns the average value.

    :param clouds: List of clouds. Individual scans from a data sequences.
    :param masks: List of tuples masks[i] = (mask1, mask2) where mask1 defines indices of points from 1st point cloud
                  in a pair that intersect (close enough) with points from 2nd cloud in the pair,
                  mask2 is list of indices of intersection points from the 2nd point cloud in a pair.
    :param inlier_ratio: Ratio of inlier points between a two pairs of neighboring clouds.
    :param differentiable: Whether to use differentiable method of finding neighboring points (from Pytorch3d: slow on CPU)
                           or from scipy (faster but not differentiable).
    :param verbose:
    :return:
    """
    assert 0.0 <= inlier_ratio <= 1.0
    if masks is not None:
        assert len(clouds) == len(masks) + 1
        # print('Using precomputed intersection masks for point to plane loss')
    point2point_dist = 0.0
    n_pairs = len(clouds) - 1
    for i in range(n_pairs):
        cloud1 = clouds[i]
        cloud2 = clouds[i + 1]

        points1 = cloud1.to_points() if cloud1.points is None else cloud1.points
        points2 = cloud2.to_points() if cloud2.points is None else cloud2.points
        assert not torch.all(torch.isnan(points1))
        assert not torch.all(torch.isnan(points2))
        points1 = torch.as_tensor(points1, dtype=torch.float)
        points2 = torch.as_tensor(points2, dtype=torch.float)

        # find intersections between neighboring point clouds (1 and 2)
        if masks is None:
            if not differentiable:
                tree = cKDTree(points2)
                dists, ids = tree.query(points1, k=1)
            else:
                dists, ids, _ = knn_points(points1[None], points2[None], K=1)
                dists = torch.sqrt(dists).squeeze()
                ids = ids.squeeze()
            dist_th = torch.quantile(dists[~torch.isnan(dists)], inlier_ratio)
            mask1 = dists <= dist_th
            mask2 = ids[mask1]
            inl_err = dists[mask1].mean()
        else:
            mask1, mask2 = masks[i]
            inl_err = torch.tensor(-1.0)

        points1_inters = points1[mask1]
        assert len(points1_inters) > 0, "Point clouds do not intersect. Try to sample lidar scans more frequently"
        points2_inters = points2[mask2]
        assert len(points2_inters) > 0, "Point clouds do not intersect. Try to sample lidar scans more frequently"

        # point to point distance
        vectors = points2_inters - points1_inters
        point2point_dist = torch.linalg.norm(vectors, dim=1).mean()

        if inl_err > 0.3:
            warnings.warn('ICP inliers error is too big: %.3f (> 0.3) [m] for pairs (%i, %i)' % (inl_err, i, i + 1))

        if verbose:
            print('Mean point to plane distance: %.3f [m] for scans: (%i, %i), inliers error: %.6f' %
                  (point2point_dist.item(), i, i+1, inl_err.item()))

    point2point_dist = torch.as_tensor(point2point_dist / n_pairs)

    return point2point_dist


def loss_by_name(name):
    assert name in ('min_eigval_loss', 'trace_loss', 'icp_loss')
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
        cloud = preprocess_cloud(cloud, min_depth=min_depth, max_depth=max_depth, grid_res=grid_res, k=k, r=r)
        cloud = cloud.transform(pose)
        clouds.append(cloud)
        poses.append(pose)

    cloud = DepthCloud.concatenate(clouds)
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


def pose_correction_demo():
    from data.fee_corridor import dataset_names, Dataset
    from depth_correction.preproc import filtered_cloud, local_feature_cloud
    from depth_correction.transform import matrix_to_xyz_axis_angle, xyz_axis_angle_to_matrix
    import open3d as o3d
    from matplotlib import pyplot as plt

    cfg = Config()
    cfg.grid_res = 0.2
    cfg.min_depth = 1.0
    cfg.max_depth = 20.0
    cfg.nn_r = 0.4
    cfg.device = 'cuda'
    cfg.loss_kwargs['icp_inliers_ratio'] = 0.5
    cfg.loss_kwargs['icp_point_to_plane'] = False

    ds = Dataset(name=dataset_names[0])
    id = int(np.random.choice(range(len(ds) - 1)))
    print('Using a pair of scans (%i, %i) from sequence: %s' % (id, id+1, dataset_names[0]))
    points1, pose1 = ds[id]
    points2, pose2 = ds[id + 1]
    # points2, pose2 = ds[id]

    cloud1 = DepthCloud.from_structured_array(points1)
    cloud2 = DepthCloud.from_structured_array(points2)

    cloud1 = filtered_cloud(cloud1, cfg)
    cloud2 = filtered_cloud(cloud2, cfg)

    pose1 = torch.tensor(pose1, dtype=torch.float32)
    pose2 = torch.tensor(pose2, dtype=torch.float32)
    xyza1 = torch.tensor(matrix_to_xyz_axis_angle(pose1[None]), dtype=torch.float32).squeeze()

    xyza1_delta = torch.tensor([-0.01, 0.01, 0.0, 0.0, 0.0, 0.0], dtype=pose1.dtype)
    xyza1_delta.requires_grad = True

    optimizer = torch.optim.Adam([{'params': xyza1_delta, 'lr': 1e-3}])

    cloud2 = cloud2.transform(pose2)
    cloud2.update_points()

    # compute cloud features necessary for optimization (like normals and incidence angles
    cloud1 = local_feature_cloud(cloud1, cfg)
    cloud2 = local_feature_cloud(cloud2, cfg)

    cloud1 = cloud1[cloud1.mask]
    cloud2 = cloud2[cloud2.mask]

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(cloud2.points.detach())
    pcd2.colors = o3d.utility.Vector3dVector(torch.zeros_like(cloud2.points.detach()) + torch.tensor([0, 0, 1]))

    plt.figure(figsize=(20, 5))
    losses = []
    iters = []
    xyza_deltas = []
    # run optimization loop
    for it in range(1000):
        # add noise to poses
        xyza1_corr = xyza1 + xyza1_delta
        pose1_corr = xyz_axis_angle_to_matrix(xyza1_corr[None]).squeeze()

        # transform point clouds to the same world coordinate frame
        cloud1_corr = cloud1.transform(pose1_corr)
        cloud1_corr.update_points()

        train_clouds = [cloud1_corr, cloud2]

        loss, _ = icp_loss([train_clouds], **cfg.loss_kwargs, verbose=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('At iter %i ICP loss: %f' % (it, loss.item()))

        iters.append(it)
        losses.append(loss.item())
        xyza_deltas.append(xyza1_delta.clone())

        plt.cla()
        plt.subplot(1, 3, 1)
        plt.ylabel('ICP point to %s loss' % ('plane' if cfg.loss_kwargs['icp_point_to_plane'] else 'point'))
        plt.xlabel('Iterations')
        plt.plot(iters, losses, color='k')
        plt.grid(visible=True)

        plt.subplot(1, 3, 2)
        plt.ylabel('L2 pose distance')
        plt.xlabel('Iterations')
        plt.plot(iters, torch.stack(xyza_deltas, dim=0).detach()[:, 0], color='r', label='dx')
        plt.plot(iters, torch.stack(xyza_deltas, dim=0).detach()[:, 1], color='g', label='dy')
        plt.plot(iters, torch.stack(xyza_deltas, dim=0).detach()[:, 2], color='b', label='dz')
        plt.grid(visible=True)

        plt.subplot(1, 3, 3)
        plt.ylabel('L2 orient distance')
        plt.xlabel('Iterations')
        plt.plot(iters, torch.linalg.norm(torch.stack(xyza_deltas, dim=0).detach()[:, 3:], dim=1), label='da')
        plt.grid(visible=True)

        plt.pause(0.01)
        plt.draw()

        if True and it % 200 == 0:
            print('Distance between clouds: %f', (torch.linalg.norm(pose1[:3, 3] - pose2[:3, 3])))
            print('Changed pose of the first cloud by: %s [m]' % torch.linalg.norm(xyza1_delta[:3]).detach())

            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(cloud1_corr.points.detach())
            pcd1.colors = o3d.utility.Vector3dVector(torch.zeros_like(cloud1_corr.points.detach()) +
                                                     torch.tensor([1, 0, 0]))
            pcd1.normals = o3d.utility.Vector3dVector(cloud1_corr.normals.detach())

            o3d.visualization.draw_geometries([pcd1, pcd2], point_show_normal=cfg.loss_kwargs['icp_point_to_plane'])
    plt.show()


def test():
    test_eigh3()


def main():
    # test()
    # demo()
    pose_correction_demo()


if __name__ == '__main__':
    main()

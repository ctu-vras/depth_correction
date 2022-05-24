from __future__ import absolute_import, division, print_function
from .depth_cloud import DepthCloud
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import torch
import torch.nn.functional as fun

__all__ = [
    'filter_depth',
    'filter_eigenvalue',
    'filter_eigenvalue_ratio',
    'filter_eigenvalue_ratios',
    'filter_eigenvalues',
    'filter_grid',
    'filter_shadow_points',
    'filter_valid_neighbors',
    'within_bounds',
]

default_rng = np.random.default_rng(135)


def filter_grid(cloud, grid_res, only_mask=False, keep='random', preserve_order=False, log=False, rng=default_rng):
    """Keep single point within each cell. Order is not preserved."""
    assert isinstance(cloud, (DepthCloud, np.ndarray, torch.Tensor))
    assert isinstance(grid_res, float) and grid_res > 0.0
    assert keep in ('first', 'random', 'last')

    # Convert to numpy array with positions.
    if isinstance(cloud, DepthCloud):
        x = cloud.get_points().detach().cpu().numpy()
    elif isinstance(cloud, np.ndarray):
        if cloud.dtype.names:
            x = structured_to_unstructured(cloud[['x', 'y', 'z']])
        else:
            x = cloud
    elif isinstance(cloud, torch.Tensor):
        x = cloud.detach().cpu().numpy()

    # Create voxel indices.
    keys = np.floor(x / grid_res).astype(int).tolist()

    # Last key will be kept, shuffle if needed.
    # Create index array for tracking the input points.
    ind = list(range(len(keys)))
    if keep == 'first':
        # Make the first item last.
        keys = keys[::-1]
        ind = ind[::-1]
    elif keep == 'random':
        # Make the last item random.
        rng.shuffle(ind)
        # keys = keys[ind]
        keys = [keys[i] for i in ind]
    elif keep == 'last':
        # Keep the last item last.
        pass

    # Convert to immutable keys (tuples).
    keys = [tuple(i) for i in keys]

    # Dict keeps the last value for each key (already reshuffled).
    key_to_ind = dict(zip(keys, ind))
    if preserve_order:
        ind = sorted(key_to_ind.values())
    else:
        ind = list(key_to_ind.values())

    if log:
        # print('%.3f = %i / %i points kept (grid res. %.3f m).'
        #       % (mask.double().mean(), mask.sum(), mask.numel(), grid_res))
        print('%.3f = %i / %i points kept (grid res. %.3f m).'
              % (len(ind) / len(keys), len(ind), len(keys), grid_res))

    # TODO: Convert to boolean mask?
    if only_mask:
        # return mask
        return ind

    filtered = cloud[ind]
    return filtered


def within_bounds(x, min=None, max=None, bounds=None, log_variable=None):
    """Mask of x being within bounds  min <= x <= max."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    assert isinstance(x, torch.Tensor)

    keep = torch.ones((x.numel(),), dtype=torch.bool, device=x.device)

    if bounds:
        assert min is None and max is None
        min, max = bounds

    if min is not None and min > -float('inf'):
        if not isinstance(min, torch.Tensor):
            min = torch.tensor(min)
        keep = keep & (x.flatten() >= min)
    if max is not None and max < float('inf'):
        if not isinstance(max, torch.Tensor):
            max = torch.tensor(max)
        keep = keep & (x.flatten() <= max)

    if log_variable is not None:
        print('%.3f = %i / %i points kept (%.3g <= %s <= %.3g).'
              % (keep.double().mean(), keep.sum(), keep.numel(),
                 min if min is not None else float('nan'),
                 log_variable,
                 max if max is not None else float('nan')))

    return keep


def filter_depth(cloud, min=None, max=None, only_mask=False, log=False):
    """Keep points with depth in bounds."""
    assert isinstance(cloud, (DepthCloud, np.ndarray))

    if isinstance(cloud, DepthCloud):
        depth = cloud.depth
    elif isinstance(cloud, np.ndarray):
        if cloud.dtype.names:
            x = structured_to_unstructured(cloud[['x', 'y', 'z']])
        else:
            x = cloud

        if cloud.dtype.names and 'vp_x' in cloud.dtype.names:
            vp = structured_to_unstructured(cloud[['vp_%s' % f for f in 'xyz']])
        else:
            vp = np.zeros((1, 3), dtype=x.dtype)

        x = torch.as_tensor(x)
        vp = torch.as_tensor(vp)
        depth = torch.linalg.norm(x - vp, dim=1, keepdim=True)

    keep = within_bounds(depth, min=min, max=max, log_variable='depth' if log else None)
    if only_mask:
        return keep
    filtered = cloud[keep]
    return filtered


def filter_valid_neighbors(cloud, min=None, only_mask=False, log=False):
    """Keep points with enough valid neighbors."""
    assert isinstance(cloud, DepthCloud)
    assert cloud.neighbors is not None
    num_valid = cloud.valid_neighbor_mask().sum(dim=-1)
    keep = within_bounds(num_valid, min=min, log_variable='valid neighbors' if log else None)
    if only_mask:
        return keep
    filtered = cloud[keep]
    return filtered


def filter_eigenvalue(cloud, eigenvalue=0, min=None, max=None, only_mask=False, log=False):
    """Keep points with specific eigenvalue in bounds."""
    with torch.no_grad():
        keep = within_bounds(cloud.eigvals[:, eigenvalue],
                             min=min, max=max, log_variable='eigenvalue %i' % eigenvalue if log else None)
    if only_mask:
        return keep
    filtered = cloud[keep]
    return filtered


def filter_eigenvalues(cloud: DepthCloud, bounds: list, only_mask: bool=False, log: bool=False):
    mask = None
    if bounds:
        for eig, min, max in bounds:
            eig_mask = filter_eigenvalue(cloud, eig, min=min, max=max, only_mask=True, log=log)
            mask = eig_mask if mask is None else mask & eig_mask
    else:
        mask = torch.ones((cloud.size(),), dtype=torch.bool)
    if log and mask is not None:
        print('%.3f = %i / %i points kept (eigenvalues within bounds).'
              % (mask.double().mean(), mask.sum(), mask.numel()))
    if only_mask:
        return mask
    cloud = cloud[mask]
    return cloud


def filter_eigenvalue_ratio(cloud, eigenvalues=(0, 1), min=None, max=None, only_mask=False, log=False):
    """Keep points with specific eigenvalue ratio in bounds."""
    assert cloud.eigvals is not None
    assert len(eigenvalues) == 2
    assert all(0 <= i <= 2 for i in eigenvalues)
    i, j = eigenvalues
    with torch.no_grad():
        ratio = cloud.eigvals[:, i] / cloud.eigvals[:, j]
        keep = within_bounds(ratio, min=min, max=max,
                             log_variable='eigenvalue %i / eigenvalue %i' % eigenvalues if log else None)
    if only_mask:
        return keep
    filtered = cloud[keep]
    return filtered


def filter_eigenvalue_ratios(cloud: DepthCloud, bounds: list, only_mask: bool=False, log: bool=False):
    mask = None
    if bounds:
        for i, j, min, max in bounds:
            eig_mask = filter_eigenvalue_ratio(cloud, (i, j), min=min, max=max, only_mask=True, log=log)
            mask = eig_mask if mask is None else mask & eig_mask
    else:
        mask = torch.ones((cloud.size(),), dtype=torch.bool)
    if log and mask is not None:
        print('%.3f = %i / %i points kept (eigenvalue ratios within bounds).'
              % (mask.double().mean(), mask.sum(), mask.numel()))
    if only_mask:
        return mask
    cloud = cloud[mask]
    return cloud


def filter_shadow_points(cloud: DepthCloud, angle_bounds: list, only_mask: bool=False, log: bool=False):
    """Filter shadow points from the cloud.

    Filter similar to https://wiki.ros.org/laser_filters#ScanShadowsFilter
    bounding minimum and maximum angle among neighboring beams.

    :param cloud:
    :param angle_bounds:
    :param only_mask:
    :param log:
    :return:
    """
    assert cloud.vps is not None
    assert cloud.dir_neighbors is not None

    # Sanitize angle bounds (make both valid).
    if angle_bounds[0] is None or not (angle_bounds[0] >= 0.0):
        angle_bounds[0] = 0.0
    if angle_bounds[1] is None or not (angle_bounds[1] <= torch.pi):
        angle_bounds[1] = torch.pi
    angle_bounds = torch.as_tensor(angle_bounds)
    assert isinstance(angle_bounds, torch.Tensor)

    # TODO: Convert to cos and bound cos.
    # cos_bounds = torch.cos(angle_bounds)

    # Create vectors (viewpoint - x) and (neighbor - x).
    x = cloud.get_points()
    o = cloud.vps
    ox = o.unsqueeze(dim=1) - x.unsqueeze(dim=1)
    nx = x[cloud.dir_neighbors] - x.unsqueeze(dim=1)

    # Compute angle between these vectors.
    c = fun.cosine_similarity(ox, nx, dim=-1)
    a = torch.acos(c)
    # Sanitize invalid angles (put them within bounds).
    invalid = (cloud.dir_neighbor_weights != 1.0)
    a[invalid] = angle_bounds.mean()

    # Compare minimum and maximum angles among neighbors to the bounds.
    a_min = a.amin(dim=-1)
    a_max = a.amax(dim=-1)
    mask = (a_min >= angle_bounds[0]) & (a_max <= angle_bounds[1])

    if log and mask is not None:
        print('%.3f = %i / %i points kept (shadow points removed).'
              % (mask.double().mean(), mask.sum(), mask.numel()))

    if only_mask:
        return only_mask

    cloud = cloud[mask]
    return cloud

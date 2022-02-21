from __future__ import absolute_import, division, print_function
from .depth_cloud import DepthCloud
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import torch


def filter_grid(cloud, grid_res, only_mask=False, keep='random', preserve_order=False, log=False):
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
        np.random.shuffle(ind)
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


def within_bounds(x, min=None, max=None, log_variable=None):
    """Mask of x being within bounds  min <= x <= max."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    assert isinstance(x, torch.Tensor)

    keep = torch.ones((x.numel(),), dtype=torch.bool, device=x.device)

    if min is not None:
        if not isinstance(min, torch.Tensor):
            min = torch.tensor(min)
        keep = keep & (x.flatten() >= min)
    if max is not None:
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


def filter_eigenvalue(cloud, eigenvalue=0, min=None, max=None, only_mask=False, log=False):
    """Keep points with specific eigenvalue in bounds."""
    keep = within_bounds(cloud.eigvals[:, eigenvalue],
                         min=min, max=max, log_variable='eigenvalue %i' % eigenvalue if log else None)
    if only_mask:
        return keep
    filtered = cloud[keep]
    return filtered


def filter_eigenvalues(cloud, eig_bounds, only_mask=False, log=False):
    mask = None
    for eig, min, max in eig_bounds:
        eig_mask = filter_eigenvalue(cloud, eig, min=min, max=max, only_mask=True, log=log)
        mask = eig_mask if mask is None else mask & eig_mask
    if only_mask:
        return mask
    cloud = cloud[mask]
    return cloud

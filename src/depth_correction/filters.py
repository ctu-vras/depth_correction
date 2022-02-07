from __future__ import absolute_import, division, print_function
from .depth_cloud import DepthCloud
import numpy as np
import torch


def filter_grid(cloud, grid_res, keep='random'):
    """Keep single point within each cell. Order is not preserved."""
    if isinstance(cloud, DepthCloud):
        assert keep == 'last'
        # x = cloud.points or cloud.to_points()
        x = cloud.points if cloud.points is not None else cloud.to_points()
        x = x.detach()
        keys = (x / grid_res).floor().int()
        keys = [tuple(i) for i in keys.numpy().tolist()]
        keep = dict(zip(keys, range(cloud.size())))
        keep = list(keep.values())
        filtered = cloud[keep]
        return filtered

    assert isinstance(cloud, (np.ndarray, torch.Tensor))
    assert isinstance(grid_res, float) and grid_res > 0.0
    assert keep in ('first', 'random', 'last')

    x = cloud
    if isinstance(x, DepthCloud):
        x = x.points or x.to_points()
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()

    if keep == 'first':
        # Make the first item last.
        x = x[::-1]
    elif keep == 'random':
        # Make the last item random.
        np.random.shuffle(x)
    elif keep == 'last':
        # Keep the last item last.
        pass

    # Get integer cell indices, as tuples.
    idx = np.floor(x / grid_res).astype(int)
    idx = [tuple(i) for i in idx]
    # TODO: Allow backward using index_select on cloud.
    # Dict keeps the last value for each key as given by the keep param above.
    x = dict(zip(idx, x))
    x = np.stack(x.values())
    return x


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


def filter_depth(cloud, min=None, max=None, log=True):
    """Keep points with depth in bounds."""
    keep = within_bounds(cloud.depth, min=min, max=max, log_variable='depth' if log else None)
    filtered = cloud[keep]
    return filtered


def filter_eigenvalue(cloud, eigenvalue=0, min=None, max=None, only_mask=False, log=True):
    """Keep points with specific eigenvalue in bounds."""
    keep = within_bounds(cloud.eigvals[:, eigenvalue],
                         min=min, max=max, log_variable='eigenvalue %i' % eigenvalue if log else None)
    if only_mask:
        return keep
    filtered = cloud[keep]
    return filtered

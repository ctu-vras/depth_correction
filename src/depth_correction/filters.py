from __future__ import absolute_import, division, print_function
from .depth_cloud import DepthCloud
import numpy as np
import torch


def filter_grid(cloud, grid_res, keep='random'):
    """Keep single point within each cell. Order is not preserved."""
    if isinstance(cloud, DepthCloud):
        assert keep == 'last'
        x = cloud.points or cloud.to_points()
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

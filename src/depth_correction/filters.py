from __future__ import absolute_import, division, print_function
import numpy as np
import torch


def filter_grid(cloud, grid_res, keep='random'):
    """Select random point within each cell. Order is not preserved."""
    assert isinstance(cloud, (np.ndarray, torch.Tensor))
    assert isinstance(grid_res, float) and grid_res > 0.0
    assert keep in ('first', 'random', 'last')

    x = cloud
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
    # Dict keeps the last value for each key, which set above by keep param.
    x = dict(zip(idx, x))
    # Concatenate and convert rows to cols.
    x = np.stack(x.values())
    return x

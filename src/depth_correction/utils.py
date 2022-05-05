from __future__ import absolute_import, division, print_function
from matplotlib import cm
from timeit import default_timer as timer
import torch

__all__ = [
    'covs',
    'map_colors',
    'timer',
    'timing',
    'trace',
]


def map_colors(values, colormap=cm.gist_rainbow, min_value=None, max_value=None):
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)
    # if not isinstance(colormap, torch.Tensor):
    #     colormap = torch.tensor(colormap, dtype=torch.float64)
    # assert colormap.shape[1] == (2, 3)
    # assert callable(colormap)
    assert callable(colormap) or isinstance(colormap, torch.Tensor)
    if min_value is None:
        min_value = values.min()
    if max_value is None:
        max_value = values.max()
    scale = max_value - min_value
    a = (values - min_value) / scale if scale > 0.0 else values - min_value
    if callable(colormap):
        colors = colormap(a.squeeze())[:, :3]
        return colors
    # TODO: Allow full colormap with multiple colors.
    assert isinstance(colormap, torch.Tensor)
    num_colors = colormap.shape[0]
    a = a.reshape([-1, 1])
    if num_colors == 2:
        # Interpolate the two colors.
        colors = (1 - a) * colormap[0:1] + a * colormap[1:]
    else:
        # Select closest based on scaled value.
        i = torch.round(a * (num_colors - 1))
        colors = colormap[i]
    return colors


def timing(f):
    def timing_wrapper(*args, **kwargs):
        t0 = timer()
        try:
            ret = f(*args, **kwargs)
            return ret
        finally:
            t1 = timer()
            print('%s %.6f s' % (f.__name__, t1 - t0))

    return timing_wrapper


def covs(x, obs_axis=-2, var_axis=-1, center=True, correction=True, weights=None):
    """Create covariance matrices from multiple samples."""
    assert isinstance(x, torch.Tensor)
    assert obs_axis != var_axis
    assert weights is None or isinstance(weights, torch.Tensor)

    # Use sum of provided weights or number of observation for normalization.
    if weights is not None:
        w = weights.sum(dim=obs_axis, keepdim=True)
    else:
        w = x.shape[obs_axis]

    # Center the points if requested.
    if center:
        if weights is not None:
            xm = (weights * x).sum(dim=obs_axis, keepdim=True) / w
        else:
            xm = x.mean(dim=obs_axis, keepdim=True)
        xc = x - xm
    else:
        xc = x

    # Construct possibly weighted xx = x * x^T.
    var_axis_2 = var_axis + 1 if var_axis >= 0 else var_axis - 1
    xx = xc.unsqueeze(var_axis) * xc.unsqueeze(var_axis_2)
    if weights is not None:
        xx = weights.unsqueeze(var_axis) * xx

    # Compute weighted average of x * x^T to get cov.
    if obs_axis < var_axis and obs_axis < 0:
        obs_axis -= 1
    elif obs_axis > var_axis and obs_axis > 0:
        obs_axis += 1
    xx = xx.sum(dim=obs_axis)
    if correction:
        w = w - 1
    w = w.clamp(1e-6, None)
    xx = xx / w

    return xx


def trace(x, dim1=-2, dim2=-1):
    tr = x.diagonal(dim1=dim1, dim2=dim2).sum(dim=-1)
    return tr

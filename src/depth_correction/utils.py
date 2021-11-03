from __future__ import absolute_import, division, print_function
from timeit import default_timer as timer
import torch


def map_colors(values, colormap, min_value=None, max_value=None):
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)
    if not isinstance(colormap, torch.Tensor):
        colormap = torch.tensor(colormap, dtype=torch.float64)
    assert tuple(colormap.shape) == (2, 3)
    if min_value is None:
        min_value = values.min()
    if max_value is None:
        max_value = values.max()
    a = (values - min_value) / (max_value - min_value)
    # TODO: Allow full colormap with multiple colors.
    # num_colors = colormap.shape[0]
    # i0 = torch.floor(a * (num_colors - 1))
    # i0 = i1 + 1
    a = a.reshape([-1, 1])
    colors = (1 - a) * colormap[0:1] + a * colormap[1:]
    return colors


def timing(f):
    def timing_wrapper(*args, **kwargs):
        t0 = timer()
        ret = f(*args, **kwargs)
        t1 = timer()
        print('%s %.6f s' % (f.__name__, t1 - t0))
        return ret
    return timing_wrapper

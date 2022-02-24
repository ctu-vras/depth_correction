from __future__ import absolute_import, division, print_function

import pandas as pd
from matplotlib import cm
from timeit import default_timer as timer
import torch
import tabulate


__all__ = [
    'map_colors',
    'timer',
    'timing',
]


# def map_colors(values, colormap=cm.nipy_spectral, min_value=None, max_value=None):
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
        ret = f(*args, **kwargs)
        t1 = timer()
        print('%s %.6f s' % (f.__name__, t1 - t0))
        return ret
    return timing_wrapper


class Table:
    def __init__(self, data=None):
        assert isinstance(data, list) or isinstance(data, dict)
        self.data = data
        self.headers = ''
        if isinstance(data, dict):
            self.headers = 'keys'

    def show(self):
        table = tabulate.tabulate(self.data, headers=self.headers)
        print(table)
        return table

    def to_latex(self):
        table = tabulate.tabulate(self.data, headers=self.headers, tablefmt='latex')
        print(table)
        return table

    def get_row(self):
        pass

    def get_column(self):
        pass

    def show_header(self):
        pass

    @staticmethod
    def concatenate(tables: list, axis=0):
        dfs = []
        for i, tab in enumerate(tables):
            assert isinstance(tab.data, dict)
            dfs.append(pd.DataFrame(tab.data))
        df = pd.concat(dfs, axis=axis)
        table = Table(list(df.values))  # remove index column in pd Dataframe
        return table


def test():
    from data.asl_laser import dataset_names

    data1 = {'Sequence': dataset_names, 'Loss': torch.rand(len(dataset_names))}
    data2 = {'Sequence': dataset_names, 'Loss': torch.rand(len(dataset_names))}
    data3 = {'Sequence': dataset_names, 'Loss': torch.rand(len(dataset_names))}

    # tab = Table(data1)
    tab = Table.concatenate([Table(data1), Table(data2), Table(data3)])
    tab.show()
    # tab.to_latex()


if __name__ == '__main__':
    test()

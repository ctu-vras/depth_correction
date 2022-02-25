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
        # assert isinstance(data, list) or isinstance(data, dict)
        self.headers = ''
        if isinstance(data, dict):
            self.headers = 'keys'
        if data is not None:
            self.data = pd.DataFrame(data)

    def show(self):
        table = tabulate.tabulate(self.data, headers=self.headers)
        print(table)

    def to_latex(self):
        table = tabulate.tabulate(self.data, headers=self.headers, tablefmt='latex')
        print(table)
        return table

    def concatenate(self, tables: list, names=None, axis=0):
        dfs = []
        for i, tab in enumerate(tables):
            assert isinstance(tab.data, pd.DataFrame)
            dfs.append(tab.data)
        data = pd.concat(dfs, names=names, axis=axis)
        self.headers = names
        self.data = data

    def mean(self, axis=1, keep_names_series=None):
        assert axis == 0 or axis == 1
        data = pd.DataFrame(self.data.mean(axis=axis))
        if keep_names_series:
            names = pd.DataFrame(self.data[keep_names_series])
            # remove duplicates
            if axis == 0:
                names = names.loc[~names.columns.duplicated(), :]
            elif axis == 1:
                names = names.loc[:, ~names.columns.duplicated()]
            data = pd.concat([names, data], axis=axis)
        self.data = data


def tables_demo():
    from data.asl_laser import dataset_names
    import os

    # loss demo: average across sequences
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..',
                                    'gen/depth_1.0-15.0_grid_0.10_r0.20/loss_eval_min_eigval_loss.csv'),
                       delimiter=' ', names=['Sequence', 'Loss'])
    # data = pd.DataFrame(data1['Loss'].values, index=data1['Sequence'])
    data = data.groupby("Sequence").mean()

    tab = Table(data)
    tab.show()
    tab.to_latex()

    # concatenation and averaging demo
    data1 = {'Sequence': dataset_names, 'Loss': torch.rand(len(dataset_names))}
    data2 = {'Sequence': dataset_names, 'Loss': torch.rand(len(dataset_names))}

    tab = Table()
    tab.concatenate([Table(data1), Table(data2)], names=['Sequence', 'Loss', 'Sequence', 'Loss'], axis=1)
    tab.show()
    tab.mean(axis=1, keep_names_series='Sequence')
    tab.show()
    tab.to_latex()

    # different losses concatenation
    data1 = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..',
                                     'gen/depth_1.0-15.0_grid_0.10_r0.20/loss_eval_min_eigval_loss.csv'),
                        delimiter=' ', names=['Sequence', 'Loss'])
    data1 = data1.groupby("Sequence").mean()
    data2 = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..',
                                     'gen/depth_1.0-15.0_grid_0.10_r0.20/loss_eval_trace_loss.csv'),
                        delimiter=' ', names=['Sequence', 'Loss'])
    data2 = data2.groupby("Sequence").mean()

    tab = Table()
    tab.concatenate([Table(data1), Table(data2)], names=['Sequence', 'Min eigval loss', 'Trace loss'], axis=1)
    tab.show()
    tab.to_latex()


if __name__ == '__main__':
    tables_demo()

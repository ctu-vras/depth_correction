from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from matplotlib import cm
from timeit import default_timer as timer
import torch
import tabulate
import os
import glob

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
    # https://towardsdatascience.com/how-to-create-latex-tables-directly-from-python-code-5228c5cea09a
    def __init__(self, data=None):
        # assert isinstance(data, list) or isinstance(data, dict)
        self.headers = ''
        self.data = data
        if data is not None:
            self.data = pd.DataFrame(data)
            self.headers = self.data.columns

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
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html

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
    data1 = data
    data2 = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..',
                                     'gen/depth_1.0-15.0_grid_0.10_r0.20/loss_eval_trace_loss.csv'),
                        delimiter=' ', names=['Sequence', 'Loss'])
    data2 = data2.groupby("Sequence").mean()

    tab = Table()
    tab.concatenate([Table(data1), Table(data2)], names=['Sequence', 'Min eigval loss', 'Trace loss'], axis=1)
    tab.show()
    tab.to_latex()

    # SLAM accuracy data
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..',
                                    'gen/depth_1.0-15.0_grid_0.10_r0.20/slam_eval_ethzasl_icp_mapper.csv'),
                       delimiter=' ', names=['Sequence', 'Orient accuracy [rad]', 'Pose accuracy [m]'])
    data = data.groupby("Sequence").mean()

    tab = Table(data)
    tab.show()
    tab.to_latex()


def slam_postprocess_demo():
    # poses_sources = ['gt', 'ethzasl_icp_mapper']
    models = ['polynomial', 'scaledpolynomial']
    # folds = range(4)
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'gen')
    # experiments = os.listdir(path)

    print("\n-------------------------------------------------------------------------------------------------------\n")
    print(" SLAM accuracy table ")

    def get_slam_accuracy(pose_src='gt', model='*', loss='*', split='train'):
        dfs = None
        for i, fname in enumerate(glob.glob(os.path.join(path,
                                            '*/%s_%s_%s/fold_*/slam_eval*%s.csv' % (pose_src, model, loss, split)))):
            df = pd.read_csv(fname, delimiter=' ', header=None)
            if i == 0:
                dfs = df
            else:
                dfs = pd.concat([dfs, df])
        orient_acc_rad, trans_acc_m = dfs.mean(axis=0).values
        orient_acc_deg = np.rad2deg(orient_acc_rad)

        orient_acc_rad_std, trans_acc_m_std = dfs.std(axis=0).values
        orient_acc_deg_std = np.rad2deg(orient_acc_rad_std)
        return (orient_acc_deg, orient_acc_deg_std), (trans_acc_m, trans_acc_m_std)

    for fname in glob.glob(os.path.join(path, '*/slam_eval*.csv')):
        print(fname)
        df = pd.read_csv(fname, delimiter=' ', header=None)
        orient_acc_rad_base, trans_acc_m_base = df.mean(axis=0).values
        orient_acc_deg_base = np.rad2deg(orient_acc_rad_base)

        orient_acc_rad_base_std, trans_acc_m_base_std = df.std(axis=0).values
        orient_acc_deg_base_std = np.rad2deg(orient_acc_rad_base_std)

    table = [["base model", u"%.6f (\u00B1 %.6f)" % (orient_acc_deg_base, orient_acc_deg_base_std),
                            u"%.6f (\u00B1 %.6f)" % (trans_acc_m_base, trans_acc_m_base_std)]]
    for model in models:
        table.append([model, ", ".join([u"%.6f (\u00B1 %.6f)" % get_slam_accuracy(model=model, split='train')[0],
                                        u"%.6f (\u00B1 %.6f)" % get_slam_accuracy(model=model, split='val')[0],
                                        u"%.6f (\u00B1 %.6f)" % get_slam_accuracy(model=model, split='test')[0]]),
                             ", ".join([u"%.6f (\u00B1 %.6f)" % get_slam_accuracy(model=model, split='train')[1],
                                        u"%.6f (\u00B1 %.6f)" % get_slam_accuracy(model=model, split='val')[1],
                                        u"%.6f (\u00B1 %.6f)" % get_slam_accuracy(model=model, split='test')[1]])])

    print(tabulate.tabulate(table,
                            ["model", "orientation accuracy (train, val, test), [deg]",
                             "translation accuracy (train, val, test), [m]"], tablefmt="grid"))

    print(tabulate.tabulate(table,
                            ["model", "orientation accuracy (train, val, test), [rad]",
                             "translation accuracy (train, val, test), [m]"], tablefmt="latex"))
    print("\n-------------------------------------------------------------------------------------------------------\n")


def loss_postprocess_demo():
    # poses_sources = ['gt', 'ethzasl_icp_mapper']
    models = ['polynomial', 'scaledpolynomial']
    losses = ['min_eigval_loss', 'trace_loss']
    # folds = range(4)
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'gen')
    # experiments = os.listdir(path)

    print(" Losses table ")

    def get_losses(pose_src='gt', model='*', loss='*', split='*'):
        dfs = None
        for i, fname in enumerate(glob.glob(os.path.join(path,
                                                         '*/%s_%s_*/fold_*/loss_eval_%s_%s.csv' % (
                                                                 pose_src, model, loss, split)))):
            df = pd.read_csv(fname, delimiter=' ', header=None)
            if i == 0:
                dfs = df
            else:
                dfs = pd.concat([dfs, df])
        loss_mean, loss_std = dfs.mean(axis=0).values[0], dfs.std(axis=0).values[0]
        return loss_mean, loss_std

    base_loss_values = []
    for loss in losses:
        for fname in glob.glob(os.path.join(path, '*/loss_eval_%s.csv' % loss)):
            df = pd.read_csv(fname, delimiter=' ', header=None)
            base_loss_values.append([df.mean(axis=0).values[0], df.std(axis=0).values[0]])
    assert len(base_loss_values) == len(losses)

    table = [["base model"] + [u"%.6f (\u00B1 %.6f)" % (loss, std) for loss, std in base_loss_values]]
    for model in models:
        table.append(
            [model, ", ".join([u"%.6f (\u00B1 %.6f)" % get_losses(pose_src='gt', model=model, loss='min_eigval_loss',
                                                                  split='train'),
                               u"%.6f (\u00B1 %.6f)" % get_losses(pose_src='gt', model=model, loss='min_eigval_loss',
                                                                  split='val'),
                               u"%.6f (\u00B1 %.6f)" % get_losses(pose_src='gt', model=model, loss='min_eigval_loss',
                                                                  split='test')
                               ]),
             ", ".join([u"%.6f (\u00B1 %.6f)" % get_losses(pose_src='gt', model=model, loss='trace_loss',
                                                           split='train'),
                        u"%.6f (\u00B1 %.6f)" % get_losses(pose_src='gt', model=model, loss='trace_loss', split='val'),
                        u"%.6f (\u00B1 %.6f)" % get_losses(pose_src='gt', model=model, loss='trace_loss', split='test')
                        ])
             ])

    print(tabulate.tabulate(table,
                            ["model", "min eigval loss (train, val, test)",
                             "trace loss (train, val, test)"], tablefmt="grid"))

    print(tabulate.tabulate(table,
                            ["model", "min eigval loss (train, val, test)",
                             "trace loss (train, val, test)"], tablefmt="latex"))


def sequences_postprocess_demo():
    losses = ['min_eigval_loss', 'trace_loss']
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'gen')

    def get_mean_losses_for_sequences(pose_src='gt', model='*', split='*'):
        mean_losses = {}
        for loss in losses:
            dfs = None
            for i, fname in enumerate(glob.glob(os.path.join(path,
                                                             '*/%s_%s_*/fold_*/loss_eval_%s_%s.csv' % (
                                                                     pose_src, model, loss, split)))):
                df = pd.read_csv(fname, delimiter=' ', header=None, names=['sequence', loss])
                if i == 0:
                    dfs = df
                else:
                    dfs = pd.concat([dfs, df], axis=0)
            df = dfs.groupby('sequence').mean()
            mean_losses[loss] = df
        return mean_losses

    mean_losses = get_mean_losses_for_sequences()
    for loss in losses:
        table = mean_losses[loss]
        # print(tabulate.tabulate(table, ['sequence', loss]))
        print(tabulate.tabulate(table, ['sequence', loss], tablefmt='latex'))


if __name__ == '__main__':
    # tables_demo()
    # slam_postprocess_demo()
    # loss_postprocess_demo()
    sequences_postprocess_demo()

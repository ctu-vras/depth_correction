from __future__ import absolute_import, division, print_function
from .config import Loss, Model, PoseProvider, SLAM
from itertools import product
import glob
import numpy as np
import os
import pandas as pd
import tabulate
import torch
import traceback

poses_sources = list(PoseProvider)
models = list(Model)
losses = list(Loss)
path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'gen'))
print(path)

# Choose dataset
dataset = 'asl_laser'
# dataset = 'semantic_kitti'
if dataset == 'asl_laser':
    # preproc = '{dataset}*g0.10'.format(dataset=dataset)
    preproc = '{dataset}*s0.0175_0.0873-nan'.format(dataset=dataset)
elif dataset == 'semantic_kitti':
    # preproc = '{dataset}*g0.20'.format(dataset=dataset)
    preproc = '{dataset}*s0.0175_0.0873-nan'.format(dataset=dataset)
else:
    raise ValueError('Unsupported dataset: %s.' % dataset)
preproc = os.path.join(path, preproc)
preproc = glob.glob(preproc)
assert len(preproc) == 1
preproc = preproc[0]
slam_eval_baseline_format = '{{preproc}}/{dataset}/*/slam_eval_{{slam}}.csv'.format(dataset=dataset)

# SLAM eval with depth correction filter from training
# slam_eval_format = os.path.join(path, '{preproc}/{pose_provider}_{model}_{loss}_*/split_{split}/slam_eval_{slam}_{set}.csv')
# slam_eval_format = os.path.join(path, '{preproc}/{pose_provider}_{model}_*_{loss}/split_{split}/slam_eval_{slam}_{set}.csv')
slam_eval_format = '{preproc}/{pose_provider}_{model}_*_{loss}/split_{split}/slam_eval_{slam}_{set}.csv'
# SLAM eval with all points corrected
# slam_eval_format = os.path.join(path, '{preproc}/{pose_provider}_{model}_{loss}/split_{split}/eval_all_corrected/slam_eval_{slam}_{set}.csv')


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


def tables_basic_demo():
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

    # SLAM error data
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..',
                                    'gen/depth_1.0-15.0_grid_0.10_r0.20/slam_eval_ethzasl_icp_mapper.csv'),
                       delimiter=' ', names=['Sequence', 'Orient error [rad]', 'Pose error [m]'])
    data = data.groupby("Sequence").mean()

    tab = Table(data)
    tab.show()
    tab.to_latex()


def stats_from_csv(csv_paths, cols):

    assert csv_paths
    assert cols

    dfs = None
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, delimiter=' ', header=None)
        dfs = df if dfs is None else pd.concat([dfs, df])

    mean = dfs[cols].mean(axis=0).values
    std = dfs[cols].std(axis=0).values

    stats = list(zip(mean, std))
    return stats


def slam_error_from_csv(csv_paths, cols=2):
    if not csv_paths:
        traceback.print_stack()

        if cols == 2:
            return (float('nan'), float('nan')), (float('nan'), float('nan'))

        return (float('nan'), float('nan')), (float('nan'), float('nan')), (float('nan'), float('nan'))

    dfs = None

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, delimiter=' ', header=None)
        dfs = df if dfs is None else pd.concat([dfs, df])

    if cols == 2:
        orient_acc_rad, trans_acc_m = dfs[[1, 2]].mean(axis=0).values
        orient_acc_rad_std, trans_acc_m_std = dfs[[1, 2]].std(axis=0).values

        orient_acc_deg = np.rad2deg(orient_acc_rad)
        orient_acc_deg_std = np.rad2deg(orient_acc_rad_std)

        return (orient_acc_deg, orient_acc_deg_std), (trans_acc_m, trans_acc_m_std)

    mean = dfs[cols].mean(axis=0).values
    std = dfs[cols].std(axis=0).values
    return zip(mean, std)


def get_slam_error(preproc=preproc, pose_src='*', model='*', loss='*', split='train', slam=list(SLAM)[0], cols=2):
    csv_pattern = slam_eval_format.format(preproc=preproc, pose_provider=pose_src, model=model, loss=loss,
                                          split='*', set=split, slam=slam)
    print(csv_pattern)
    csv_paths = glob.glob(csv_pattern)
    print('\n'.join([csv_path[csv_path.index(dataset):] for csv_path in csv_paths]))
    return slam_error_from_csv(csv_paths, cols=cols)


def get_mean_loss(preproc=preproc, pose_src='*', model='*', loss='*', eval_loss='*', subset='train'):
    loss_eval_format = os.path.join(path, '{preproc}/{pose_provider}_{model}_*_{loss}/split_{split}/loss_eval_{eval_loss}_{subset}.csv')
    csv_pattern = loss_eval_format.format(preproc=preproc, pose_provider=pose_src, model=model, loss=loss,
                                          split='*', eval_loss=eval_loss, subset=subset)
    print(csv_pattern)
    csv_paths = glob.glob(csv_pattern)
    print('\n'.join([csv_path[csv_path.index(dataset):] for csv_path in csv_paths]))
    return stats_from_csv(csv_paths, cols=[1])


def slam_localization_error_demo():
    print(" SLAM error table ")
    # TODO: *base* variables are rewritten in each iterations.
    slam_eval_baseline_pattern = os.path.join(path, slam_eval_baseline_format.format(preproc=preproc, slam=list(SLAM)[0]))
    print('slam_eval_baseline_pattern:', slam_eval_baseline_pattern)
    csv_paths = glob.glob(slam_eval_baseline_pattern)
    print(*csv_paths, sep='\n')
    (orient_acc_deg_base, orient_acc_deg_base_std), (trans_acc_m_base, trans_acc_m_base_std) = \
            slam_error_from_csv(csv_paths)

    for pose_src in poses_sources:

        print('\nSLAM accuracy for initial localization source provider: %s\n' % pose_src)

        for loss in losses + ["*"]:  # for each separate loss as well as averaged
            print("\nLocalization error evaluation with loss: %s\n" % loss)

            table = [["Base model", u"%.3f (\u00B1 %.3f)" % (orient_acc_deg_base, orient_acc_deg_base_std),
                      u"%.3f (\u00B1 %.3f)" % (trans_acc_m_base, trans_acc_m_base_std)]]
            for model in models:
                table.append(
                    [model.capitalize()] + [
                        ", ".join([u"%.3f (\u00B1 %.3f)" % get_slam_error(pose_src=pose_src,
                                                                          model=model,
                                                                          loss=loss,
                                                                          split='train')[i],
                                   u"%.3f (\u00B1 %.3f)" % get_slam_error(pose_src=pose_src,
                                                                          model=model,
                                                                          loss=loss,
                                                                          split='val')[i],
                                   u"%.3f (\u00B1 %.3f)" % get_slam_error(pose_src=pose_src,
                                                                          model=model,
                                                                          loss=loss,
                                                                          split='test')[i]])
                        for i in range(2)])

            print(tabulate.tabulate(table,
                                    ["model", "orientation error (train, val, test), [deg]",
                                     "translation error (train, val, test), [m]"], tablefmt="grid"))

            print(tabulate.tabulate(table,
                                    ["model", "orientation error (train, val, test), [deg]",
                                     "translation error (train, val, test), [m]"], tablefmt="latex"))


def slam_localization_error_tables():
    slam_eval_baseline_pattern = os.path.join(path, slam_eval_baseline_format.format(preproc=preproc, slam=list(SLAM)[0]))
    print('slam_eval_baseline_pattern:', slam_eval_baseline_pattern)
    csv_paths = glob.glob(slam_eval_baseline_pattern)
    print(*csv_paths, sep='\n')
    # cols = [1, 2, 3]
    cols = [1, 2]
    base_res = slam_error_from_csv(csv_paths, cols)
    base_res = list(base_res)
    print(base_res)

    model_map = {Model.Polynomial: '$\\epsilon_\\mathrm{p}$ (\\ref{eq:polynomial_model})',
                 Model.ScaledPolynomial: '$\\epsilon_\\mathrm{sp}$ (\\ref{eq:scaled_polynomial_model})'}
    loss_map = {Loss.min_eigval_loss: '$\\lambda_1$ (\\ref{eq:min_eig_loss})',
                Loss.trace_loss: '$\\trace \\m{Q}$ (\\ref{eq:trace_loss})'}
    pose_map = {PoseProvider.ground_truth: 'GT',
                SLAM.norlab_icp_mapper: 'SLAM'}
    subsets = ['train', 'val', 'test']
    table = []
    # for model, loss, pose_src, col, subset in product(models, losses, poses_sources, cols, subsets):
    for model, loss, pose_src in product(models, losses, poses_sources):

        table.append([model_map[model], loss_map[loss], pose_map[pose_src]])

        for col, subset in product(cols, subsets):

            res = get_slam_error(pose_src=pose_src, model=model, loss=loss, split=subset, cols=[col])
            res = list(res)
            assert len(res) == 1
            mean, std = res[0]

            if col == 1:  # orientation, from rad to deg
                mean, std = np.rad2deg([mean, std])
            elif col == 3:  # ratio to percentage
                mean, std = 100 * mean, 100 * std

            if subset == 'test':
                # table[-1] += ['%.3f \u00B1 %.3f' % (mean, std)]
                # table[-1] += ['%.3f \\pm %.3f' % (mean, std)]
                table[-1] += ['$%.3f \\pm %.3f$' % (mean, std)]
            else:
                table[-1] += ['$%.3f$' % mean]

    base_table = [[]]
    for i, col in enumerate(cols):
        mean, std = base_res[i]
        if col == 1:
            mean, std = np.rad2deg([mean, std])
        elif col == 3:
            mean, std = 100 * mean, 100 * std

        base_table[-1] += ['$%.3f \\pm %.3f$' % (mean, std)]

    print()
    print('SLAM results with no correction')
    print(tabulate.tabulate(base_table, tablefmt='latex_raw'))
    print()
    print('SLAM results with depth correction')
    print(tabulate.tabulate(table, tablefmt='latex_raw'))


def mean_loss_tables():
    # TODO: baseline
    # slam_eval_baseline_pattern = os.path.join(path, slam_eval_baseline_format.format(preproc=preproc, slam=list(SLAM)[0]))
    # print('slam_eval_baseline_pattern:', slam_eval_baseline_pattern)
    # csv_paths = glob.glob(slam_eval_baseline_pattern)
    # print(*csv_paths, sep='\n')
    # # cols = [1, 2, 3]
    # cols = [1, 2]
    # base_res = slam_error_from_csv(csv_paths, cols)
    # base_res = list(base_res)
    # print(base_res)

    model_map = {Model.Polynomial: '$\\epsilon_\\mathrm{p}$ (\\ref{eq:polynomial_model})',
                 Model.ScaledPolynomial: '$\\epsilon_\\mathrm{sp}$ (\\ref{eq:scaled_polynomial_model})'}
    loss_map = {Loss.min_eigval_loss: '$\\lambda_1$ (\\ref{eq:min_eig_loss})',
                Loss.trace_loss: '$\\trace \\m{Q}$ (\\ref{eq:trace_loss})'}
    pose_map = {PoseProvider.ground_truth: 'GT',
                SLAM.norlab_icp_mapper: 'SLAM'}
    subsets = ['train', 'val', 'test']
    headers = []
    headers_done = False
    table = []
    # for model, loss, pose_src, col, subset in product(models, losses, poses_sources, cols, subsets):
    for model, loss, pose_src in product(models, losses, poses_sources):

        table.append([model_map[model], loss_map[loss], pose_map[pose_src]])

        for eval_loss, subset in product(losses, subsets):

            if not headers_done:
                headers.append('%s %s' % (eval_loss, subset))

            res = get_mean_loss(pose_src=pose_src, model=model, loss=loss, eval_loss=eval_loss, subset=subset)
            assert len(res) == 1

            mean, std = res[0]

            # table[-1] += ['$%.3f$' % (1000. * mean)]
            table[-1] += ['$%.6f$' % mean]
            # table[-1] += ['$%.6f$' % (1000 * mean)]

        headers_done = True

    # base_table = [[]]
    # for i, col in enumerate(cols):
    #     mean, std = base_res[i]
    #     if col == 1:
    #         mean, std = np.rad2deg([mean, std])
    #     elif col == 3:
    #         mean, std = 100 * mean, 100 * std
    #
    #     base_table[-1] += ['$%.3f \\pm %.3f$' % (mean, std)]
    #
    # print()
    # print('SLAM results with no correction')
    # print(tabulate.tabulate(base_table, tablefmt='latex_raw'))

    print()
    print('Loss evaluation results with depth correction')
    print(headers)
    print(tabulate.tabulate(table, tablefmt='latex_raw'))


def mean_loss_over_sequences_and_data_splits_demo():

    def get_losses(pose_src='*', model='*', loss='*', split='*'):
        for i, fname in enumerate(glob.glob(os.path.join(path,
                                                         '*/%s_%s_*/split_*/loss_eval_%s_%s.csv' % (
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
            base_loss_values.append([df[[1]].mean(axis=0).values[0], df[[1]].std(axis=0).values[0]])
    assert len(base_loss_values) == len(losses)

    for pose_src in poses_sources:

        print('\nMean losses over sequences for localization source: %s\n' % pose_src)

        table = [["Base model"] + [u"%.3f (\u00B1 %.3f)" % tuple(10e3 * np.asarray([loss, std])) for loss, std in
                                   base_loss_values]]
        for model in models:
            table.append(
                [model.capitalize()] + [
                    ", ".join([u"%.3f (\u00B1 %.3f)" % tuple(10e3 * np.asarray(get_losses(pose_src=pose_src,
                                                                                          model=model,
                                                                                          loss=loss,
                                                                                          split='train'))),
                               u"%.3f (\u00B1 %.3f)" % tuple(10e3 * np.asarray(get_losses(pose_src=pose_src,
                                                                                          model=model,
                                                                                          loss=loss,
                                                                                          split='val'))),
                               u"%.3f (\u00B1 %.3f)" % tuple(10e3 * np.asarray(get_losses(pose_src=pose_src,
                                                                                          model=model,
                                                                                          loss=loss,
                                                                                          split='test')))
                               ]) for loss in losses]
            )

        print(tabulate.tabulate(table,
                                ["model", "min eigval loss (train, val, test) * 10e-3",
                                 "trace loss (train, val, test) * 10e-3"], tablefmt="grid"))

        print(tabulate.tabulate(table,
                                ["model", "min eigval loss (train, val, test) * 10e-3",
                                 "trace loss (train, val, test) * 10e-3"], tablefmt="latex"))


def results_for_individual_sequences_demo(std=False):

    def get_data(optimized=True, model='*', split='*'):
        data = []
        names = ['sequence']
        for loss in losses:
            if std:
                if optimized:
                    names += [loss + ' * 10e-3', loss + '_std * 10e-3']
                else:
                    names += [loss + '_0 * 10e-3', loss + '_0_std * 10e-3']
            else:
                if optimized:
                    names += [loss + ' * 10e-3']
                else:
                    names += [loss + '_0 * 10e-3']

            # results, corrected with model or not
            if optimized:
                files_dir = '*/*_%s_*/split_*/loss_eval_%s_%s.csv' % (model, loss, split)
                # files_dir = '*/*_%s_*/split_*/eval_all_corrected/loss_eval_%s_%s.csv' % (model, loss, split)
            else:
                files_dir = '*/loss_eval_%s.csv' % loss

            for i, fname in enumerate(glob.glob(os.path.join(path, files_dir))):
                df = pd.read_csv(fname, delimiter=' ', header=None, names=['sequence', loss])
                if i == 0:
                    dfs = df
                else:
                    dfs = pd.concat([dfs, df], axis=0)
            if std:
                data.append(pd.concat([dfs.groupby('sequence').mean(), dfs.groupby('sequence').std()], axis=1))
            else:
                data.append(dfs.groupby('sequence').mean())

        df = pd.concat(data, names=names, axis=1)
        df = df.apply(lambda x: 10e3 * x).apply(lambda x: np.round(x, 3))
        df.index = df.index.to_series().apply(lambda x: x.replace('asl_laser/', '').replace('_', ' ')).apply(
            lambda s: s.capitalize())

        return df, names

    def flatten(t):
        return [item for sublist in t for item in sublist]

    df0, names0 = get_data(optimized=False)
    df1, names1 = get_data(optimized=True, split='test')
    names = names1[1:] + names0[1:]

    df = pd.concat([df1, df0], axis=1).set_axis(names, axis=1)
    names = flatten([[loss, loss0] for loss, loss0 in zip(names1[1:], names0[1:])])
    df = df[names]

    print(tabulate.tabulate(df, names))
    print(tabulate.tabulate(df, names, tablefmt='latex'))


if __name__ == '__main__':
    # slam_localization_error_demo()
    slam_localization_error_tables()
    # mean_loss_tables()
    # mean_loss_over_sequences_and_data_splits_demo()
    # results_for_individual_sequences_demo()

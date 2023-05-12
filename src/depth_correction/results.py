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
# dataset = 'asl_laser'
# dataset = 'semantic_kitti'
# dataset = 'fee_corridor'
dataset = 'kitti360'
if dataset == 'asl_laser':
    # preproc = '{dataset}*g0.10'.format(dataset=dataset)
    preproc = '{dataset}*s0.0175_0.0873-nan'.format(dataset=dataset)
elif dataset == 'semantic_kitti':
    # preproc = '{dataset}*g0.20'.format(dataset=dataset)
    preproc = '{dataset}*s0.0175_0.0873-nan'.format(dataset=dataset)
elif dataset == 'fee_corridor':
    preproc = f'{dataset}*g0.20'
elif dataset == 'kitti360':
    preproc = f'{dataset}_d5-25_g0.20'
else:
    raise ValueError('Unsupported dataset: %s.' % dataset)
preproc = os.path.join(path, preproc)
preproc = glob.glob(preproc)
assert len(preproc) == 1
preproc = preproc[0]
slam_eval_baseline_format = '{{preproc}}/{dataset}/*/slam_eval_{{slam}}.csv'.format(dataset=dataset)
loss_eval_baseline_format = '{{preproc}}/{dataset}/*/loss_eval_{{loss}}.csv'.format(dataset=dataset)

# SLAM eval with depth correction filter from training
# slam_eval_format = os.path.join(path, '{preproc}/{pose_provider}_{model}_{loss}_*/split_{split}/slam_eval_{slam}_{set}.csv')
# slam_eval_format = os.path.join(path, '{preproc}/{pose_provider}_{model}_*_{loss}/split_{split}/slam_eval_{slam}_{set}.csv')
slam_eval_format = '{preproc}/{pose_provider}_*_{model}_*_{loss}_*/split_{split}/slam_eval_{slam}_{set}.csv'
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
    from depth_correction.datasets.asl_laser import dataset_names
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
    # print(csv_pattern)
    csv_paths = glob.glob(csv_pattern)
    # print('\n'.join([csv_path[csv_path.index(dataset):] for csv_path in csv_paths]))
    return slam_error_from_csv(csv_paths, cols=cols)


def get_mean_loss(preproc=preproc, pose_src='*', model='*', loss='*', eval_loss='*', subset='train'):
    loss_eval_format = '{preproc}/{pose_provider}_*_{model}_*_{loss}_*/split_{split}/loss_eval_{eval_loss}_{subset}.csv'
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

    # models = ['Polynomial', 'ScaledPolynomial']
    models = ['Polynomial']
    poses_sources = ['ground_truth']
    # losses = ['min_eigval_loss', 'icp_loss']
    losses = ['min_eigval_loss']

    model_map = {Model.Polynomial: 'SLAM \cite{Pomerleau-2013-AR} + $\\epsilon_\\mathrm{p}$ (\\ref{eq:polynomial_model})',
                 Model.ScaledPolynomial: 'SLAM \cite{Pomerleau-2013-AR} + $\\epsilon_\\mathrm{sp}$ (\\ref{eq:scaled_polynomial_model})'}

    for pose_src in poses_sources:

        print('\nSLAM accuracy for initial localization source provider: %s\n' % pose_src)

        # for loss in losses + ["*"]:  # for each separate loss as well as averaged
        for loss in losses:
            print("\nLocalization error evaluation with loss: %s\n" % loss)

            table = [["SLAM", "$%.2f \\pm %.2f$" % (orient_acc_deg_base, orient_acc_deg_base_std),
                      "$%.2f \\pm %.2f$" % (trans_acc_m_base, trans_acc_m_base_std)]]
            for model in models:
                orient_means, orient_stds = [], []
                trans_means, trans_stds = [], []
                for split in ['train', 'val', 'test']:
                    orient_err, trans_err = get_slam_error(pose_src=pose_src, model=model, loss=loss, split=split)
                    orient_mean, orient_std = orient_err
                    trans_mean, trans_std = trans_err

                    trans_means.append(trans_mean)
                    trans_stds.append(trans_std)
                    orient_means.append(orient_mean)
                    orient_stds.append(orient_std)

                table.append([model_map[model]] +
                             ["$%.2f \\pm %.2f$" % (np.mean(orient_means).item(), np.mean(orient_stds).item())] +
                             ["$%.2f \\pm %.2f$" % (np.mean(trans_means).item(), np.mean(trans_stds).item())])

            print(tabulate.tabulate(table,
                                    ["pipeline", "orientation error [deg]",
                                     "translation error [m]"], tablefmt="latex_raw"))


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
                # table[-1] += ['%.2f \\pm %.2f' % (mean, std)]
                # table[-1] += ['%.2f \\pm %.2f' % (mean, std)]
                table[-1] += ['$%.2f \\pm %.2f$' % (mean, std)]
            else:
                table[-1] += ['$%.3f$' % mean]

    base_table = [[]]
    for i, col in enumerate(cols):
        mean, std = base_res[i]
        if col == 1:
            mean, std = np.rad2deg([mean, std])
        elif col == 3:
            mean, std = 100 * mean, 100 * std

        base_table[-1] += ['$%.2f \\pm %.2f$' % (mean, std)]

    print()
    print('SLAM results with no correction')
    print(tabulate.tabulate(base_table, headers=['orientation error [deg]', 'translation error [m]'], tablefmt='latex_raw'))
    print()
    print('SLAM results with depth correction')
    print(tabulate.tabulate(table, tablefmt='latex_raw'))


def mean_loss_tables():
    # baselines
    for loss in list(Loss):
        if loss == 'icp_loss':
            continue

        loss_eval_baseline_pattern = os.path.join(path, loss_eval_baseline_format.format(preproc=preproc, loss=loss))
        # print('loss_eval_baseline_pattern:', loss_eval_baseline_pattern)
        csv_paths = glob.glob(loss_eval_baseline_pattern)
        # print(*csv_paths, sep='\n')
        cols = [0, 1]
        base_res = stats_from_csv(csv_paths, cols)
        base_res = list(base_res)

        mean, std = base_res[0]

        print(loss, '$%.2f \\pm %.2f$' % (1000 * mean, 1000 * std))

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
    #     base_table[-1] += ['$%.2f \\pm %.2f$' % (mean, std)]
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

        table = [["Base model"] + ["$%.2f \\pm %.2f$" % tuple(10e3 * np.asarray([loss, std])) for loss, std in
                                   base_loss_values]]
        for model in models:
            table.append(
                [model.capitalize()] + [
                    ", ".join(["$%.2f \\pm %.2f$" % tuple(10e3 * np.asarray(get_losses(pose_src=pose_src,
                                                                                       model=model,
                                                                                       loss=loss,
                                                                                       split='train'))),
                               "$%.2f \\pm %.2f$" % tuple(10e3 * np.asarray(get_losses(pose_src=pose_src,
                                                                                       model=model,
                                                                                       loss=loss,
                                                                                       split='val'))),
                               "$%.2f \\pm %.2f$" % tuple(10e3 * np.asarray(get_losses(pose_src=pose_src,
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


def plot_slam_trajs():
    import matplotlib.pyplot as plt
    from .datasets.kitti360 import read_poses, Dataset, prefix

    slam = list(SLAM)[0]
    preproc = f'{dataset}_d5-25_g0.20'
    # slam_poses_baseline_format = f'{preproc}/{dataset}_baseline/*/slam_poses_{slam}.csv'
    slam_poses_format = f'{preproc}/{dataset}/*/slam_poses_{slam}.csv'
    slam_poses_pattern = os.path.join(path, slam_poses_format)

    # plt.figure()
    for poses_scv in glob.glob(slam_poses_pattern):
        # SLAM poses
        _, poses = read_poses(poses_scv)
        _, poses_baseline = read_poses(poses_scv.replace(f'/{dataset}/', f'/{dataset}_baseline_500scans/'))
        print('Poses: %i' % len(poses))
        print('Poses baseline: %s' % len(poses_baseline))
        N = min(len(poses), len(poses_baseline))
        poses = poses[:N]
        poses_baseline = poses_baseline[:N]

        print(np.allclose(poses, poses_baseline))

        seq = poses_scv.split('/')[-2][:2]
        start = 1
        end = N + start
        subseq = seq + '_start_%i_end_%i_step_1' % (start, end)

        # GT poses
        ds = Dataset(name='%s/%s' % (prefix, subseq), zero_origin=True)
        poses_gt = ds.poses

        # transform SLAM poses to global coord frame
        # poses = np.asarray([poses_gt[0] @ p for p in poses])

        # visualization
        plt.rcParams.update({'font.size': 28})
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

        ax.plot(poses_gt[:, 0, 3], poses_gt[:, 1, 3], color='b', linewidth=4, label='GT')
        ax.plot(poses_baseline[:, 0, 3], poses_baseline[:, 1, 3], color='r', linewidth=3, label='SLAM')
        ax.plot(poses[:, 0, 3], poses[:, 1, 3], color='g', linewidth=3, label='SLAM+DC')

        plt.title('Seq.: %s. Start pose: %i, end pose: %i' % (seq, start, end))
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.grid()
        ax.axis('equal')
        ax.legend(loc='lower left')
        # plt.savefig(f'/home/ruslan/Desktop/{dataset}_seq_{seq}_start{start}_end{end}_slam_dc_results.png')
        plt.show()


def slam_error_for_sequences():
    slam = list(SLAM)[0]
    pose_src = 'ground_truth'
    model = 'Polynomial'
    # losses = ['min_eigval_loss', 'trace_loss', 'icp_loss']
    losses = ['min_eigval_loss']

    dfs = None
    for loss in losses:
        for split in ['train', 'val', 'test']:
            slam_eval_pattern = slam_eval_format.format(preproc=preproc, pose_provider=pose_src, model=model, loss=loss,
                                                        split='*', set=split, slam=slam)
            for csv_path in glob.glob(slam_eval_pattern):
                df = pd.read_csv(csv_path, delimiter=' ', header=None)
                dfs = df if dfs is None else pd.concat([dfs, df])

    # set sequences name as index
    # df = dfs.set_index(dfs[0])

    # rename column names
    df = dfs.rename(columns={0: 'seq', 1: 'r_angle', 2: 't_norm', 3: 'rel_angle', 4: 'rel_offset'})
    # select only orient and trans errors (not relative changes)
    df = df[['seq', 'r_angle', 't_norm']]

    # for each sequence compute mean errors
    aggregation_functions = {'r_angle': 'mean', 't_norm': 'mean'}
    df = df.groupby('seq').aggregate(aggregation_functions)
    df['r_angle'] = df['r_angle'].apply(np.rad2deg)

    print(df.to_latex())


def main():
    # slam_localization_error_demo()
    # slam_localization_error_tables()
    # mean_loss_tables()
    # mean_loss_over_sequences_and_data_splits_demo()
    # results_for_individual_sequences_demo()
    # plot_slam_trajs()
    slam_error_for_sequences()


if __name__ == '__main__':
    main()

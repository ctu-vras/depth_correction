#! /usr/bin/env python

from __future__ import absolute_import, division, print_function
import sys
sys.path.append('../src/')
import torch
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer
import numpy as np
from depth_correction.depth_cloud import DepthCloud
from depth_correction.filters import filter_grid
from depth_correction.loss import min_eigval_loss
from depth_correction.model import Linear, Polynomial


MODEL_TYPE = 'Polynomial'  # 'Linear' or 'Polynomial'
N_OPT_ITERS = 100
LR = 0.01
SHOW_RESULTS = False
DATASET = 'ASL_laser'  # 'ASL_laser' or 'UTIAS_3dmap'

if DATASET == 'ASL_laser':
    from data.asl_laser import Dataset, dataset_names
elif DATASET == 'UTIAS_3dmap':
    from data.utias_3dmap import Dataset, dataset_names


def construct_corrected_global_map(ds: Dataset,
                                   model: (Linear, Polynomial),
                                   k_nn=None, r_nn=None) -> DepthCloud:
    assert k_nn or r_nn
    grid_res = 0.05
    clouds = []
    poses = []
    sample_k = 4
    seq_len = 2
    seq_n = np.random.choice(range(len(ds) - seq_len), 1)[0]
    for id in ds.ids[seq_n:seq_n + seq_len * sample_k:sample_k]:
        t = timer()
        cloud = ds.local_cloud(id)
        pose = torch.tensor(ds.cloud_pose(id))
        dc = DepthCloud.from_points(cloud)
        print('%i points read from dataset %s, cloud %i (%.3f s).'
              % (dc.size(), ds.name, id, timer() - t))

        t = timer()
        dc = filter_grid(dc, grid_res, keep='last')
        print('%i points kept by grid filter with res. %.2f m (%.3f s).'
              % (dc.size(), grid_res, timer() - t))

        dc = dc.transform(pose)
        dc.update_all(k=k_nn, r=r_nn)
        dc = model(dc)

        clouds.append(dc)
        poses.append(pose)

    combined = DepthCloud.concatenate(clouds, True)
    return combined


def main():
    print('Loading the datasets...')
    # datasets = [Dataset(name) for name in ('eth',)]
    datasets = [Dataset(name) for name in dataset_names]
    device = torch.device('cpu')

    if MODEL_TYPE == 'Polynomial':
        model = Polynomial(p0=0.0, p1=0.0, device=device)
    elif MODEL_TYPE == 'Linear':
        model = Linear(w0=1.0, w1=0.0, b=0.0, device=device)
    else:
        raise 'Model type is not supported'

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    plot_period = 2
    r_nn = 0.15
    # k_nn = 10
    k_nn = None

    writer = SummaryWriter('./tb_runs/model_%s_lr_%f' % (MODEL_TYPE, LR))
    for i in range(N_OPT_ITERS):
        ds = np.random.choice(datasets, 1)[0]
        print('Dataset len:', len(ds))
        optimizer.zero_grad()

        # TODO: run everything on GPU
        combined = construct_corrected_global_map(ds, model, k_nn, r_nn)  # model is passed to correct local maps
        # combined = model(combined)
        # combined.update_all(r=r_nn, k=k_nn)

        loss, loss_dc = min_eigval_loss(combined, r=r_nn, k=k_nn, offset=True, eigenvalue_bounds=(0.0, 0.05**2))
        print('Loss:', loss.item())
        writer.add_scalar("Loss/min_eigval", loss, i)

        if SHOW_RESULTS and i % plot_period == 0:
            combined.visualize(colors='inc_angles')
            combined.visualize(colors='min_eigval')
            loss_dc.visualize(colors='loss')

        # Optimization step
        loss.backward()
        optimizer.step()

    del ds, datasets
    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()

#! /usr/bin/env python

from __future__ import absolute_import, division, print_function
import sys
sys.path.append('../src/')
import torch
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer
import numpy as np
from depth_correction.depth_cloud import DepthCloud
from depth_correction.filters import filter_depth, filter_eigenvalue, filter_grid
from depth_correction.loss import min_eigval_loss
from depth_correction.model import Linear, Polynomial, ScaledPolynomial


MODEL_TYPE = 'ScaledPolynomial'  # 'Linear', 'Polynomial', 'ScaledPolynomial'
N_OPT_ITERS = 100
LR = 0.001
SHOW_RESULTS = False
DATASET = 'ASL_laser'  # 'ASL_laser', 'UTIAS_3dmap', 'Chilean_Mine'

if DATASET == 'ASL_laser':
    from data.asl_laser import Dataset, dataset_names
elif DATASET == 'UTIAS_3dmap':
    from data.utias_3dmap import Dataset, dataset_names
elif DATASET == 'Chilean_Mine':
    from data.chilean_underground_mine import Dataset, dataset_names
else:
    raise ValueError("Supported datasets: 'ASL_laser', 'UTIAS_3dmap', 'Chilean_Mine'")


def construct_corrected_global_map(ds: Dataset,
                                   model: (Linear, Polynomial),
                                   k_nn=None, r_nn=None) -> DepthCloud:
    assert k_nn or r_nn

    # Cloud preprocessing params
    min_depth = 1.0
    max_depth = 10.0
    grid_res = 0.05

    clouds = []
    poses = []
    sample_k = 4
    seq_len = 2
    seq_n = np.random.choice(range(len(ds) - seq_len), 1)[0]
    device = model.device
    for id in ds.ids[seq_n:seq_n + seq_len * sample_k:sample_k]:
        # t = timer()
        cloud = ds.local_cloud(id)
        pose = torch.tensor(ds.cloud_pose(id))
        dc = DepthCloud.from_points(cloud)
        # print('%i points read from dataset %s, cloud %i (%.3f s).'
        #       % (dc.size(), ds.name, id, timer() - t))  # ~0.06 sec for 180000 pts

        # t = timer()
        dc = filter_depth(dc, min=min_depth, max=max_depth, log=False)
        # print('%i points kept by depth filter with min_depth %.2f, max_depth %.2f m (%.3f s).'
        #       % (dc.size(), min_depth, max_depth, timer() - t))  # ~0.002 sec

        # t = timer()
        dc = filter_grid(dc, grid_res, keep='last')
        # print('%i points kept by grid filter with res. %.2f m (%.3f s).'
        #       % (dc.size(), grid_res, timer() - t))  # ~0.1 sec

        # t = timer()
        pose = pose.to(device)
        dc = dc.to(device)
        # print('moved poses and depth cloud to device (%.3f s).' % (timer() - t))  # ~0.001 sec

        # t = timer()
        dc = dc.transform(pose)
        # print('transformed depth cloud to global frame (%.3f s).' % (timer() - t))  # ~0.001 sec

        t = timer()
        dc.update_all(k=k_nn, r=r_nn)
        print('update_all took (%.3f s).' % (timer() - t))  # ~2.0 sec on CPU and ~7.5 sec on GPU (!)

        t = timer()
        keep = filter_eigenvalue(dc, 0, max=(grid_res / 5)**2, only_mask=True, log=False)
        keep = keep & filter_eigenvalue(dc, 1, min=grid_res**2, only_mask=True, log=False)
        dc = dc[keep]
        dc.update_all(k=k_nn, r=r_nn)
        print('filtering eigvals and update_all took (%.3f s).'
              % (timer() - t))  # ~1.0 sec on CPU and 3.2 sec on GPU (!)

        # t = timer()
        dc = model(dc)
        # print('model inference took (%.3f s).' % (timer() - t))  # ~0.001 sec

        clouds.append(dc)
        poses.append(pose)

    combined = DepthCloud.concatenate(clouds, True)
    return combined


def main():
    print('Loading the datasets...')
    datasets = [Dataset(name) for name in ('eth',)]
    # datasets = [Dataset(name) for name in dataset_names]
    device = torch.device('cuda:0')
    # device = torch.device('cpu')

    if MODEL_TYPE == 'Polynomial':
        model = Polynomial(p0=0.0, p1=0.0, device=device)
    elif MODEL_TYPE == 'ScaledPolynomial':
        model = ScaledPolynomial(p0=0.0, p1=0.0, device=device)
    elif MODEL_TYPE == 'Linear':
        model = Linear(w0=1.0, w1=0.0, b=0.0, device=device)
    else:
        raise ValueError('Model type is not supported')

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    plot_period = 10
    r_nn = 0.15
    # k_nn = 10
    k_nn = None

    writer = SummaryWriter('./tb_runs/model_%s_lr_%f_%s' % (MODEL_TYPE, LR, DATASET))
    min_loss = np.inf
    optimizer.zero_grad()
    for i in range(N_OPT_ITERS):
        # ds = np.random.choice(datasets, 1)[0]
        # dc_combined = construct_corrected_global_map(ds, model, k_nn, r_nn)  # model is passed to correct local maps
        # loss, dc_loss = min_eigval_loss(dc_combined, r=r_nn, k=k_nn, offset=True, eigenvalue_bounds=(0.0, 0.05 ** 2))

        dcs = []
        for ds in datasets:
            dc_combined = construct_corrected_global_map(ds, model, k_nn, r_nn)
            dcs.append(dc_combined)
        loss, dc_loss = min_eigval_loss(dcs, r=r_nn, k=k_nn, offset=True, eigenvalue_bounds=(0.0, 0.05 ** 2))

        print('loss:', loss.item())
        writer.add_scalar("Loss/min_eigval", loss, i)

        if SHOW_RESULTS and i % plot_period == 0:
            for dc in dcs:
                dc.visualize(colors='inc_angles')
                dc.visualize(colors='min_eigval')
            dc_loss.visualize(colors='loss')

        if min_loss > loss.item():
            min_loss = loss.item()
            torch.save(model.state_dict(), './weights/%s.pth' % MODEL_TYPE)
            print('better %s model is saved, loss: %g' % (MODEL_TYPE, min_loss))

        # Optimization step
        loss.backward()
        optimizer.step()

    del ds, datasets
    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()

#! /usr/bin/env python

from __future__ import absolute_import, division, print_function
import sys
sys.path.append('../src/')
import torch
from timeit import default_timer as timer
from data.asl_laser import Dataset, dataset_names
from depth_correction.depth_cloud import DepthCloud
from depth_correction.filters import filter_grid
from depth_correction.loss import min_eigval_loss
from depth_correction.model import Linear, Polynomial


def construct_corrected_global_map(ds: Dataset,
                                   model: (Linear, Polynomial),
                                   k_nn=None, r_nn=None) -> DepthCloud:
    assert k_nn or r_nn
    clouds = []
    poses = []
    sample_k = 10
    device = model.device
    for id in ds.ids[::sample_k]:
        t = timer()
        cloud = ds.local_cloud(id)
        pose = torch.tensor(ds.cloud_pose(id))
        dc = DepthCloud.from_points(cloud)
        print('%i points read from dataset %s, cloud %i (%.3f s).'
              % (dc.size(), ds.name, id, timer() - t))

        t = timer()
        grid_res = 0.05
        dc = filter_grid(dc, grid_res, keep='last')
        print('%i points kept by grid filter with res. %.2f m (%.3f s).'
              % (dc.size(), grid_res, timer() - t))

        t = timer()
        pose = pose.to(device)
        dc = dc.to(device)
        print('Moving DepthCloud to device (%.3f s).'
              % (timer() - t))

        dc = dc.transform(pose)
        dc.update_all(k=k_nn, r=r_nn)
        dc = model(dc)
        # dc.visualize(colors='inc_angles')
        # dc.visualize(colors='min_eigval')

        clouds.append(dc)
        poses.append(pose)

    combined = DepthCloud.concatenate(clouds, True)
    return combined


def main():
    print('Loading the dataset...')
    ds = Dataset(dataset_names[0])
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    model = Polynomial(p0=0.0, p1=0.0, device=device)
    # model = Linear(w0=1.0, w1=0.0, b=0.0, device=device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    Niter = 100
    plot_period = 20
    r_nn = 0.15
    k_nn = 10

    for i in range(Niter):
        optimizer.zero_grad()

        # TODO: run everything on GPU
        combined = construct_corrected_global_map(ds, model, k_nn, r_nn)  # model is passed to correct local maps
        # combined = model(combined)
        # combined.update_all(r=r_nn, k=k_nn)

        loss, loss_dc = min_eigval_loss(combined, r=r_nn, k=k_nn, offset=True, updated_eigval_bounds=(0.0, 0.05 ** 2))
        print('Loss:', loss.item())

        if i % plot_period == 0:
            combined.visualize(colors='inc_angles')
            combined.visualize(colors='min_eigval')
            loss_dc.visualize(colors='loss')

        # Optimization step
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()

#! /usr/bin/env python

from __future__ import absolute_import, division, print_function
import sys
sys.path.append('../src/')
import torch
import numpy as np
from depth_correction.depth_cloud import DepthCloud
from depth_correction.filters import filter_depth, filter_eigenvalue, filter_grid
from depth_correction.loss import min_eigval_loss
from depth_correction.model import Linear, Polynomial
import open3d as o3d


def main():
    device = torch.device('cpu')
    model = Polynomial(p0=0.1, p1=-0.2, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    N = 10000
    r_nn = 0.4
    # create flat point cloud (a wall)
    cloud = np.zeros((N, 3), dtype=np.float32)
    cloud[:, :2] = np.random.rand(N, 2) * 20 - 10  # 10 x 10 m
    cloud[:, 2] = -20.0

    dc = DepthCloud.from_points(cloud)
    dc.update_all(r=r_nn)

    # add disturbance: modify Z coord of point with high incidence angle
    # dc.depth[dc.inc_angles.squeeze() > 0.7] = dc.depth[dc.inc_angles.squeeze() > 0.7] + 1.0
    dc.depth = dc.depth + torch.exp(-torch.sin((dc.inc_angles**2) / 0.3 ** 2))
    dc.update_all(r=r_nn)

    # use model to correct the distortion (hopefully)
    for i in range(10):
        optimizer.zero_grad()

        dc = model(dc)
        # dc.update_all(r=r_nn)

        loss, _ = min_eigval_loss(dc, r=r_nn, offset=True)

        print(torch.unique(dc.eigvals))

        print('loss: %f' % loss.item())

        # Optimization step
        loss.backward(retain_graph=True)
        optimizer.step()

        if i % 2 == 0:
            # dc.visualize(colors='inc_angles')
            dc.visualize(colors='min_eigval')


if __name__ == '__main__':
    main()

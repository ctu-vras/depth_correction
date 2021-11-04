from __future__ import absolute_import, division, print_function
import torch
from timeit import default_timer as timer
from data.asl_laser import Dataset, dataset_names
from depth_correction.depth_cloud import DepthCloud
from depth_correction.filters import filter_grid
from depth_correction.loss import min_eigval_loss
from depth_correction.model import Linear, Polynomial


def construct_corrected_global_map(ds: Dataset, model: (Linear, Polynomial)) -> DepthCloud:
    clouds = []
    poses = []
    for id in ds.ids[::20]:
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

        dc = dc.transform(pose)
        dc.update_all(r=0.15)
        dc = model(dc)
        # dc.visualize(colors='inc_angles')
        # dc.visualize(colors='min_eigval')

        clouds.append(dc)
        poses.append(pose)

    combined = DepthCloud.concatenate(clouds, True)
    return combined


def main():
    # ds = Dataset(dataset_names[0])
    ds = Dataset('eth')

    model = Polynomial(p0=0.0, p1=0.0)
    # model = Linear(w0=1.0, w1=0.0, b=0.0)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    Niter = 100
    plot_period = 20

    for i in range(Niter):
        optimizer.zero_grad()

        combined = construct_corrected_global_map(ds, model)  # model is passed to correct local maps
        # combined = model(combined)
        # combined.update_all(r=0.15)

        loss, loss_dc = min_eigval_loss(combined, r=0.15, offset=True, bounds=(0.0, 0.05 ** 2))
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

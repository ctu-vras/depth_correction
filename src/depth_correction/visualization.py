import torch

from .config import Config
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured


def visualize_incidence_angles(clouds, titles=None, cfg: Config=None):
    if not clouds:
        return
    if hasattr(clouds[0], 'inc_angles'):
        clouds = [obj.inc_angles for obj in clouds]

    fig, axes = plt.subplots(nrows=len(clouds), ncols=1, figsize=(8.0, 12.0), squeeze=False, constrained_layout=True,
                             sharex=True, sharey=True)
    # fig, axes = plt.subplots(nrows=len(clouds), ncols=1, figsize=(8.0, 12.0), squeeze=False,
    #                          sharex=True, sharey=True)
    bins = np.linspace(0.0, 90.0, int(90 / 2.5) + 1)
    for i in range(len(clouds)):
        ax = axes[i, 0]
        cloud = clouds[i]
        if isinstance(cloud, torch.Tensor):
            cloud = cloud.detach().numpy()
        ax.hist(np.rad2deg(cloud), bins=bins, density=True)
        # ax.set_xlabel('incidence angle')
        # ax.set_ylabel('relative frequency')
        if i == len(clouds) - 1:
            ax.set_xlabel('incidence angle')
        if i == len(clouds) // 2:
            ax.set_ylabel('relative frequency')
        ax.set_title(titles[i] if titles else '%i' % i)
        # ax.set_xlim((0.0, np.pi / 2.0))
        ax.set_xlim((0.0, 90.0))
        ax.grid()

    # FIXME: Shared x, y labels.
    #  Constrained and tight layout destroys the shared labels,
    #  not using any layout destroys the titles.
    # fig.text(0.5, 0.04, 'incidence angle', in_layout=True, ha='center')
    # fig.text(0.04, 0.5, 'relative frequency', in_layout=True, va='center', rotation='vertical')
    # fig.tight_layout()
    # fig.subplots_adjust()
    plt.show(block=True)
    # path = os.path.join(cfg.log_dir, 'loss_landscape.png')
    # print('Loss landscape written to %s.' % path)
    # os.makedirs(os.path.dirname(path), exist_ok=True)
    # fig.savefig(path, dpi=300)


def visualize_dataset(ds, cfg: Config):
    import open3d as o3d
    from .preproc import filtered_cloud

    pcd = o3d.geometry.PointCloud()
    poses_pcd = o3d.geometry.PointCloud()
    for k, (cloud, pose) in enumerate(ds):
        cloud = structured_to_unstructured(cloud[['x', 'y', 'z']])
        # filter cloud
        cloud = filtered_cloud(cloud, cfg)

        # transform cloud to common map coordinate frame
        cloud_map = np.matmul(cloud[:, :3], pose[:3, :3].T) + pose[:3, 3:].T

        pcd.points.extend(o3d.utility.Vector3dVector(cloud_map))

    poses_pcd.points.extend(o3d.utility.Vector3dVector(ds.poses[:, :3, 3]))
    green_color = np.zeros_like(ds.poses[:, :3, 3]) + np.array([0, 1, 0])
    poses_pcd.colors = o3d.utility.Vector3dVector(green_color)

    # blue_color = np.zeros_like(pcd.points) + np.array([0, 0, 1])
    # pcd.colors = o3d.utility.Vector3dVector(blue_color)

    o3d.visualization.draw_geometries([pcd, poses_pcd])

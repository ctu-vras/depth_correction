from __future__ import absolute_import, division, print_function
import numpy as np
import torch

__all__ = [
    'DepthCloud'
]


def depth_cloud_from_points(pts, vps=None):
    """Create depth cloud from points and viewpoints.

    :param pts: Points as ...-by-3 tensor.
    :param vps: Viewpoints as ...-by-3 tensor, or None for zero vector.
    :return:
    """
    assert isinstance(pts, torch.Tensor)
    if vps is None:
        vps = torch.zeros((1, 3))
    assert isinstance(vps, torch.Tensor)
    assert vps.shape == pts.shape or vps.shape == (1, 3)
    dirs = pts - vps
    depth = dirs.norm(dim=-1).unsqueeze(-1)
    # TODO: Handle invalid points (zero depth).
    dirs = dirs / depth
    depth_cloud = DepthCloud(vps, dirs, depth)
    return depth_cloud


class DepthCloud(object):

    def __init__(self, vps=None, dirs=None, depth=None):
        """Create depth cloud from viewpoints, directions, and depth.

        :param vps: Viewpoints as ...-by-3 tensor, or None for zero vector.
        :param dirs: Observation directions, ...-by-3 tensor.
        :param depth: Depth map as ...-by-1 tensor.
        """
        if vps is None:
            vps = torch.zeros((1, 3))
        assert isinstance(vps, torch.Tensor)
        assert vps.shape[-1] == 3

        assert isinstance(dirs, torch.Tensor)
        assert dirs.shape[-1] == 3
        assert dirs.shape == vps.shape or vps.shape == (1, 3)

        assert isinstance(depth, torch.Tensor)
        assert depth.shape[-1] == 1
        assert depth.shape[:-1] == dirs.shape[:-1]

        self.vps = vps
        self.dirs = dirs
        self.depth = depth

    def to_points(self):
        pts = self.vps + self.depth * self.dirs
        return pts

    def visualize(self):
        try:
            import open3d as o3d
        except:
            print('Install Open3d for visualization')
            return
        cloud = self.to_points()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        o3d.visualization.draw_geometries([pcd])

    @staticmethod
    def from_points(pts, vps=None):
        """Create depth cloud from points and viewpoints.

        :param pts: Points as ...-by-3 tensor.
        :param vps: Viewpoints as ...-by-3 tensor, or None for zero vector.
        :return:
        """
        if isinstance(pts, np.ndarray):
            pts = torch.tensor(pts)
        assert isinstance(pts, torch.Tensor)
        if vps is None:
            vps = torch.zeros((3,))
        if isinstance(vps, np.ndarray):
            vps = torch.tensor(vps)
        assert isinstance(vps, torch.Tensor)
        print(pts.shape)
        print(vps.shape)
        assert vps.shape == pts.shape or tuple(vps.shape) == (3,)
        dirs = pts - vps
        depth = dirs.norm(dim=-1, keepdim=True)
        # TODO: Handle invalid points (zero depth).
        print(dirs.shape)
        print(depth.shape)
        dirs = dirs / depth
        depth_cloud = DepthCloud(vps, dirs, depth)
        return depth_cloud

from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import open3d as o3d  # used for normals estimation and visualization

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

        self.vps = torch.as_tensor(vps, dtype=torch.float32)
        self.dirs = torch.as_tensor(dirs, dtype=torch.float32)
        self.depth = torch.as_tensor(depth, dtype=torch.float32)

        self.normals = None
        self.inc_angles = None

    def to_points(self):
        pts = self.vps + self.depth * self.dirs
        return torch.as_tensor(pts, dtype=torch.float32)

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

    def estimate_normals(self, knn=15):
        pcd = o3d.geometry.PointCloud()
        pts = self.to_points()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals()
        pcd.normalize_normals()
        pcd.orient_normals_consistent_tangent_plane(k=knn)
        self.normals = torch.as_tensor(pcd.normals, dtype=torch.float32)

    def estimate_incidence_angles(self):
        if self.normals is None:
            self.estimate_normals()
        coss = torch.matmul(self.normals.view(-1, 3), -self.dirs.view(-1, 3).T)[:, 0].unsqueeze(-1)
        self.inc_angles = torch.as_tensor(torch.arccos(coss), dtype=torch.float32)  # shape = (N, 1)

    def visualize(self, normals=False):
        cloud = self.to_points().detach().cpu()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        if self.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(self.normals.detach().cpu())
        o3d.visualization.draw_geometries([pcd], point_show_normal=normals)

    def to(self, device=torch.device('cuda:0')):
        if self.depth is not None:
            self.depth = self.depth.to(device)
        if self.dirs is not None:
            self.dirs = self.dirs.to(device)
        if self.vps is not None:
            self.vps = self.vps.to(device)
        if self.normals is not None:
            self.normals = self.normals.to(device)
        if self.inc_angles is not None:
            self.inc_angles = self.inc_angles.to(device)
        return self

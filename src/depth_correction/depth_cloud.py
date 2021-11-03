from __future__ import absolute_import, division, print_function
from .nearest_neighbors import nearest_neighbors
from .utils import map_colors, timing
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

    def __init__(self, vps=None, dirs=None, depth=None,
                 points=None, cov=None, normals=None, inc_angles=None,
                 eigvals=None, trace=None):
        """Create depth cloud from viewpoints, directions, and depth.

        Dependent fields are not updated automatically, they can be passed in
        by the caller and/or they can also by manually recomputed.

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

        # Dependent features
        self.points = points
        # self.update_points()

        # Neighborhood features
        self.neighbors = None
        self.dist = None
        self.cov = None
        self.normals = normals
        self.inc_angles = None
        self.eigvals = None
        self.trace = None

    def size(self):
        return self.dirs.shape[0]

    def to_points(self):
        pts = self.vps + self.depth * self.dirs
        return pts

    def update_points(self):
        self.points = self.to_points()

    def transform(self, T):
        assert isinstance(T, torch.Tensor)
        assert T.shape == (4, 4)
        R = T[:3, :3]
        t = T[:3, 3:]
        vps = torch.matmul(self.vps, R.transpose(-1, -2)) + t.transpose(-1, -2)
        dirs = torch.matmul(self.dirs, R.transpose(-1, -2))
        dc = DepthCloud(vps, dirs, self.depth)
        return dc

    def __getitem__(self, item):
        vps = self.vps[item]
        dirs = self.dirs[item]
        depth = self.depth[item]
        dc = DepthCloud(vps, dirs, depth)
        return dc

    def __add__(self, other):
        return DepthCloud.concatenate([self, other], dependent=True)

    @timing
    def update_neighbors(self, k=None, r=None):
        assert self.points is not None
        self.dist, self.neighbors = nearest_neighbors(self.points, self.points, k=k, r=r)

    def neighbor_fun(self, fun):
        assert self.points is not None
        assert self.neighbors is not None
        assert callable(fun)

        result = []
        for i in range(self.size()):
            p = torch.index_select(self.points, 0, torch.tensor(self.neighbors[i]))
            q = self.points[i:i + 1]
            out = fun(p, q)
            result.append(out)

        return result

    def cov_fun(self):
        assert self.cov is not None

    @timing
    def update_cov(self, correction=1, invalid=0.0):
        invalid = torch.full((3, 3), invalid)
        fun = lambda p, q: torch.cov(p.transpose(-1, -2), correction=correction) if p.shape[0] >= 2 else invalid
        cov = self.neighbor_fun(fun)
        cov = torch.stack(cov)
        self.cov = cov

    def compute_eigvals(self, invalid=0.0):
        assert self.cov is not None

        # Serial eigvals.
        # invalid = torch.tensor(invalid)
        # for i in range(self.size()):
        #     self.cov[i]
        # fun = lambda cov: torch.linalg.eigvalsh(torch.cov(p.transpose(-1, -2))) if p.shape[0] >= 3 else invalid
        # eigvals = self.cov_fun(fun)
        # eigvals = torch.stack(eigvals)

        # Parallel eigvals.
        # Degenerate cov matrices must be skipped to avoid exception.
        eigvals = torch.full([self.size(), 3], invalid, dtype=self.cov.dtype)
        # eigvals = torch.linalg.eigvalsh(self.cov)
        valid = [i for i, n in enumerate(self.neighbors) if len(n) >= 3]
        eigvals[valid] = torch.linalg.eigvalsh(self.cov[valid])

        return eigvals

    @timing
    def update_eigvals(self, invalid=0.0):
        self.eigvals = self.compute_eigvals()

    def estimate_normals(self, knn=15):
        pcd = o3d.geometry.PointCloud()
        pts = self.to_points()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals()
        pcd.normalize_normals()
        pcd.orient_normals_consistent_tangent_plane(k=knn)
        self.normals = torch.as_tensor(pcd.normals)

    def to_point_cloud(self, colors=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.to_points())
        if self.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(self.normals)

        if colors == 'min_eigval':
            assert self.eigvals is not None
            colormap = torch.tensor([[0., 1., 0.], [1., 0., 0.]], dtype=torch.float64)
            vals = self.eigvals[:, :1]
            min_val, max_val = torch.quantile(vals, torch.tensor([0., 0.99], dtype=vals.dtype))
            print('min, max: %.6g, %.6g' % (min_val, max_val))
            colors = map_colors(vals, colormap, min_value=min_val, max_value=max_val)
            # print(colors.shape)
            pcd.colors = o3d.utility.Vector3dVector(colors.detach().numpy())
            # o3d.visualization.draw_geometries([pcd])

        return pcd

    def visualize(self, normals=False, colors=None):
        pcd = self.to_point_cloud(colors=colors)
        o3d.visualization.draw_geometries([pcd], point_show_normal=normals)

    @staticmethod
    def concatenate(depth_clouds, dependent=False):
        vps = torch.concat([dc.vps for dc in depth_clouds])
        dirs = torch.concat([dc.dirs for dc in depth_clouds])
        depth = torch.concat([dc.depth for dc in depth_clouds])

        # Dependent fields
        cov = None
        normals = None
        inc_angles = None
        eigvals = None
        trace = None
        if dependent:
            cov = torch.concat([dc.cov for dc in depth_clouds])
            normals = torch.concat([dc.normals for dc in depth_clouds])
            inc_angles = torch.concat([dc.inc_angles for dc in depth_clouds])
            eigvals = torch.concat([dc.eigvals for dc in depth_clouds])
            trace = torch.concat([dc.trace for dc in depth_clouds])

        dc = DepthCloud(vps, dirs, depth,
                        cov=cov, normals=normals,
                        inc_angles=inc_angles,
                        eigvals=eigvals,
                        trace=trace)

        return dc

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
            vps = torch.zeros((pts.shape[0], 3))
        elif isinstance(vps, np.ndarray):
            vps = torch.tensor(vps)
        assert isinstance(vps, torch.Tensor)
        # print(pts.shape)
        # print(vps.shape)
        # assert vps.shape == pts.shape or tuple(vps.shape) == (3,)
        assert vps.shape == pts.shape

        dirs = pts - vps
        assert dirs.shape == pts.shape

        depth = dirs.norm(dim=-1, keepdim=True)
        assert depth.shape[0] == pts.shape[0]

        # TODO: Handle invalid points (zero depth).
        # print(dirs.shape)
        # print(depth.shape)
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

from __future__ import absolute_import, division, print_function
from .nearest_neighbors import nearest_neighbors
from .utils import map_colors, timing
import numpy as np
from numpy.lib.recfunctions import merge_arrays, structured_to_unstructured, unstructured_to_structured
import rospy
import torch
import open3d as o3d  # used for normals estimation and visualization


__all__ = [
    'DepthCloud'
]


# @timing
def covs(x, obs_axis=-2, var_axis=-1, center=True, correction=True, weights=None):
    """Create covariance matrices from multiple samples."""
    assert isinstance(x, torch.Tensor)
    assert obs_axis != var_axis
    assert weights is None or isinstance(weights, torch.Tensor)

    # Use sum of provided weights or number of observation for normalization.
    if weights is not None:
        w = weights.sum(dim=obs_axis, keepdim=True)
    else:
        w = x.shape[obs_axis]

    # Center the points if requested.
    if center:
        if weights is not None:
            xm = (weights * x).sum(dim=obs_axis, keepdim=True) / w
        else:
            xm = x.mean(dim=obs_axis, keepdim=True)
        xc = x - xm
    else:
        xc = x

    # Construct possibly weighted xx = x * x^T.
    var_axis_2 = var_axis + 1 if var_axis >= 0 else var_axis - 1
    xx = xc.unsqueeze(var_axis) * xc.unsqueeze(var_axis_2)
    if weights is not None:
        xx = weights.unsqueeze(var_axis) * xx

    # Compute weighted average of x * x^T to get cov.
    if obs_axis < var_axis and obs_axis < 0:
        obs_axis -= 1
    elif obs_axis > var_axis and obs_axis > 0:
        obs_axis += 1
    xx = xx.sum(dim=obs_axis)
    if correction:
        w = w - 1
    xx = xx / w

    return xx


class DepthCloud(object):

    # Fields kept during slicing cloud[index].
    sliced_fields = ['vps', 'dirs', 'depth',
                     'points',
                     'cov', 'eigvals', 'eigvecs', 'normals', 'inc_angles', 'trace',
                     'loss']

    def __init__(self, vps=None, dirs=None, depth=None,
                 points=None, mean=None, cov=None, eigvals=None, eigvecs=None,
                 normals=None, inc_angles=None, trace=None,
                 loss=None):
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

        self.vps = torch.as_tensor(vps, dtype=torch.float64)
        self.dirs = torch.as_tensor(dirs, dtype=torch.float64)
        self.depth = torch.as_tensor(depth, dtype=torch.float64)

        # Dependent features
        self.points = points
        # self.update_points()

        # Nearest neighbor graph
        self.neighbors = None
        self.dist = None
        # Expanded neighbor points
        self.neighbor_points = None
        # Expanded nearest neighbor weights (some may be invalid)
        self.weights = None

        # Neighborhood features
        self.mean = mean
        self.cov = cov
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.normals = normals
        self.inc_angles = inc_angles
        self.trace = trace

        self.loss = loss

    def copy(self):
        dc = DepthCloud(vps=self.vps, dirs=self.dirs, depth=self.depth)
        dc.neighbors = self.neighbors
        dc.dist = self.dist
        return dc

    def deepcopy(self):
        # TODO: deepcopy?
        dc = DepthCloud(vps=self.vps, dirs=self.dirs, depth=self.depth,
                        points=self.points, cov=self.cov, eigvals=self.eigvals, eigvecs=self.eigvecs,
                        normals=self.normals, inc_angles=self.inc_angles, trace=self.trace)
        dc.neighbors = self.neighbors
        dc.dist = self.dist
        return dc

    def size(self):
        return self.dirs.shape[0]

    def to_points(self):
        pts = self.vps + self.depth * self.dirs
        return pts

    def update_points(self):
        self.points = self.to_points()

    def get_points(self):
        if self.points is None:
            self.update_points()
        return self.points

    def transform(self, T):
        assert isinstance(self.vps, torch.Tensor)
        assert isinstance(self.dirs, torch.Tensor)
        assert isinstance(T, torch.Tensor)
        assert T.shape == (4, 4)
        R = T[:3, :3]
        # print('det(R) = ', torch.linalg.det(R))
        t = T[:3, 3:]
        vps = torch.matmul(self.vps, R.transpose(-1, -2)) + t.transpose(-1, -2)
        dirs = torch.matmul(self.dirs, R.transpose(-1, -2))
        dc = DepthCloud(vps, dirs, self.depth)
        return dc

    def __getitem__(self, item):
        # TODO: Allow slicing neighbors etc. (need squeezing).
        kwargs = {}
        for f in DepthCloud.sliced_fields:
            x = getattr(self, f)
            if x is not None:
                kwargs[f] = x[item]
        dc = DepthCloud(**kwargs)
        return dc

    def __add__(self, other):
        return DepthCloud.concatenate([self, other], dependent=True)

    # @timing
    def update_neighbors(self, k=None, r=None):
        assert self.points is not None
        self.dist, self.neighbors = nearest_neighbors(self.points, self.points, k=k, r=r)
        self.weights = (self.neighbors >= 0).float()[..., None]

    # @timing
    def filter_neighbors_normal_angle(self, max_angle):
        assert isinstance(self.neighbors, (list, np.ndarray))
        assert isinstance(self.dist, (type(None), list, np.ndarray))
        if isinstance(self.neighbors, np.ndarray):
            self.neighbors = list(self.neighbors)
            # print('Neighbors converted to list to allow variable number of items.')
        if isinstance(self.dist, np.ndarray):
            self.dist = list(self.dist)
        assert isinstance(self.neighbors, list)
        assert isinstance(self.normals, torch.Tensor)
        assert isinstance(max_angle, float) and max_angle >= 0.0

        min_cos = np.cos(max_angle)
        n_kept = 0
        n_total = 0

        for i in range(self.size()):
            p = torch.index_select(self.normals, 0, torch.tensor(self.neighbors[i]))
            q = self.normals[i:i + 1]
            cos = (p * q).sum(dim=-1)
            keep = cos >= min_cos
            n_kept += keep.sum().item()
            n_total += keep.numel()
            # self.neighbors[i] = self.neighbors[i][keep]
            # print(type(self.neighbors[i]))
            self.neighbors[i] = [n for n, k in zip(self.neighbors[i], keep) if k]
            assert isinstance(self.neighbors[i], list)
            if self.dist is not None:
                self.dist[i] = self.dist[i][keep]
        print('%i / %i = %.1f %% neighbors kept in average (normals angle <= %.3f).'
              % (n_kept, n_total, 100 * n_kept / n_total, max_angle))

    def neighbor_fun(self, fun):
        assert self.points is not None
        assert self.neighbors is not None
        assert callable(fun)

        empty = torch.zeros((0, 3), device=self.points.device)
        result = []
        for i in range(self.size()):
            if len(self.neighbors[i]) > 0:
                # p = torch.index_select(self.points, 0, torch.as_tensor(self.neighbors[i]))
                if isinstance(self.neighbors, torch.Tensor):
                    nn = self.neighbors[i]
                    nn = nn[nn >= 0]
                else:
                    nn = torch.as_tensor(self.neighbors[i], device=self.points.device)
                p = torch.index_select(self.points, 0, nn)
            else:
                p = empty
            q = self.points[i:i + 1]
            out = fun(p, q)
            result.append(out)

        return result

    # @timing
    def update_mean(self, invalid=0.0):
        w = self.weights.sum(dim=(-2, -1))[..., None]
        mean = (self.weights * self.get_neighbor_points()).sum(dim=-2) / w
        self.mean = mean
        return

        invalid = torch.full((1, 3), invalid, device=self.points.device)
        fun = lambda p, q: p.mean(dim=0) if p.shape[0] >= 1 else invalid
        mean = self.neighbor_fun(fun)
        mean = torch.stack(mean)
        self.mean = mean

    def compute_neighbor_points(self):
        return self.get_points()[self.neighbors]

    def update_neighbor_points(self):
        self.neighbor_points = self.compute_neighbor_points()

    def get_neighbor_points(self):
        if self.neighbor_points is None:
            self.update_neighbor_points()
        return self.neighbor_points

    def neighbor_points(self):
        pts = self.get_points()
        nn = pts[self.neighbors]
        rospy.loginfo('Expanded neighbors: %s', nn.shape)
        return nn

    # @timing
    def update_cov(self, correction=1, invalid=0.0):
        cov = covs(self.get_neighbor_points(), weights=self.weights)
        self.cov = cov
        return

        invalid = torch.full((3, 3), invalid, device=self.points.device)
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

    # @timing
    def update_eigvals(self):
        self.eigvals = self.compute_eigvals()

    # @timing
    def compute_eig(self, invalid=0.0):
        assert self.cov is not None

        # FIXME: Fast eigh cuda implementation?
        # Avoid slow eigh cuda implementation.
        device = self.cov.device
        device = torch.device('cpu')
        eigvals = torch.full([self.size(), 3], invalid, dtype=self.cov.dtype, device=device)
        eigvecs = torch.full([self.size(), 3, 3], invalid, dtype=self.cov.dtype, device=device)
        # Degenerate cov matrices must be skipped to avoid exception.
        # valid = [i for i, n in enumerate(self.neighbors) if len(n) >= 3]
        valid = self.weights.sum(dim=(-2, -1)).to(device) >= 3
        eigvals[valid], eigvecs[valid] = torch.linalg.eigh(self.cov.to(device)[valid])

        # return eigvals, eigvecs
        return eigvals.to(self.cov.device), eigvecs.to(self.cov.device)

    # @timing
    def update_eig(self):
        self.eigvals, self.eigvecs = self.compute_eig()

    def orient_normals(self):
        assert isinstance(self.dirs, torch.Tensor)
        assert isinstance(self.normals, torch.Tensor)
        # cos = self.dirs.dot(self.normals)
        cos = (self.dirs * self.normals).sum(dim=-1)
        sign = torch.sign(cos)[..., None]
        self.normals = - sign * self.normals

    def update_normals(self):
        assert self.eigvecs is not None
        # TODO: Keep grad?
        with torch.no_grad():
            self.normals = self.eigvecs[..., 0]
        self.orient_normals()

    def update_incidence_angles(self):
        assert self.dirs is not None
        assert self.normals is not None
        inc_angles = torch.arccos(-(self.dirs * self.normals).sum(dim=-1)).unsqueeze(-1)
        self.inc_angles = inc_angles

    # @timing
    def update_features(self):
        self.update_mean()
        self.update_cov()
        # self.update_eigvals()
        self.update_eig()
        self.update_normals()
        # Keep incidence angles from the original observations?
        self.update_incidence_angles()

    # @timing
    def update_all(self, k=None, r=None):
        self.update_points()
        self.update_neighbors(k=k, r=r)
        self.update_features()

    def get_colors(self, colors='z'):
        assert (isinstance(colors, torch.Tensor)
                or colors in ('inc_angles', 'loss', 'min_eigval', 'z'))

        if isinstance(colors, torch.Tensor):
            vals = colors
        elif colors == 'inc_angles':
            assert self.inc_angles is not None
            vals = self.inc_angles
        elif colors == 'loss':
            assert self.loss is not None
            vals = self.loss
        elif colors == 'min_eigval':
            assert self.eigvals is not None
            vals = self.eigvals[:, :1]
        elif colors == 'z':
            assert self.points is not None
            vals = self.points[:, 2:]
        else:
            raise ValueError("Something's wrong.")

        assert isinstance(vals, torch.Tensor)
        vals = vals.detach()

        # min_val, max_val = torch.quantile(vals, torch.tensor([0.01, 0.99], dtype=vals.dtype))
        # min_val, max_val = vals.min(), vals.max()
        valid = torch.isfinite(vals)
        # min_val, max_val = vals[valid].min(), vals[valid].max()
        q = torch.tensor([0.01, 0.99], dtype=vals.dtype, device=vals.device)
        min_val, max_val = torch.quantile(vals[valid], q)
        print('min, max: %.6g, %.6g' % (min_val, max_val))
        # colormap = torch.tensor([[0., 1., 0.], [1., 0., 0.]], dtype=torch.float64)
        # colors = map_colors(vals, colormap, min_value=min_val, max_value=max_val)
        colors = map_colors(vals, min_value=min_val, max_value=max_val)
        return colors

    def to_point_cloud(self, colors=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.get_points().detach().cpu())
        if self.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(self.normals.detach().cpu())

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(self.get_colors(colors))

        return pcd

    def visualize(self, window_name='Depth Correction', normals=False, colors=None):
        pcd = self.to_point_cloud(colors=colors)
        o3d.visualization.draw_geometries([pcd], window_name=window_name, point_show_normal=normals)
        # def cb():
        #     pcd =
        #     vis.update_geometry(geometry)
        #     vis.poll_events()
        #     vis.update_renderer()
        # o3d.visualization.draw_geometries_with_key_callbacks([pcd], window_name=window_name, {ord('c'): cb})

    def to_structured_array(self):
        pts = unstructured_to_structured(np.asarray(self.get_points().detach().cpu().numpy(), dtype=np.float32),
                                         names=list('xyz'))
        vps = unstructured_to_structured(np.asarray(self.vps.detach().cpu().numpy(), dtype=np.float32),
                                         names=['vp_%s' % f for f in 'xyz'])
        parts = [pts, vps]
        if self.normals is not None:
            normals = unstructured_to_structured(np.asarray(self.normals.detach().cpu().numpy(), dtype=np.float32),
                                                 names=['normal_%s' % f for f in 'xyz'])
            parts.append(normals)
        if self.loss is not None:
            loss = unstructured_to_structured(np.asarray(self.loss.detach().cpu().numpy(), dtype=np.float32),
                                              names=['loss'])
            parts.append(loss)
        cloud = merge_arrays(parts, flatten=True)
        return cloud

    @staticmethod
    def concatenate(depth_clouds, dependent=False):
        # TODO: Concatenate neighbors and dist, shift indices as necessary.
        fields = DepthCloud.sliced_fields if dependent else ['vps', 'dirs', 'depth']
        kwargs = {}
        for f in fields:
            xs = [getattr(dc, f) for dc in depth_clouds]
            if all([x is not None for x in xs]):
                kwargs[f] = torch.concat(xs)
        dc = DepthCloud(**kwargs)

        return dc

    @staticmethod
    def from_structured_array(arr):
        """Create depth cloud from points """
        assert isinstance(arr, np.ndarray)
        pts = structured_to_unstructured(arr[['x', 'y', 'z']])
        if 'vp_x' in arr.dtype.names:
            vps = structured_to_unstructured(arr[['vp_%s' % f for f in 'xyz']])
        else:
            vps = None
        return DepthCloud.from_points(pts, vps)

    @staticmethod
    def from_points(pts, vps=None):
        """Create depth cloud from points and viewpoints.

        :param pts: Points as ...-by-3 tensor,
                or structured array with 'x', 'y', 'z', 'vp_x', 'vp_y', 'vp_z' fields.
        :param vps: Viewpoints as ...-by-3 tensor, or None for zero vector.
        :return:
        """
        if pts.dtype.names:
            # vps = structured_to_unstructured(pts[['vp_%s' % f for f in 'xyz']])
            # pts = structured_to_unstructured(pts[['x', 'y', 'z']])
            return DepthCloud.from_structured_array(pts)

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
        pts = self.to_points().detach().cpu()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals()
        pcd.normalize_normals()
        pcd.orient_normals_consistent_tangent_plane(k=knn)
        self.normals = torch.as_tensor(pcd.normals, dtype=torch.float64)

    def estimate_incidence_angles(self):
        if self.normals is None:
            self.estimate_normals()
        coss = torch.matmul(self.normals.view(-1, 3), -self.dirs.view(-1, 3).T)[:, 0].unsqueeze(-1)
        self.inc_angles = torch.as_tensor(torch.arccos(coss), dtype=torch.float64)  # shape = (N, 1)

    def to_mesh(self, colors=None):
        if self.normals is None:
            self.estimate_normals()
        if colors is None:
            colors = torch.rand(size=self.normals.shape)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.to_points().detach().cpu())
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.normals = o3d.utility.Vector3dVector(self.normals.detach().cpu())
        mesh_o3d, _ = \
            o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1,
                                                                      linear_fit=False)
        # add a cropping step to clean unwanted artifacts
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh_o3d = mesh_o3d.crop(bbox)
        return mesh_o3d

    def to_pytorch3d_mesh(self):  # -> pytorch3d.structures.meshes.Meshes
        mesh_o3d = self.to_mesh()
        # convert to pytorch3d Mesh
        from pytorch3d.structures import Meshes
        from pytorch3d.io import load_obj
        o3d.io.write_triangle_mesh("/tmp/poisson_mesh.obj", mesh_o3d)
        # We read the target 3D model using load_obj
        verts, faces, _ = load_obj("/tmp/poisson_mesh.obj")
        # We construct a Meshes structure for the target mesh
        mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
        return mesh

    def to(self, device=torch.device('cuda:0')):
        for f in DepthCloud.sliced_fields:
            x = getattr(self, f)
            if x is not None:
                x = x.to(device)
                setattr(self, f, x)
        return self

    def cpu(self):
        return self.to(torch.device('cpu'))

    def gpu(self):
        return self.to(torch.device('cuda:0'))

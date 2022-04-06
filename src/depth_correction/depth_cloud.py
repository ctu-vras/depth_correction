from __future__ import absolute_import, division, print_function
from .nearest_neighbors import nearest_neighbors
from .utils import map_colors, timing
import numpy as np
from numpy.lib.recfunctions import merge_arrays, structured_to_unstructured, unstructured_to_structured
from ros_numpy import msgify
from sensor_msgs.msg import PointCloud2
import torch
import open3d as o3d  # used for normals estimation and visualization


__all__ = [
    'covs',
    'DepthCloud',
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
    w = w.clamp(1e-6, None)
    xx = xx / w

    return xx


class DepthCloud(object):
    """Point cloud constructed from viewpoints, directions, and depths.

    In-place operation are avoided, in general, so that using a shallow copy
    is enough to create a snapshot.
    """
    # Fields kept during slicing cloud[index].
    sliced_fields = ['vps', 'dirs', 'depth',
                     'points', 'mean', 'cov', 'eigvals', 'eigvecs',
                     'normals', 'inc_angles', 'trace',
                     'loss', 'mask']
    not_sliced_fields = ['neighbors', 'weights', 'distances', 'neighbor_points']
    all_fields = sliced_fields + not_sliced_fields

    def __init__(self, vps=None, dirs=None, depth=None,
                 points=None, mean=None, cov=None, eigvals=None, eigvecs=None,
                 normals=None, inc_angles=None, trace=None,
                 loss=None, mask=None,
                 neighbors=None, distances=None, neighbor_points=None, weights=None):
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

        self.vps = vps
        self.dirs = dirs
        self.depth = depth

        # Dependent features
        self.points = points

        # Nearest neighbor graph
        self.neighbors = None
        # Expanded nearest neighbor weights (some may be invalid)
        self.weights = None
        self.distances = None
        # Expanded neighbor points
        self.neighbor_points = None

        # Neighborhood features
        self.mean = mean
        self.cov = cov
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.normals = normals
        self.inc_angles = inc_angles
        self.trace = trace

        self.loss = loss
        self.mask = mask

    def copy(self):
        """Create shallow copy of the cloud."""
        kwargs = {}
        for f in DepthCloud.all_fields:
            x = getattr(self, f)
            if x is not None:
                kwargs[f] = x
        dc = DepthCloud(**kwargs)
        return dc

    def clone(self):
        """Create deep copy of the cloud.

        Gradients are still propagated if detach is not called."""
        kwargs = {}
        for f in DepthCloud.all_fields:
            x = getattr(self, f)
            if x is not None:
                kwargs[f] = x.clone()
        dc = DepthCloud(**kwargs)
        return dc

    def size(self):
        return self.dirs.shape[0]

    def __len__(self):
        return self.size()

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
        # TODO: Optionally, transform all / specified fields.
        assert isinstance(self.vps, torch.Tensor)
        assert isinstance(self.dirs, torch.Tensor)
        assert isinstance(T, torch.Tensor)
        assert T.shape == (4, 4)
        T = T.to(dtype=self.vps.dtype)
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

    def update_distances(self):
        assert self.neighbors is not None
        x = self.get_points()
        d = torch.linalg.norm(x.unsqueeze(dim=1) - x[self.neighbors], dim=-1)
        self.distances = d

    # @timing
    def update_neighbors(self, k=None, r=None):
        assert self.points is not None
        self.distances, self.neighbors = nearest_neighbors(self.get_points(), self.get_points(), k=k, r=r)
        self.weights = (self.neighbors >= 0).float()[..., None]

    # @timing
    def filter_neighbors_normal_angle(self, max_angle):
        # TODO: Batch computation using neighbors tensor.
        assert isinstance(self.neighbors, (list, np.ndarray))
        assert isinstance(self.distances, (type(None), list, np.ndarray))
        if isinstance(self.neighbors, np.ndarray):
            self.neighbors = list(self.neighbors)
            # print('Neighbors converted to list to allow variable number of items.')
        if isinstance(self.distances, np.ndarray):
            self.distances = list(self.distances)
        assert isinstance(self.neighbors, list)
        assert isinstance(self.normals, torch.Tensor)
        assert isinstance(max_angle, float) and max_angle >= 0.0

        min_cos = np.cos(max_angle)
        n_kept = 0
        n_total = 0

        for i in range(self.size()):
            p = torch.index_select(self.normals, 0, torch.as_tensor(self.neighbors[i]))
            q = self.normals[i:i + 1]
            cos = (p * q).sum(dim=-1)
            keep = cos >= min_cos
            n_kept += keep.sum().item()
            n_total += keep.numel()
            # self.neighbors[i] = self.neighbors[i][keep]
            # print(type(self.neighbors[i]))
            self.neighbors[i] = [n for n, k in zip(self.neighbors[i], keep) if k]
            assert isinstance(self.neighbors[i], list)
            if self.distances is not None:
                self.distances[i] = self.distances[i][keep]
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

    # @timing
    def compute_eig(self, invalid=0.0):
        assert self.cov is not None
        # TODO: Switch to a faster cuda eigh implementation.
        # Use faster cpu eigh implementation.
        # device = self.cov.device
        device = torch.device('cpu')
        eigvals, eigvecs = torch.linalg.eigh(self.cov.to(device))
        return eigvals.to(self.cov.device), eigvecs.to(self.cov.device)

        # # Degenerate cov matrices must be skipped to avoid exception.
        # eigvals = torch.full([self.size(), 3], invalid, dtype=self.cov.dtype, device=device)
        # eigvecs = torch.full([self.size(), 3, 3], invalid, dtype=self.cov.dtype, device=device)
        # valid = self.weights.sum(dim=(-2, -1)).to(device) >= 3
        # eigvals[valid], eigvecs[valid] = torch.linalg.eigh(self.cov.to(device)[valid])
        # return eigvals.to(self.cov.device), eigvecs.to(self.cov.device)

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
    def update_all(self, k=None, r=None, keep_neighbors=False):
        self.update_points()
        if keep_neighbors:
            self.update_distances()
        else:
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
            raise ValueError("Unsupported color specification.")

        assert isinstance(vals, torch.Tensor)
        vals = vals.detach().float()
        # min_val, max_val = torch.quantile(vals, torch.tensor([0.01, 0.99], dtype=vals.dtype))
        # min_val, max_val = vals.min(), vals.max()
        valid = torch.isfinite(vals)
        # min_val, max_val = vals[valid].min(), vals[valid].max()
        q = torch.tensor([0.01, 0.99], dtype=vals.dtype, device=vals.device)
        min_val, max_val = torch.quantile(vals[valid], q)
        # print('min, max: %.6g, %.6g' % (min_val, max_val))
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
    def concatenate(clouds, dependent=False):
        # TODO: Concatenate neighbors and distances, shift indices as necessary.
        # fields = DepthCloud.sliced_fields if dependent else ['vps', 'dirs', 'depth']
        fields = DepthCloud.all_fields if dependent else ['vps', 'dirs', 'depth']
        kwargs = {}
        for f in fields:
            # Collect field values from individual clouds.
            xs = [getattr(dc, f) for dc in clouds]
            # Concatenate the values if present in all clouds.
            if all([x is not None for x in xs]):
                # Shift indices by number of points in preceding clouds.
                if f == 'neighbors':
                    sizes = [len(cloud) for cloud in clouds]
                    shift = [0] + list(np.cumsum(sizes[:-1]))
                    for i in range(len(xs)):
                        xs[i] += shift[i]

                kwargs[f] = torch.concat(xs)
            else:
                # print('Field %s not available for some of %i clouds.'
                #       % (f, len(clouds)))
                pass
        dc = DepthCloud(**kwargs)

        return dc

    @staticmethod
    def from_structured_array(arr, dtype=None):
        """Create depth cloud from points """
        assert isinstance(arr, np.ndarray)
        pts = structured_to_unstructured(arr[['x', 'y', 'z']], dtype=dtype)
        if 'vp_x' in arr.dtype.names:
            vps = structured_to_unstructured(arr[['vp_%s' % f for f in 'xyz']], dtype=dtype)
        else:
            print('Viewpoints not provided.')
            vps = None
        return DepthCloud.from_points(pts, vps)

    @staticmethod
    def from_points(pts, vps=None, dtype=None):
        """Create depth cloud from points and viewpoints.

        :param pts: Points as ...-by-3 tensor,
                or structured array with 'x', 'y', 'z', 'vp_x', 'vp_y', 'vp_z' fields.
        :param vps: Viewpoints as ...-by-3 tensor, or None for zero vector.
        :return:
        """
        if pts.dtype.names:
            return DepthCloud.from_structured_array(pts)

        if isinstance(pts, np.ndarray):
            pts = torch.as_tensor(np.asarray(pts, dtype=dtype))

        assert isinstance(pts, torch.Tensor)

        if vps is None:
            vps = torch.from_numpy(np.zeros([pts.shape[0], 3], dtype=dtype))
        elif isinstance(vps, np.ndarray):
            vps = torch.as_tensor(np.asarray(vps, dtype=dtype))
        assert isinstance(vps, torch.Tensor)
        assert vps.shape == pts.shape

        dirs = pts - vps
        assert dirs.shape == pts.shape

        depth = dirs.norm(dim=-1, keepdim=True)
        assert depth.shape[0] == pts.shape[0]

        # TODO: Handle invalid points (zero depth).
        dirs = dirs / depth
        assert dirs.dtype == vps.dtype
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

    def to_msg(dc, frame_id=None, stamp=None):
        pc_msg = msgify(PointCloud2, dc.to_structured_array())
        if frame_id:
            pc_msg.header.frame_id = frame_id
        if stamp:
            pc_msg.header.stamp = stamp
        return pc_msg

    def to(self, device=torch.device('cuda:0'), dtype=None, float_type=None, int_type=None):
        # for f in DepthCloud.sliced_fields:
        for f in DepthCloud.all_fields:
            x = getattr(self, f)
            if x is not None:
                if (float_type and x.dtype.is_floating_point) \
                        or (int_type and not x.dtype.is_floating_point) \
                        or (dtype and dtype.is_floating_point == x.dtype.is_floating_point):
                    x_type = dtype or float_type or int_type
                else:
                    x_type = None

                # x_dtype =  dtype.is_floating_point == x.dtype.is_floating_point
                x = x.to(device=device, dtype=x_type)
                setattr(self, f, x)
        return self

    def cpu(self):
        return self.to(torch.device('cpu'))

    def gpu(self):
        return self.to(torch.device('cuda:0'))

    def type(self, dtype=None):
        if dtype is None:
            assert self.vps.dtype == self.dirs.dtype == self.depth.dtype
            return self.vps.dtype
        else:
            for f in DepthCloud.all_fields:
                x = getattr(self, f)
                if x is not None and dtype.is_floating_point == x.dtype.is_floating_point:
                    x = x.type(dtype)
                    setattr(self, f, x)
            return self

    def float(self):
        return self.type(torch.float32)

    def double(self):
        return self.type(torch.float64)

    def detach(self):
        for f in DepthCloud.all_fields:
            x = getattr(self, f)
            if x is not None:
                x = x.detach()
                setattr(self, f, x)
        return self

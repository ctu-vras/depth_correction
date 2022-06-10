from __future__ import absolute_import, division, print_function
from .config import Config
from .configurable import ValueEnum
from .depth_cloud import DepthCloud
from .preproc import filtered_cloud
from .model import BaseModel
from .utils import cached, hashable, timing, transform, transform_inv, load_mesh
from copy import copy
from data.asl_laser import read_poses
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.recfunctions import merge_arrays, structured_to_unstructured, unstructured_to_structured
import open3d as o3d
import os
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import interpolate_face_attributes, sample_points_from_meshes
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    look_at_view_transform,
    MeshRasterizer,
    MeshRenderer,
    OrthographicCameras,
    PerspectiveCameras,
    RasterizationSettings,
)
from tf.transformations import euler_matrix
import torch


def box_point_cloud(size=(1.0, 1.0, 0.0), density=100.0, rng=np.random.default_rng()):
    size = np.asarray(size).reshape((1, 3))
    measure = np.prod([s for s in size.flatten() if s])
    n_pts = int(np.ceil(measure * density))
    x = size * rng.uniform(-0.5, 0.5, (n_pts, 3))
    return x


class GroundPlaneDataset(object):
    def __init__(self, name=None, n=10, size=(5.0, 5.0, 0.0), step=1.0, height=1.0, density=100.0, **kwargs):
        """Dataset composed of multiple measurements of ground plane.

        :param n: Number of viewpoints.
        :param size: Local clous size.
        :param step: Distance between neighboring viewpoints (along x-axis).
        :param height: Sensor height above ground plane.
        :param density: Point density in unit volume/area.
        """
        if name:
            parts = name.split('/')
            if len(parts) == 2:
                assert parts[0] == 'ground_plane'
                name = parts
            # TODO: Parse other params from name.
            if isinstance(name, str):
                n = int(name)

        self.n = n
        self.size = size
        self.step = step
        self.height = height
        self.density = density
        self.ids = list(range(self.n))

    # @cached
    def local_cloud(self, id):
        rng = np.random.default_rng(id)
        pts = box_point_cloud(size=self.size, density=self.density, rng=rng)
        pts[:, 2] -= self.height
        normals = np.zeros_like(pts)
        normals[:, 2] = 1.0
        pts = unstructured_to_structured(pts, names=['x', 'y', 'z'])
        normals = unstructured_to_structured(normals, names=['normal_x', 'normal_y', 'normal_z'])
        cloud = merge_arrays([pts, normals], flatten=True)
        return cloud

    def cloud_pose(self, id):
        pose = np.eye(4)
        pose[0, 3] = id * self.step
        pose[2, 3] = self.height
        return pose

    def __getitem__(self, i):
        if isinstance(i, int):
            id = self.ids[i]
            cloud = self.local_cloud(id)
            pose = self.cloud_pose(id)
            return cloud, pose

        ds = GroundPlaneDataset(n=self.n, size=self.size, step=self.step, height=self.height, density=self.density)
        if isinstance(i, (list, tuple)):
            ds.ids = [self.ids[j] for j in i]
        else:
            assert isinstance(i, slice)
            ds.ids = self.ids[i]
        return ds

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class OpenBoxDataset(object):
    def __init__(self, name=None, n=None, size=None, height=None, density=None, **kwargs):
        """Open box dataset with viewpoints inside along a circle.

        :param name: Name of the sequence.
        :param n: Number of viewpoints.
        :param size: Open box size.
        :param height: Sensor height above ground plane.
        :param density: Point density in unit area.
        """
        if name:
            parts = name.split('/')
            if len(parts) == 2:
                assert parts[0] == 'open_box'
                name = parts[1]
            # Parse other params from name.
            if isinstance(name, str):
                parts = name.split('_')
                if 'n' in parts:
                    assert n is None
                    i = parts.index('n')
                    n = int(parts[i + 1])
                if 'size' in parts:
                    assert size is None
                    i = parts.index('size')
                    size = [int(s) for s in parts[i + 1:i + 4]]
                if 'height' in parts:
                    assert height is None
                    i = parts.index('height')
                    height = float(parts[i + 1])
                if 'density' in parts:
                    assert density is None
                    i = parts.index('density')
                    density = float(parts[i + 1])
        for k, v in kwargs.items():
            print('Unused key-value argument: %s: %s.' % (k, v))

        # Fill defaults.
        if n is None:
            n = 10
        if size is None:
            size = 10.0, 10.0, 5.0
        if height is None:
            height = 1.0
        if density is None:
            density = 100.0

        self.n = n
        self.size = size
        self.height = height
        self.density = density
        self.ids = list(range(self.n))

    # @cached
    def local_cloud(self, id):
        rng = np.random.default_rng(id)
        # pts = box_point_cloud(size=self.size, density=self.density, rng=rng)
        pts = []
        normals = []
        # ground plane
        pts.append(box_point_cloud(size=(self.size[0], self.size[1], 0.0), density=self.density, rng=rng))
        normals.append(np.repeat(np.array([[0.0, 0.0, 1.0]]), pts[-1].shape[0], axis=0))
        # side -y
        pts.append(box_point_cloud(size=(self.size[0], 0.0, self.size[2]), density=self.density, rng=rng)
                   + np.array([[0.0, -self.size[1] / 2, self.size[2] / 2]]))
        normals.append(np.repeat(np.array([[0.0, 1.0, 0.0]]), pts[-1].shape[0], axis=0))
        # side +y
        # pts.append(box_point_cloud(size=(self.size[0], 0.0, self.size[2]), density=self.density, rng=rng)
        #            + np.array([[0.0, self.size[1] / 2, self.size[2] / 2]]))
        # normals.append(np.repeat(np.array([[0.0, -1.0, 0.0]]), pts[-1].shape[0], axis=0))
        # side -x
        pts.append(box_point_cloud(size=(0.0, self.size[1], self.size[2]), density=self.density, rng=rng)
                   + np.array([[-self.size[0] / 2, 0.0, self.size[2] / 2]]))
        normals.append(np.repeat(np.array([[1.0, 0.0, 0.0]]), pts[-1].shape[0], axis=0))
        # side +x
        # pts.append(box_point_cloud(size=(0.0, self.size[1], self.size[2]), density=self.density, rng=rng)
        #            + np.array([[self.size[0] / 2, 0.0, self.size[2] / 2]]))
        # normals.append(np.repeat(np.array([[-1.0, 0.0, 0.0]]), pts[-1].shape[0], axis=0))

        pts = np.concatenate(pts)
        normals = np.concatenate(normals)

        pts = unstructured_to_structured(pts, names=['x', 'y', 'z'])
        normals = unstructured_to_structured(normals, names=['normal_x', 'normal_y', 'normal_z'])
        cloud = merge_arrays([pts, normals], flatten=True)

        T_inv = transform_inv(self.cloud_pose(id))
        cloud = transform(T_inv, cloud)

        return cloud

    def cloud_pose(self, id):
        rng = np.random.default_rng(id)
        # rotation
        a = id * 2 * np.pi / self.n
        e = 0.1 * rng.uniform(size=(3,))
        e[2] += a
        pose = euler_matrix(*e)
        # translation
        # pose[0, 3] = np.cos(a) * self.size[0] / 3
        # pose[1, 3] = np.sin(a) * self.size[1] / 3
        # pose[2, 3] = self.height
        pose[:3, 3] = [np.cos(a) * self.size[0] / 3, np.sin(a) * self.size[1] / 3, self.height]
        pose[:3, 3] += 0.1 * rng.uniform(size=(3,))
        return pose

    def to_str(self):
        return 'open_box/n_%i_size_%i_%i_%i_height_%.3g_density_%.3g' % (self.n, *self.size, self.height, self.density)

    def __getitem__(self, i):
        if isinstance(i, int):
            id = self.ids[i]
            cloud = self.local_cloud(id)
            pose = self.cloud_pose(id)
            return cloud, pose
        ds = copy(self)
        if isinstance(i, (list, tuple)):
            ds.ids = [self.ids[j] for j in i]
        else:
            assert isinstance(i, slice)
            ds.ids = self.ids[i]
        return ds

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __str__(self):
        return self.to_str()


class BaseDataset:
    def __init__(self,
                 name: str = None,
                 n_pts: int = 10_000,
                 n_poses: int = 5,
                 height: float = 2.0,
                 size: tuple = ([-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0])):
        """BaseDataset composed of multiple measurements of a general point cloud.

        :param name: Name of the dataset
        :param n_pts: Number of points forming a global cloud.
        :param size: Local clouds size.
        :param height: Sensor height above ground plane.
        :param n_poses: Number of view points (poses of sensors, at which local clouds are measured).
        """
        self.name = name
        self.n_pts = n_pts
        self.pts = None
        self.normals = None
        self.n_poses = n_poses
        self.height = height
        self.size = size
        self.ids = range(self.n_poses)

    def __len__(self):
        """
        Denotes the total number of samples
        """
        return len(self.ids)

    def local_cloud(self, i):
        assert self.pts is not None
        rng = np.random.default_rng(i)

        mask = rng.choice(range(self.n_pts), size=self.n_pts // self.n_poses, replace=False)
        cloud = self.pts[mask]
        normals = self.normals[mask]
        assert cloud.shape == (self.n_pts // self.n_poses, 3)

        cloud = unstructured_to_structured(cloud, names=['x', 'y', 'z'])
        normals = unstructured_to_structured(normals, names=['normal_x', 'normal_y', 'normal_z'])
        cloud = merge_arrays([cloud, normals], flatten=True)

        # transform the point cloud to view point frame as it was sampled from global map
        T_inv = transform_inv(self.cloud_pose(i))
        cloud = transform(T_inv, cloud)
        return cloud

    def cloud_pose(self, i):
        rng = np.random.default_rng(i)
        pose = np.eye(4)
        for p in range(2):
            pose[p, 3] = rng.uniform(low=0.6*self.size[p][0], high=0.6*self.size[p][1])
        pose[2, 3] = self.height
        return pose

    @staticmethod
    def transform(cloud, pose):
        cloud = np.matmul(cloud, pose[:3, :3].T) + pose[:3, 3:].T
        return cloud

    def __getitem__(self, i):
        if isinstance(i, int):
            id = self.ids[i]
            return self.local_cloud(id), self.cloud_pose(id)

        ds = BaseDataset(name=self.name, n_pts=self.n_pts, n_poses=self.n_poses, size=self.size)
        ds.pts = self.pts
        if isinstance(i, (list, tuple)):
            ds.ids = [self.ids[j] for j in i]
        else:
            assert isinstance(i, slice)
            ds.ids = self.ids[i]
        return ds

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class PlaneDataset(BaseDataset):
    def __init__(self, name='plane',
                 n_pts: int = 10_000,
                 n_poses: int = 2,
                 size: tuple = ([-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0])):
        """PlaneDataset composed of multiple measurements of a ground plane.

        :param name: Name of the dataset
        :param n_pts: Number of points forming a global cloud.
        :param size: Local clouds size.
        :param n_poses: Number of view points (poses of sensors, at which local clouds are measured).
        """
        super(PlaneDataset, self).__init__(name=name, n_pts=n_pts, n_poses=n_poses, size=size)
        self.pts, self.normals = self.construct_global_cloud()

    def construct_global_cloud(self, seed=135):
        # create flat point cloud (a wall)
        np.random.seed(seed)
        pts = np.zeros((self.n_pts, 3), dtype=np.float64)
        pts[:, :2] = np.concatenate([np.random.uniform(0, self.size[0][1], size=(self.n_pts // 2, 2)),
                                     np.random.uniform(0, self.size[1][1], size=(self.n_pts // 2, 2)) +
                                     np.array([self.size[0][0], 0])])
        normals = np.zeros_like(pts)
        normals[:, 2] = 1.0
        return pts, normals

    def __getitem__(self, i):
        if isinstance(i, int):
            id = self.ids[i]
            return self.local_cloud(id), self.cloud_pose(id)

        ds = PlaneDataset(n_pts=self.n_pts, n_poses=self.n_poses, size=self.size)
        ds.pts = self.pts
        if isinstance(i, (list, tuple)):
            ds.ids = [self.ids[j] for j in i]
        else:
            assert isinstance(i, slice)
            ds.ids = self.ids[i]
        return ds


class AngleDataset(PlaneDataset):
    def __init__(self, name='angle',
                 n_pts: int = 10_000,
                 n_poses: int = 5,
                 size: tuple = ([-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]),
                 degrees: float = 0.0):
        """AngleDataset composed of multiple point cloud measurements of two intersecting planes forming an angle.

        :param name: Name of the dataset
        :param n_pts: Number of points forming a global cloud.
        :param size: Local clouds size.
        :param n_poses: Number of view points (poses of sensors, at which local clouds are measured).
        :param degrees: Angle between the two planes (around Y axis in degrees).
        """
        super(AngleDataset, self).__init__(name=name, n_pts=n_pts, n_poses=n_poses, size=size)
        self.degrees = degrees
        if self.degrees != 0.0:
            self.pts[self.n_pts // 2:] = self.rotate(self.pts[self.n_pts // 2:], origin=(0, 0, 0),
                                                     degrees=degrees, axis='Y')
            self.normals[self.n_pts // 2:] = self.rotate(self.normals[self.n_pts // 2:], origin=(0, 0, 0),
                                                         degrees=degrees, axis='Y')

    @staticmethod
    def rotate(p, origin=(0, 0, 0), degrees=0.0, axis='X'):
        angle = np.deg2rad(degrees)
        if axis == 'Z':
            R = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]])
        elif axis == 'Y':
            R = np.array([[np.cos(angle), 0, np.sin(angle)],
                          [0, 1, 0],
                          [-np.sin(angle), 0, np.cos(angle)]])
        elif axis == 'X':
            R = np.array([[1, 0, 0],
                          [0, np.cos(angle), -np.sin(angle)],
                          [0, np.sin(angle), np.cos(angle)]])
        o = np.atleast_3d(origin)
        p = np.atleast_3d(p)
        return np.squeeze((R @ (p.T - o.T) + o.T).T)

    def __getitem__(self, i):
        if isinstance(i, int):
            id = self.ids[i]
            return self.local_cloud(id), self.cloud_pose(id)

        ds = AngleDataset(n_pts=self.n_pts, n_poses=self.n_poses, size=self.size, degrees=self.degrees)
        ds.pts = self.pts
        if isinstance(i, (list, tuple)):
            ds.ids = [self.ids[j] for j in i]
        else:
            assert isinstance(i, slice)
            ds.ids = self.ids[i]
        return ds


class MeshDataset(BaseDataset):
    def __init__(self, mesh_name,
                 n_poses: int = 5,
                 size: tuple = ([-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]),
                 n_pts_to_sample: int = 10_000_000):
        """MeshDataset composed of multiple point cloud measurements of a ground truth environment represented as a mesh.

        :param mesh_name: Name of the mesh file with an extension (*.obj or *.ply).
        :param size: Local clouds size.
        :param n_poses: Number of view points (poses of sensors, at which local clouds are measured).
        :param n_pts_to_sample: Number of points to sample from the mesh to form a global cloud.
        """
        super(MeshDataset, self).__init__(name=mesh_name, n_poses=n_poses, size=size)
        if os.path.isabs(self.name):
            self.mesh_path = self.name
        else:
            self.mesh_path = os.path.join(Config().pkg_dir, 'data', 'meshes', self.name)
        if not os.path.exists(self.mesh_path):
            raise FileExistsError('Mesh file %s does not exist, download meshes from'
                                  'https://drive.google.com/drive/folders/1S3UlJ4MgNsU72PTwJku-gyHZbv3aw26Z?usp=sharing'
                                  'and place them to ./data/ folder' % self.mesh_path)
        else:
            print('Loading mesh: %s' % mesh_name)
        self.n_pts_to_sample = n_pts_to_sample
        self.pts, self.normals = self.construct_global_cloud()

    def construct_global_cloud(self, pts_src='sampled_from_mesh'):
        """
        :param pts_src: 'sampled_from_mesh': sample points from mesh, 'mesh_vertices': use mesh vertices
        """
        assert pts_src == 'sampled_from_mesh' or pts_src == 'mesh_vertices'
        mesh = load_mesh(self.mesh_path)

        pts = sample_points_from_meshes(mesh, num_samples=self.n_pts_to_sample).squeeze().numpy()
        # cropping points in a volume defined by size variable
        X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
        mask = np.logical_and(np.logical_and(np.logical_and(X >= self.size[0][0], X <= self.size[0][1]),
                                             np.logical_and(Y >= self.size[1][0], Y <= self.size[1][1])),
                              np.logical_and(Z >= self.size[2][0], Z <= self.size[2][1]))
        pts = pts[mask]
        if self.normals is None:
            print('Estimating normals for sampled from mesh point cloud ...')
            dc = DepthCloud.from_structured_array(unstructured_to_structured(pts, names=['x', 'y', 'z']))
            assert isinstance(dc, DepthCloud)
            if dc.normals is None:
                dc.update_all(k=20, r=0.2)
            normals = dc.normals.numpy()
            self.normals = normals
        assert pts.shape == self.normals.shape
        self.n_pts = len(pts)
        return pts, self.normals

    def __getitem__(self, i):
        if isinstance(i, int):
            id = self.ids[i]
            return self.local_cloud(id), self.cloud_pose(id)

        ds = BaseDataset(n_pts=self.n_pts, n_poses=self.n_poses, size=self.size)
        ds.pts = self.pts
        ds.normals = self.normals
        ds.name = self.name
        ds.n_pts_to_sample = self.n_pts_to_sample
        if isinstance(i, (list, tuple)):
            ds.ids = [self.ids[j] for j in i]
        else:
            assert isinstance(i, slice)
            ds.ids = self.ids[i]
        return ds

    def __str__(self):
        return self.name


class RenderedMeshDataset(object):

    dataset_name = 'rendered_name'
    cloud_dtype = np.dtype([(f, np.float32) for f in ('x', 'y', 'z',
                                                      'vp_x', 'vp_y', 'vp_z',
                                                      'normal_x', 'normal_y', 'normal_z')])

    def __init__(self, name, n=10, size=(64, 512), fov=(45., 360.), cache=True, device='cpu'):
        """Create rendered dataset from mesh.

        :param name: Name of the mesh (possibly with parameters), or absolute path.
        :param n: Number of poses.
        :param size: Lidar scan size, (height, width).
        :param fov: Lidar scan field of view, (vertical, horizontal).
        """
        if os.path.isabs(name):
            path = name
        else:
            parts = name.split('/')
            assert len(parts) in (1, 3)

            if len(parts) == 3:
                assert parts[0] == RenderedMeshDataset.dataset_name
                # Parse mesh name from qualified name.
                name = parts[1]
                # Parse parameters specified in name.
                params = parts[2].split('_')
                self.parse_params(params, n, size, fov)

            path = os.path.join(Config().pkg_dir, 'data', 'meshes', name)

        if not os.path.exists(path):
            raise FileExistsError('Mesh %s does not exist.' % path)

        # FIXME: Debug
        # size = (32, 128)

        assert isinstance(n, int)
        assert n > 0
        assert len(size) == 2
        assert len(fov) == 2
        assert 0. <= fov[0] <= 180.
        assert 0. <= fov[1] <= 360.
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert device in ('cpu', 'cuda') or isinstance(device, torch.device)
        self.path = path
        self.n = n
        self.size = size
        self.fov = fov
        self.cache = cache
        self.device = device

        # Rotation from mesh to ROS.
        self.mesh_to_ros = np.array([[ 0., 1., 0., 0.],
                                     [-1., 0., 0., 0.],
                                     [ 0., 0., 1., 0.],
                                     [ 0., 0., 0., 1.]])
        self.ros_to_mesh = np.linalg.inv(self.mesh_to_ros)
        # Poses are generated with x-forward, y-left, z-up.
        # Vertical FoV is centered around xy plane.
        self.ids, self.poses = read_poses(self.poses_path())
        self.poses = np.stack(self.poses)
        # print(self.poses[::100, :3, 3])
        self.poses = np.matmul(self.ros_to_mesh, self.poses)
        # self.poses = np.matmul(self.poses, self.ros_to_mesh)
        # self.poses = np.matmul(self.poses, self.mesh_to_ros)
        # self.poses = np.matmul(self.poses, self.mesh_to_ros)
        # print(self.poses[::100, :3, 3])
        self.poses[:, 2, 3] += 1.0

        print('Loading mesh: %s...' % self.path)
        # self.mesh = load_mesh(self.path)
        self.mesh = load_objs_as_meshes([self.path], device=device)

    def __getitem__(self, i):
        if isinstance(i, int):
            id = self.ids[i]
            cloud = self.local_cloud(id)
            pose = self.cloud_pose(id)
            return cloud, pose
        ds = copy(self)
        if isinstance(i, (list, tuple)):
            ds.ids = [self.ids[j] for j in i]
        elif isinstance(i, slice):
            ds.ids = self.ids[i]
        else:
            raise ValueError('Invalid index: %s.' % i)
        return ds

    def __len__(self):
        return self.poses.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def parse_params(self, params, n, size, fov):
        if 'n' in params:
            i = params.index('n')
            n = int(params[i + 1])
            print('Parsed n=%i from name.' % n)
        if 'size' in params:
            i = params.index('size')
            size = [int(x) for x in params[i + 1:i + 3]]
            print('Parsed size=(%i, %i) from name.' % size)
        if 'fov' in params:
            i = params.index('fov')
            fov = [float(x) for x in params[i + 1:i + 3]]
            print('Parsed fov=(%.3g, %.3g) from name.' % size)
        return n, size, fov

    def dataset_dir(self):
        name = '%s' % os.path.basename(self.path)
        path = os.path.join(Config().pkg_dir, 'gen', 'rendered_mesh', name)
        return path

    def poses_path(self):
        path = os.path.join(self.dataset_dir(), 'n%i' % self.n, 'poses.csv')
        return path

    def cloud_path(self, id):
        path = os.path.join(self.dataset_dir(), 'n%i_size_%i_%i_fov_%.0f_%.0f' % (self.n, *self.size, *self.fov),
                            'cloud_%i.bin' % id)
        return path

    def generate_viewpoints(self):
        """Generate viewpoints "inside" the mesh where something is visible."""
        path = self.poses_path()

        if not os.path.exists(path):
            poses = np.eye(4)[None]
            assert isinstance(poses, np.ndarray)
            poses[0, 2, 3] = 2.0
            os.makedirs(os.path.dirname(path), exist_ok=True)
            poses.tofile(path)

        assert os.path.exists(path)
        self.poses = np.fromfile(path)

    def render(self, pose):
        """Render lidar scan at selected pose."""
        pose = torch.as_tensor(pose)
        return render_lidar_cloud(self.mesh, pose, self.fov, self.size)

    def local_cloud(self, id):
        path = self.cloud_path(id)
        if self.cache and os.path.exists(path):
            cloud = np.fromfile(path, dtype=RenderedMeshDataset.cloud_dtype)
        else:
            cloud = self.render(self.poses[id])
        if self.cache and not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cloud.tofile(path)
            assert os.path.exists(path)
        return cloud

    def cloud_pose(self, id):
        # self.poses[id][:3, 3])
        return self.poses[id]


class Forwarding(object):
    def __init__(self, target):
        self.target = target

    def __getattr__(self, item):
        return getattr(self.target, item)

    def __iter__(self):
        return self.target.__iter__()

    def __len__(self):
        return len(self.target)

    def __str__(self):
        return str(self.target)


class TransformingDataset(Forwarding):
    def __init__(self, target):
        super().__init__(target)

    def transform_cloud(self, cloud):
        return cloud

    def transform_pose(self, pose):
        return pose

    def __iter__(self):
        for cloud, pose in self.target:
            yield self.transform_cloud(cloud), self.transform_pose(pose)

    def local_cloud(self, id):
        return self.transform_cloud(self.target.local_cloud(id))

    def cloud_pose(self, id):
        return self.transform_pose(self.target.cloud_pose(id))


class FilteredDataset(TransformingDataset):

    def __init__(self, dataset, cfg: Config):
        super().__init__(dataset)
        self.cfg = cfg

    def transform_cloud(self, cloud):
        cloud = filtered_cloud(cloud, self.cfg)
        return cloud


class NoisyPoseDataset(TransformingDataset):

    class Mode(metaclass=ValueEnum):
        pose = 'pose'
        common = 'common'

    def __init__(self, dataset, noise=0.0, mode=None):
        """
        :param dataset:
        :param noise: Pose noise, standard deviations for euler angles and position.
        :param mode:
        """
        assert isinstance(noise, float) or len(noise) == 6
        noise = np.asarray(noise)
        mode = mode or NoisyPoseDataset.Mode.common
        assert mode in NoisyPoseDataset.Mode
        super().__init__(dataset)
        self.noise = noise
        self.mode = mode

    def random_transform(self, seed):
        rng = np.random.default_rng(seed)
        noise_vec = self.noise * rng.normal(size=(6,))
        noise = euler_matrix(*noise_vec[:3])
        noise[:3, 3] = noise_vec[3:]
        return noise

    # TODO: Cache
    def transform_pose(self, pose):
        if self.mode == NoisyPoseDataset.Mode.pose:
            seed = abs(hash(hashable(pose)))
        elif self.mode == NoisyPoseDataset.Mode.common:
            seed = Config().random_seed
        if (self.noise != 0.0).any():
            noise = self.random_transform(seed)
            pose = np.matmul(pose, noise)
        return pose


class NoisyDepthDataset(TransformingDataset):

    def __init__(self, dataset, noise=None):
        """
        :param dataset:
        :param noise: Depth noise standard deviation.
        """
        super().__init__(dataset)
        self.noise = noise

    def transform_cloud(self, cloud):
        if self.noise:
            pts = structured_to_unstructured(cloud[['x', 'y', 'z']])
            if 'vp_x' in cloud.dtype.names:
                vps = structured_to_unstructured(cloud[['vp_x', 'vp_y', 'vp_z']])
                dirs = pts - vps
            else:
                dirs = pts - 0.0
            depth = np.linalg.norm(dirs, axis=1)
            valid = depth > 0.0
            depth = depth[valid]
            dirs = dirs[valid] / depth[:, None]
            # pts[valid] += dirs * self.noise * np.random.randn(*depth.shape)[:, None]
            seed = abs(hash(hashable(depth)))
            rng = np.random.default_rng(seed)
            pts[valid] += dirs * self.noise * rng.normal(size=depth.shape)[:, None]
            cloud[['x', 'y', 'z']] = unstructured_to_structured(pts, names=['x', 'y', 'z'])
        return cloud


class DepthBiasDataset(TransformingDataset):

    def __init__(self, dataset, model=None, cfg: Config=None):
        super().__init__(dataset)
        self.model = model
        self.cfg = cfg

    def transform_cloud(self, cloud):
        if self.model is not None:
            assert isinstance(self.model, BaseModel)
            dc = DepthCloud.from_structured_array(cloud)
            assert isinstance(dc, DepthCloud)
            if dc.normals is None:
                if self.cfg.log_filters:
                    print('Estimating normals from data for introducing depth bias.')
                dc.update_all(k=self.cfg.nn_k, r=self.cfg.nn_r)
            else:
                if self.cfg.log_filters:
                    print('Using provided normals for introducing depth bias.')
                dc.update_incidence_angles()
            dc = self.model.inverse(dc)
            pts = dc.to_points().detach().numpy()
            cloud[['x', 'y', 'z']] = unstructured_to_structured(pts, names=['x', 'y', 'z'])
        return cloud


class Subscriptable(Forwarding):
    """General sequence wrapper allowing slicing or indexing.

    >>> list(Subscriptable('qwerasdf')[::2])
    ['q', 'e', 'a', 'd']
    >>> Subscriptable('qwerasdf')[[1, 2, 4]][0]
    'w'
    >>> len(Subscriptable('qwerasdf')[[1, 2, 4]])
    3
    """
    def __init__(self, target, idx=None):
        assert len(target) >= 0
        self.target = target
        if idx is None:
            idx = list(range(len(target)))
        self.idx = idx

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.target[self.idx[i]]
        elif isinstance(i, (list, tuple)):
            idx = [self.idx[j] for j in i]
            return Subscriptable(self.target, idx)
        elif isinstance(i, slice):
            idx = self.idx[i]
            return Subscriptable(self.target, idx)
        raise ValueError('Invalid index: %s' % i)

    def __iter__(self):
        for i in self.idx:
            yield self.target[i]

    def __len__(self):
        return len(self.idx)


def dataset_by_name(name):
    parts = name.split('/')
    if len(parts) == 2:
        name = parts[0]

    if name == 'ground_plane':
        return GroundPlaneDataset
    elif name == 'open_box':
        return OpenBoxDataset
    elif name == 'angle':
        return AngleDataset
    elif '.obj' in name or '.ply' in name:
        return MeshDataset
    elif name == 'asl_laser':
        import data.asl_laser
        return getattr(data.asl_laser, 'Dataset')
    elif name == 'semantic_kitti':
        import data.semantic_kitti
        return getattr(data.semantic_kitti, 'Dataset')
    raise ValueError('Unknown dataset: %s.' % name)


def create_dataset(name, cfg: Config, **kwargs):
    Dataset = dataset_by_name(name)
    ds = Dataset(name, *cfg.dataset_args, **cfg.dataset_kwargs, **kwargs)
    ds = FilteredDataset(ds, cfg)
    try:
        ds = ds[::cfg.data_step]
    except TypeError as ex:
        ds = Subscriptable(ds)[::cfg.data_step]
    return ds


def demo():
    import matplotlib.pyplot as plt
    from depth_correction.model import model_by_name

    cfg = Config()
    cfg.data_step = 1

    # cfg.dataset_kwargs = dict(size=([-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]),
    #                           n_pts=10_000, n_poses=4, degrees=80.0)
    # ds = create_dataset(name='angle', cfg=cfg)

    cfg.dataset_kwargs = dict(size=([-10.0, 10.0], [-10.0, 10.0], [-1.0, 5.0]), n_poses=4)
    # ds = create_dataset(name='simple_cave_01.obj', cfg=cfg)
    # ds = create_dataset(name='burning_building_rubble.ply', cfg=cfg)
    ds = create_dataset(name='cave_world.ply', cfg=cfg)

    ds = FilteredDataset(ds, cfg)

    # add disturbances to dataset
    cfg.model_class = 'ScaledPolynomial'
    cfg.model_kwargs['exponent'] = [4.0]
    cfg.model_kwargs['learnable_exponents'] = False
    model_disturb = model_by_name(cfg.model_class)(w=[-0.01], **cfg.model_kwargs)

    ds = DepthBiasDataset(ds, model_disturb, cfg=cfg)
    # ds = NoisyDepthDataset(ds, noise=0.03)
    # ds = NoisyPoseDataset(ds, noise=0.05, mode='common')

    clouds = []
    poses = []
    for cloud, pose in ds:
        cloud = structured_to_unstructured(cloud[['x', 'y', 'z']])
        cloud = ds.transform(cloud, pose)
        clouds.append(cloud)
        poses.append(pose)

    cloud = np.concatenate(clouds)
    poses = np.asarray(poses)

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X, [m]')
    ax.set_ylabel('Y, [m]')
    ax.set_zlabel('Z, [m]')
    ax.set_xlim([cfg.dataset_kwargs['size'][0][0], cfg.dataset_kwargs['size'][0][1]])
    ax.set_ylim([cfg.dataset_kwargs['size'][1][0], cfg.dataset_kwargs['size'][1][1]])
    ax.set_zlim([cfg.dataset_kwargs['size'][2][0], cfg.dataset_kwargs['size'][2][1]])
    ax.grid()
    # point cloud
    ax.scatter(ds.pts[::100, 0], ds.pts[::100, 1], ds.pts[::100, 2])
    # view points trajectory
    for i in range(len(poses)-1):
        ax.plot([poses[i, 0, 3], poses[i + 1, 0, 3]],
                [poses[i, 1, 3], poses[i + 1, 1, 3]],
                [poses[i, 2, 3], poses[i + 1, 2, 3]], color='g', linewidth=5)
    plt.show()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    o3d.visualization.draw_geometries([pcd.voxel_down_sample(voxel_size=0.2)])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ds.pts)
    pcd.normals = o3d.utility.Vector3dVector(ds.normals)
    o3d.visualization.draw_geometries([pcd.voxel_down_sample(voxel_size=0.2)], point_show_normal=True)


def fragments_to_depth(fragments):
    """Convert rasterizer output fragments to depth."""
    # depth = fragments.zbuf.cpu().squeeze().numpy()
    depth = fragments.zbuf.detach().cpu().numpy()
    assert depth.shape[0] == 1  # batch size
    # assert depth.shape[3] == 1  # number of faces
    depth = depth[0, :, :, 0]
    return depth


def fragments_to_cloud(cameras, mesh, fragments):
    """Convert rasterizer output fragments and mesh to cloud."""
    assert len(cameras) == 1  # number of meshes
    assert len(mesh) == 1  # number of meshes
    # assert len(fragments) == 1  # number of fragments; there can be multiple fragments for some reason.
    assert fragments.pix_to_face.shape[0] == 1  # batch size
    assert fragments.pix_to_face.shape[3] == 1  # number of faces

    with torch.no_grad():
        face_attrs = mesh.verts_list()[0].detach()[mesh.faces_list()[0]]
        pts = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords.detach(), face_attrs)
        pts = pts.detach().cpu().reshape((-1, 3)).numpy()
        pts_struct = unstructured_to_structured(pts, names=['x', 'y', 'z'])

        vps = cameras.get_camera_center().detach().cpu()
        vps = vps.repeat((pts.shape[0], 1)).numpy()
        vps_struct = unstructured_to_structured(vps, names=['vp_x', 'vp_y', 'vp_z'])

        normal = mesh.faces_normals_list()[0].detach()[fragments.pix_to_face[0, ..., 0]]
        normal = normal.reshape((-1, 3))
        normal = normal / normal.norm(dim=-1, keepdim=True)
        normal = normal.cpu().numpy()
        normal_struct = unstructured_to_structured(normal, names=['normal_x', 'normal_y', 'normal_z'])

        cloud = merge_arrays([pts_struct, vps_struct, normal_struct], flatten=True)

    return cloud


def render_lidar_cloud(mesh, pose, fov=(90., 360.), size=(64, 512)):
    """Render lidar cloud from mesh."""
    assert len(fov) == 2
    assert 0.0 < fov[0] < 180.0
    assert 0.0 < fov[1] <= 360.0
    assert len(size) == 2
    device = mesh.device
    pose = pose.to(device=device, dtype=torch.float32)
    cam = pose[:3, 3][None]
    x = pose[:3, 0][None]
    y = pose[:3, 1][None]
    z = pose[:3, 2][None]
    fov = torch.deg2rad(torch.as_tensor(fov))
    size = torch.as_tensor(size)

    # Compose lidar scan of several perspective renders.
    n = 16
    image_fov = torch.tensor([fov[0], fov[1] / n])
    image_size = torch.tensor([size[0], int(size[1] / n)])
    f = 1.0 / torch.tan(image_fov / 2.0)
    f = f * image_size / 2.0
    p = image_size / 2.0
    f = f.flip(0)  # to xy
    p = p.flip(0)  # to xy
    clouds = []
    for i in range(n):
        print('%i / %i' % (i, n))
        a = torch.as_tensor(-fov[1] / 2 + i * image_fov[1])
        # Avoid erroneous rotation which cannot be rendered.
        a = a + 1e-3
        at = cam + torch.cos(a) * x + torch.sin(a) * y
        R, T = look_at_view_transform(eye=cam, at=at, up=z, device=device)
        # print(i, a.item())
        cameras = PerspectiveCameras(focal_length=f[None],
                                     principal_point=p[None],
                                     R=R, T=T,
                                     in_ndc=False,
                                     image_size=image_size[None],
                                     device=device)
        raster_settings = RasterizationSettings(
            image_size=image_size.tolist(),
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
            cull_backfaces=True,
            z_clip_value=1e-3,
        )
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        )
        fragments = rasterizer(mesh)
        cloud = fragments_to_cloud(cameras, mesh, fragments)
        # DepthCloud.from_structured_array(cloud).visualize()
        clouds.append(cloud)

    cloud = np.concatenate(clouds)
    # DepthCloud.from_structured_array(cloud).visualize()
    return cloud


def render_mesh(mesh, cameras, rasterizer):
    fragments = rasterizer(mesh)
    depth = fragments_to_depth(fragments)
    cloud = fragments_to_cloud(cameras, mesh, fragments)
    return depth, cloud


def demo_rendered_mesh():
    # ds = RenderedMeshDataset('simple_cave_01.obj')
    ds = RenderedMeshDataset('simple_cave_03.obj', n=1802)
    ds = ds[::10]
    print(len(ds))
    for cloud, pose in ds:
        # cloud = DepthCloud.from_structured_array(cloud).visualize()
        print(*pose[:3, 3])


def demo_orthographic():

    o = torch.tensor([[0.0, 300.0, 50.0]])
    at = o - torch.tensor([[0.0, 0.0, 1.0]])
    up = torch.tensor([[0.0, 1.0, 0.0]])  # x right, y up
    R, T = look_at_view_transform(eye=o, at=at, up=up)
    cameras = FoVOrthographicCameras(znear=1e-3, zfar=100.0,
                                     min_y=-300.0, max_y=300.0,
                                     min_x=-300.0, max_x=300.0,
                                     R=R, T=T)
    image_size = torch.tensor([512, 512])
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=image_size.tolist(),
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=True,
            z_clip_value=1e-3,
        ),
    )
    path = os.path.join(Config().pkg_dir, 'data', 'meshes', 'simple_cave_01.obj')
    mesh = load_objs_as_meshes([path])
    depth, cloud = render_mesh(mesh, cameras, rasterizer)
    plt.figure(figsize=(10, 10))
    plt.imshow(depth)
    plt.axis('off')
    plt.show()

    cloud = DepthCloud.from_structured_array(cloud)
    cloud.visualize()


def test():
    import doctest
    doctest.testmod()


def main():
    demo()


if __name__ == '__main__':
    main()

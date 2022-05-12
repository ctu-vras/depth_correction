from __future__ import absolute_import, division, print_function
from .config import Config
from .depth_cloud import DepthCloud
from .model import BaseModel
from .utils import cached, hashable, timing
import numpy as np
from numpy.lib.recfunctions import merge_arrays, structured_to_unstructured, unstructured_to_structured
from tf.transformations import euler_matrix
import torch
from copy import deepcopy
import os

default_rng = np.random.default_rng(135)


def box_point_cloud(size=(1.0, 1.0, 0.0), density=100.0, rng=default_rng):
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

    @cached
    def local_cloud(self, id):
        rng = np.random.default_rng(id)
        pts = box_point_cloud(size=self.size, density=self.density, rng=rng)
        pts -= self.height
        normals = np.zeros_like(pts)
        normals[:, 2] = 1.0
        pts = unstructured_to_structured(pts, names=['x', 'y', 'z'])
        normals = unstructured_to_structured(normals, names=['normal_x', 'normal_y', 'normal_z'])
        cloud = merge_arrays([pts, normals], flatten=True)
        return cloud

    def cloud_pose(self, id):
        pose = np.eye(4)
        pose[0, 3] = id * self.step
        pose[2, 3] = 1.0
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


class BaseDataset:
    def __init__(self,
                 name: str = None,
                 n_pts: int = 10_000,
                 n_poses: int = 5,
                 size: float = 20.0):
        self.name = name
        self.global_cloud = np.zeros([n_pts, 3])
        self.n_pts = n_pts
        self.n_poses = n_poses
        self.size = size
        self.poses = [np.eye(4) for _ in range(self.n_poses)]
        self.ids = range(self.n_poses)

    def __len__(self):
        """
        Denotes the total number of samples
        """
        return len(self.ids)

    def load_poses(self):
        poses = []
        for i in range(self.n_poses):
            cloud = self.local_cloud(i)
            vp = deepcopy(cloud[np.random.choice(range(len(cloud)))])
            vp[2] += np.random.uniform(0.1, 2.)
            pose = np.eye(4)
            pose[:3, 3] = vp
            poses.append(pose)
        assert len(poses) == self.n_poses
        return poses

    def local_cloud(self, i):
        assert self.poses is not None
        assert len(self.poses) > 0
        cloud = self.global_cloud[np.random.choice(range(self.n_pts), self.n_pts // self.n_poses)]
        # transform the point cloud to view point frame as it was sampled from global map
        R, t = self.poses[i][:3, :3], self.poses[i][:3, 3]
        cloud = cloud @ R - R.T @ t
        assert cloud.shape == (self.n_pts // self.n_poses, 3)
        return cloud

    def cloud_pose(self, i):
        pose = self.poses[i]
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
        ds.poses = self.poses
        ds.global_cloud = self.global_cloud
        if isinstance(i, (list, tuple)):
            ds.ids = [self.ids[j] for j in i]
        else:
            assert isinstance(i, slice)
            ds.ids = self.ids[i]
        return ds

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class Plane(BaseDataset):
    def __init__(self, n_pts: int = 10_000, n_poses: int = 2, size: float = 20.0):
        super(Plane, self).__init__(n_pts=n_pts, n_poses=n_poses, size=size)
        self.global_cloud = self.construct_global_cloud()
        self.poses = self.load_poses()

    def construct_global_cloud(self, seed=135):
        # create flat point cloud (a wall)
        np.random.seed(seed)
        pts = np.zeros((self.n_pts, 3), dtype=np.float64)
        pts[:, :2] = np.concatenate([np.random.uniform(0, self.size / 2, size=(self.n_pts // 2, 2)),
                                     np.random.uniform(0, self.size / 2, size=(self.n_pts // 2, 2)) + np.array(
                                            [-self.size / 2, 0])])
        return pts


class Angle(Plane):
    def __init__(self, n_pts: int = 10_000, n_poses: int = 5, size: float = 20.0, degrees: float = 0.0):
        super(Angle, self).__init__(n_pts=n_pts, n_poses=n_poses, size=size)
        if degrees != 0.0:
            self.global_cloud[self.n_pts // 2:] = self.rotate_pts(self.global_cloud[self.n_pts // 2:], origin=(0, 0, 0),
                                                                  degrees=degrees, axis='Y')
        self.poses = self.load_poses()

    @staticmethod
    def rotate_pts(p, origin=(0, 0, 0), degrees=0.0, axis='X'):
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


class Mesh(BaseDataset):
    def __init__(self, mesh_name, n_pts: int = 10_000, n_poses: int = 5, size: float = 20.0, pts_to_sample: int = 10_000_000):
        super(Mesh, self).__init__(name=mesh_name, n_pts=n_pts, n_poses=n_poses, size=size)

        self.mesh_path = os.path.join(os.path.dirname(__file__), '../../data/meshes/%s' % self.name)
        if not os.path.exists(self.mesh_path):
            raise FileExistsError('Mesh file %s does not exist, download meshes from'
                                  'https://drive.google.com/drive/folders/1S3UlJ4MgNsU72PTwJku-gyHZbv3aw26Z?usp=sharing'
                                  'and place them to ./data/ folder' % self.mesh_path)
        else:
            print('Loading mesh: %s' % mesh_name)
        self.pts_to_sample = pts_to_sample
        self.global_cloud = self.construct_global_cloud()
        self.poses = self.load_poses()

    def construct_global_cloud(self, seed=135):
        from pytorch3d.ops import sample_points_from_meshes
        from pytorch3d.io import load_obj, load_ply
        from pytorch3d.structures import Meshes

        if self.mesh_path[-3:] == 'obj':
            pts, faces, _ = load_obj(self.mesh_path)
            mesh = Meshes(verts=[pts], faces=[faces.verts_idx])
        elif self.mesh_path[-3:] == 'ply':
            pts, faces = load_ply(self.mesh_path)
            mesh = Meshes(verts=[pts], faces=[faces])
        else:
            raise ValueError('Supported mesh formats are *.obj or *.ply')

        torch.random.manual_seed(seed)
        pts = sample_points_from_meshes(mesh, self.pts_to_sample).squeeze().numpy()
        X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
        pts = pts[np.logical_and(np.logical_and(np.logical_and(X >= -self.size / 2, X <= self.size / 2),
                                                np.logical_and(Y >= -self.size / 2, Y <= self.size / 2)),
                                 np.logical_and(Z >= -self.size / 2, Z <= self.size / 2))]
        self.n_pts = len(pts)
        return pts


class Forwarding(object):
    def __init__(self, target):
        self.target = target

    def __getattr__(self, item):
        return getattr(self.target, item)

    def __iter__(self):
        return self.target.__iter__()


class ForwardingDataset(Forwarding):
    def __init__(self, target):
        super().__init__(target)

    def modify_cloud(self, cloud):
        return cloud

    def modify_pose(self, pose):
        return pose

    def __iter__(self):
        for cloud, pose in self.target:
            yield self.modify_cloud(cloud), self.modify_pose(pose)

    def local_cloud(self, id):
        return self.modify_cloud(self.target.local_cloud(id))

    def cloud_pose(self, id):
        return self.modify_pose(self.target.cloud_pose(id))


class NoisyPoseDataset(ForwardingDataset):

    def __init__(self, dataset, noise=None):
        super().__init__(dataset)
        self.cov = np.diag(noise) if noise and any(noise) else None

    def random_transform(self, id):
        rng = np.random.default_rng(id)
        noise_vec = rng.multivariate_normal(np.zeros((6,)), self.cov)
        noise = euler_matrix(*noise_vec[:3])
        noise[:3, 3] = noise_vec[3:]
        return noise

    # TODO: Cache
    def modify_pose(self, pose):
        print('NoisyPoseDataset.modify_pose')
        h = hash(hashable(pose))
        if self.cov is not None:
            noise = self.random_transform(h)
            pose = np.matmul(pose, noise)
        return pose


class NoisyDepthDataset(ForwardingDataset):

    def __init__(self, dataset, noise=None):
        super().__init__(dataset)
        self.noise = noise

    def modify_cloud(self, cloud):
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
            pts[valid] += dirs * self.noise * np.random.randn(*depth.shape)[:, None]
            cloud[['x', 'y', 'z']] = unstructured_to_structured(pts, names=['x', 'y', 'z'])
        return cloud


class DepthBiasDataset(ForwardingDataset):

    def __init__(self, dataset, model=None):
        super().__init__(dataset)
        self.model = model

    def modify_cloud(self, cloud):
        if self.model is not None:
            assert isinstance(self.model, BaseModel)
            dc = DepthCloud.from_structured_array(cloud)
            assert isinstance(dc, DepthCloud)
            assert dc.normals is not None
            dc.update_incidence_angles()
            dc = self.model.inverse(dc)
            pts = dc.to_points().detach().numpy()
            cloud[['x', 'y', 'z']] = unstructured_to_structured(pts, names=['x', 'y', 'z'])
        return cloud


def dataset_by_name(name):
    parts = name.split('/')
    if len(parts) == 2:
        name = parts[0]

    if name == 'ground_plane':
        return GroundPlaneDataset
    elif name == 'angle':
        return Angle
    elif '.obj' in name or '.ply' in name:
        return Mesh
    elif name == 'asl_laser':
        import data.asl_laser
        return getattr(data.asl_laser, 'Dataset')
    elif name == 'semantic_kitti':
        import data.semantic_kitti
        return getattr(data.semantic_kitti, 'Dataset')
    raise ValueError('Unknown dataset: %s.' % name)


def create_dataset(name, cfg: Config):
    Dataset = dataset_by_name(name)
    d = Dataset(name, *cfg.dataset_args, **cfg.dataset_kwargs)
    d = d[::cfg.data_step]
    return d


def demo():
    import open3d as o3d
    import matplotlib.pyplot as plt

    # ds = Plane()
    # ds = Angle(degrees=60.0)
    # ds = Mesh(mesh_name='simple_cave_01.obj', size=20)
    # ds = Mesh(mesh_name='burning_building_rubble.ply', size=20)
    ds = Mesh(mesh_name='cave_world.ply', size=50)

    clouds = []
    poses = []
    for cloud, pose in ds:
        cloud = ds.transform(cloud, pose)
        clouds.append(cloud)
        poses.append(pose)

    cloud = np.concatenate(clouds)
    poses = np.asarray(poses)

    plt.figure()
    plt.axis('equal')
    plt.title('Trajectory')
    plt.plot(poses[:, 0, 3], poses[:, 1, 3], '.')
    plt.grid()
    plt.show()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    o3d.visualization.draw_geometries([pcd.voxel_down_sample(voxel_size=0.2)])


def main():
    demo()


if __name__ == '__main__':
    main()

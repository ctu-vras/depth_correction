import numpy as np
from copy import deepcopy
import torch
import os

__all__ = [
    'Plane',
    'Angle',
    'Mesh'
]


class BaseDataset:
    def __init__(self,
                 name: str = 'data',
                 n_pts: int = 10_000,
                 n_poses: int = 2,
                 size: float = 20.0):
        self.name = name
        self.global_cloud = np.zeros([n_pts, 3])
        self.n_pts = n_pts
        self.n_poses = n_poses
        self.size = size
        self.poses = [np.eye(4) for _ in range(self.n_poses)]
        self.ids = range(len(self.poses))

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
        cloud = self.global_cloud[np.random.choice(range(self.n_pts), self.n_pts // self.n_poses)]
        return cloud

    def cloud_pose(self, i):
        pose = self.poses[i]
        return pose

    def __getitem__(self, item):
        if isinstance(item, int):
            id = self.ids[item]
            return self.local_cloud(id), self.cloud_pose(id)

        ds = BaseDataset()
        ds.poses = self.poses
        if isinstance(item, (list, tuple)):
            ds.ids = [self.ids[i] for i in item]
        else:
            assert isinstance(item, slice)
            ds.ids = self.ids[item]
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
    def __init__(self, n_pts: int = 10_000, n_poses: int = 2, size: float = 20.0, degrees: float = 0.0):
        super(Angle, self).__init__(n_pts=n_pts, n_poses=n_poses, size=size)
        if degrees != 0.0:
            self.global_cloud[self.n_pts // 2:] = self.rotate_pts(self.global_cloud[self.n_pts // 2:], origin=(0, 0, 0),
                                                                  degrees=degrees, axis='Y')

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
    def __init__(self, mesh_name, n_pts: int = 10_000, n_poses: int = 2, size: float = 20.0, pts_to_sample: int = 10_000_000):
        super(Mesh, self).__init__(name=mesh_name, n_pts=n_pts, n_poses=n_poses, size=size)

        self.mesh_path = os.path.join(os.path.dirname(__file__), '../../data/meshes/%s' % self.name)
        if not os.path.exists(self.mesh_path):
            raise FileExistsError('Mesh file %s does not exist' % self.mesh_path)
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


def demo():
    import open3d as o3d

    # ds = Plane()
    # ds = Angle(degrees=60.0)
    ds = Mesh(mesh_name='simple_cave_01.obj', size=20)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ds.global_cloud)
    o3d.visualization.draw_geometries([pcd.voxel_down_sample(voxel_size=0.5)])

    for i in range(len(ds)):
        cloud, pose = ds[i]
        print(f'Visualizing cloud of size: {cloud.shape} viewed from location: {pose[:3, 3]}')

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        o3d.visualization.draw_geometries([pcd.voxel_down_sample(voxel_size=0.5)])


def main():
    demo()


if __name__ == '__main__':
    main()

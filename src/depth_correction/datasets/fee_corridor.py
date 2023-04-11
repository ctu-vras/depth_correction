from __future__ import absolute_import, division, print_function
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
import os
import re

__all__ = [
    'prefix',
    'data_dir',
    'dataset_names',
    'Dataset',
]

prefix = 'fee_corridor'
data_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', prefix)
data_dir = os.path.normpath(data_dir)

seq_names = [
    'seq1',
    'seq2',
]

dataset_names = [
    'seq1_start_0_end_7_step_1',
    'seq1_start_8_end_15_step_1',
    'seq1_start_16_end_23_step_1',
    'seq1_start_24_end_33_step_1',
    'seq2_start_0_end_10_step_1',
    'seq2_start_11_end_20_step_1',
    'seq2_start_21_end_30_step_1',
    'seq2_start_31_end_42_step_1',
]


def read_points_npz(path):
    points = np.load(path)
    points = points['cloud']
    return points


def read_poses(path):
    poses = np.genfromtxt(path, delimiter=', ', skip_header=True)
    ids = np.genfromtxt(path, delimiter=', ', dtype=str, skip_header=True)[:, 0].tolist()
    # assert ids == list(range(len(ids)))
    poses = poses[:, 2:]
    poses = poses.reshape((-1, 4, 4))
    # poses = dict(zip(ids, poses))
    return ids, poses


class Dataset(object):

    def __init__(self, name=None, path=None, poses_path=None, zero_origin=False, static_poses=True,
                 xyz_from_leica_tracker=False):
        """Depth Correction dataset or a dataset in that format.

        :param name: Dataset name.
        :param path: Dataset path, takes precedence over name.
        :param poses_path: Override for poses file path.
        """
        step = 1
        sub_seq_ids = slice(None, None, 1)

        if name is None:
            name = 'seq2'

        if path:  # Use provided path.
            name = os.path.split(path)[1]
            assert name
        elif name:  # Construct path from name.
            step = re.search('_step_(\d+)', name)
            end = re.search('end_(\d+)', name)
            start = re.search('start_(\d+)', name)
            start = int(start.group(1)) if start else None
            end = int(end.group(1)) if end else None
            step = int(step.group(1)) if step else 1
            sub_seq_ids = slice(start, end, step)
            parts = name.split('/')
            assert 1 <= len(parts) <= 2
            if len(parts) == 2:
                assert parts[0] == prefix
                name = parts[1]
            name = name[:4]  # seq1
            path = os.path.join(data_dir, 'sequences/', name)

        self.name = name
        self.data_step = int(step)
        self.path = path
        self.poses_path = poses_path
        self.prefix = 'static_' if static_poses else ''
        self.zero_origin = zero_origin
        self.xyz_from_leica_tracker = xyz_from_leica_tracker
        self.static_poses = static_poses
        # read poses from Leica tracker
        self.leica_xyz = self.read_leica_xyz()

        if self.path:
            self.ids, self.poses = read_poses(self.cloud_poses_path())
            if self.xyz_from_leica_tracker:
                self.poses[:, :3, 3] = self.leica_xyz
            # move poses to origin to 0:
            if self.zero_origin:
                Tr_inv = np.linalg.inv(self.poses[0])
                self.poses = np.asarray([np.matmul(Tr_inv, pose) for pose in self.poses])
            self.poses = dict(zip(self.ids, self.poses))
            self.leica_xyz = dict(zip(self.ids, self.leica_xyz))
            if not self.poses_path:
                self.ids = self.ids[sub_seq_ids]
        else:
            self.ids = None
            self.poses = None

    def local_cloud_path(self, id):
        return os.path.join(self.path, self.prefix + 'ouster_points', '%s.npz' % id)

    def cloud_poses_path(self):
        if self.poses_path:
            return self.poses_path
        return os.path.join(self.path, 'poses', self.prefix + 'poses.csv')

    def global_cloud_path(self, resolution_cm=5):
        assert resolution_cm in (2, 5)
        return os.path.join(self.path, '../../maps/', 'npz', 'map_%icm_alligned.npz' % resolution_cm)

    def read_leica_xyz(self):
        path = os.path.join(self.path, 'poses', self.prefix + 'leica_poses_raw.txt')
        xyz_raw = np.genfromtxt(path)
        T_map2subt = np.genfromtxt(os.path.join(self.path, 'calibration', 'map2subt.txt'))
        xyz = xyz_raw @ T_map2subt[:3, :3].T + T_map2subt[:3, 3:4].T
        return xyz

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        if isinstance(item, int):
            id = self.ids[item]
            cloud = self.local_cloud(id)
            pose = self.cloud_pose(id)
            return cloud, pose

        ds = Dataset(name=self.name,
                     path=self.path,
                     zero_origin=self.zero_origin,
                     static_poses=self.static_poses,
                     xyz_from_leica_tracker=self.xyz_from_leica_tracker)
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

    def __str__(self):
        return prefix + '/' + self.name

    def local_cloud(self, id):
        cloud = read_points_npz(self.local_cloud_path(id))
        return cloud.reshape((-1,))

    def cloud_pose(self, id):
        return self.poses[id]

    def global_cloud(self, resolution_cm=5):
        assert resolution_cm in (2, 5)
        cloud = read_points_npz(self.global_cloud_path(resolution_cm))
        if self.zero_origin:
            cloud_pose = self.global_cloud_pose()
            Tr = np.linalg.inv(cloud_pose)
            pts = structured_to_unstructured(cloud[['x', 'y', 'z']])
            pts = pts @ Tr[:-1, :-1].T + Tr[:-1, -1:].T
            cloud[['x', 'y', 'z']] = unstructured_to_structured(pts, names=['x', 'y', 'z'])
        return cloud.reshape((-1,))

    def global_cloud_pose(self):
        pose = read_poses(self.cloud_poses_path())[1][0]
        return pose


def demo():
    import open3d as o3d
    import matplotlib.pyplot as plt
    from depth_correction.loss import icp_loss
    from depth_correction.depth_cloud import DepthCloud
    from depth_correction.preproc import filter_grid
    from depth_correction.config import Config

    cfg = Config()
    cfg.device = 'cuda'
    cfg.loss_kwargs['icp_point_to_plane'] = False
    cfg.dataset_kwargs['static_poses'] = False
    cfg.dataset_kwargs['zero_origin'] = False

    for name in dataset_names:
        ds = Dataset(name, **cfg.dataset_kwargs)
        print('Dataset %s contains %i clouds.' % (name, len(ds)))

        global_points_struct = ds.global_cloud(resolution_cm=5)
        global_points = structured_to_unstructured(global_points_struct[['x', 'y', 'z']])
        global_color = structured_to_unstructured(global_points_struct[['r', 'g', 'b']]) / 255.

        pcd_global = o3d.geometry.PointCloud()
        pcd_global.points = o3d.utility.Vector3dVector(global_points)
        pcd_global.colors = o3d.utility.Vector3dVector(global_color)

        points_list = []
        poses = []
        xyz_leica = []
        dists = []
        pose_prev = np.eye(4)
        for id in ds.ids:
            points = ds.local_cloud(id)
            pose = ds.poses[id]
            points = structured_to_unstructured(points[['x', 'y', 'z']])
            points = np.matmul(points, pose[:3, :3].T) + pose[:3, 3][None]

            points_list.append(points)
            poses.append(pose)
            xyz_leica.append(ds.leica_xyz[id])
            dists.append(np.linalg.norm(pose[:3, 3] - pose_prev[:3, 3]))
            pose_prev = pose
            print('%i points read from dataset %s, points.' % (points.shape[0], ds.name))

        points = np.concatenate(points_list)
        poses = np.asarray(poses)
        xyz_leica = np.asarray(xyz_leica)
        dists.pop(0)

        print('Dataset %s, pts: %d, distances between poses:  mean=%.2f, max=%.2f'
              % (name, points.shape[0], np.mean(dists).item(), np.max(dists)))

        cloud = DepthCloud.from_points(points, device=cfg.device)
        # cloud = DepthCloud.from_points(points_list[np.random.choice(range(len(points_list)))], device=cfg.device)
        cloud.update_points()
        global_cloud = DepthCloud.from_points(global_points, device=cfg.device)
        global_cloud.update_points()

        cloud = filter_grid(cloud, grid_res=cfg.grid_res)
        global_cloud = filter_grid(global_cloud, grid_res=cfg.grid_res)

        # we don't compute point to plane distance (point to point instead)
        # as we don't estimate normals for the depth clouds
        map_loss, _ = icp_loss([[cloud, global_cloud]], **cfg.loss_kwargs, verbose=True)
        print('Point to point loss: %f' % map_loss.item())
        pose_loss = np.linalg.norm(xyz_leica - poses[:, :3, 3], axis=1).mean()
        print('Pose loss: %f' % pose_loss)

        plt.figure()
        plt.axis('equal')
        plt.title('Trajectory for sequence: %s' % name)
        plt.plot(poses[:, 0, 3], poses[:, 1, 3], '*', label='SLAM poses')
        # plt.plot(xyz_leica[:, 0], xyz_leica[:, 1], '+', label='Leica poses')
        plt.legend()
        plt.grid()
        plt.show()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud.points.detach().cpu())
        o3d.visualization.draw_geometries([pcd, pcd_global])


def main():
    demo()


if __name__ == '__main__':
    main()

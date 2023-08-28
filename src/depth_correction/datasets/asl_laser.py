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
    'read_points',
    'read_points_npz',
    'read_poses',
]

prefix = 'asl_laser'
data_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', prefix)
data_dir = os.path.realpath(data_dir)

dataset_names = [
    'apartment',
    'eth',
    'gazebo_summer',
    'gazebo_winter',
    'plain',
    'stairs',
    'wood_autumn',
    'wood_summer',
]


def read_points(path):
    points = np.genfromtxt(path, delimiter=',', skip_header=1)
    points = points[:, 1:4]
    # points = points.T
    return points


def read_points_npz(path):
    points = np.load(path)
    points = points['arr_0']
    # points = points[:, 1:4]
    # points = points.T
    return points


def read_poses(path):
    poses = np.genfromtxt(path, delimiter=', ', skip_header=1)
    ids = poses[:, 0].astype(int).tolist()
    # assert ids == list(range(len(ids)))
    poses = poses[:, 2:]
    poses = poses.reshape((-1, 4, 4))
    # poses = dict(zip(ids, poses))
    poses = list(poses)
    return ids, poses
    # return dict(zip(ids, poses))


def write_poses(ids, poses, path, ts=None):
    if ts is None:
        ts = ids
    with open(path, 'w') as f:
        f.write('poseId, timestamp, T00, T01, T02, T03, T10, T11, T12, T13, T20, T21, T22, T23, T30, T31, T32, T33\n')
        for id, t, pose in zip(ids, ts, poses):
            f.write('%s, %.9f, %s\n' % (id, t, ', '.join(['%.9f' % x for x in pose.flatten()])))


class Dataset(object):
    default_poses_csv = 'pose_scanner_leica.csv'

    def __init__(self, name=None, path=None, poses_csv=default_poses_csv, poses_path=None):
        """ASL laser dataset or a dataset in that format.

        :param name: Dataset name.
        :param path: Dataset path, takes precedence over name.
        :param poses_csv: Poses CSV file name.
        :param poses_path: Override for poses CSV path.
        """
        data_step = 1
        if path:  # Use provided path.
            name = os.path.split(path)[1]
            assert name
        elif name:  # Construct path from name.
            s = re.search('_step_(\d+)', name)
            if s:
                name = name.replace(s.group(0), '')
                data_step = s.group(1)
            parts = name.split('/')
            assert 1 <= len(parts) <= 2
            if len(parts) == 2:
                assert parts[0] == prefix
                name = parts[1]
            path = os.path.join(data_dir, name)

        if not poses_csv:
            poses_csv = Dataset.default_poses_csv

        self.name = name
        self.data_step = int(data_step)
        self.path = path
        self.poses_path = poses_path
        self.poses_csv = poses_csv

        if self.poses_path or self.path:
            self.ids, self.poses = read_poses(self.cloud_poses_path())
            self.ids = self.ids[::self.data_step]
        else:
            self.ids = None
            self.poses = None

    def local_cloud_path(self, id):
        return os.path.join(self.path, 'csv_local', 'Hokuyo_%s.csv' % id)

    def local_cloud_fixed_csv_path(self, id):
        return os.path.join(self.path, 'local_fixed', '%s-Tiltlaser.csv' % id)

    def local_cloud_fixed_npz_path(self, id):
        return os.path.join(self.path, 'local_fixed', '%s-Tiltlaser.npz' % id)

    def global_cloud_path(self, id):
        return os.path.join(self.path, 'csv_global', 'PointCloud%s.csv' % id)

    def cloud_poses_path(self):
        if self.poses_path:
            return self.poses_path
        return os.path.join(self.path, 'csv_global', self.poses_csv)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        if isinstance(item, int):
            id = self.ids[item]
            cloud = self.local_cloud(id)
            pose = self.cloud_pose(id)
            return cloud, pose

        ds = Dataset()
        ds.name = self.name
        ds.path = self.path
        ds.poses_csv = self.poses_csv
        ds.poses_path = self.poses_path
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
        return read_points(self.local_cloud_path(id))
        # return read_points_npz(self.local_cloud_fixed_npz_path(id))

    def global_cloud(self, id):
        return read_points(self.global_cloud_path(id))

    def cloud_pose(self, id):
        return self.poses[id]


def demo():
    import open3d as o3d
    import matplotlib.pyplot as plt

    for name in dataset_names:
        name += '_step_5'
        ds = Dataset(name)
        print('Dataset %s contains %i clouds.' % (name, len(ds)))

        clouds = []
        poses = []
        dists = []
        pose_prev = np.eye(4)
        for id in ds.ids:
            cloud = ds.local_cloud(id)
            if cloud.dtype.names is not None:
                cloud = structured_to_unstructured(cloud[['x', 'y', 'z']])
            pose = ds.cloud_pose(id)
            cloud = np.matmul(cloud, pose[:3, :3].T) + pose[:3, 3:].T

            clouds.append(cloud)
            poses.append(pose)
            dists.append(np.linalg.norm(pose[:3, 3] - pose_prev[:3, 3]))
            pose_prev = pose
            # print('%i points read from dataset %s, cloud %i.' % (cloud.shape[0], ds.name, id))

        cloud = np.concatenate(clouds)
        poses = np.asarray(poses)
        dists.pop(0)

        print('Dataset %s, pts: %d, distances between poses:  mean=%.2f, max=%.2f'
              % (name, cloud.shape[0], np.mean(dists), np.max(dists)))

        plt.figure()
        plt.title('Trajectory for sequence: %s' % name)
        plt.plot(poses[:, 0, 3], poses[:, 1, 3], '.')
        plt.grid()
        plt.show()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        o3d.visualization.draw_geometries([pcd])


def main():
    demo()


if __name__ == '__main__':
    main()

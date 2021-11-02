from __future__ import absolute_import, division, print_function
import numpy as np
import os

data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'asl_laser')

dataset_names = [
    'apartment',
    'eth',
    'gazebo_winter',
    'gazebo_summer',
    'plain',
    'stairs',
    'wood_summer',
    'wood_autumn'
]


def read_points(path):
    points = np.genfromtxt(path, delimiter=',', skip_header=1)
    points = points[:, 1:4]
    # points = points.T
    return points


def read_poses(path):
    poses = np.genfromtxt(path, delimiter=', ', skip_header=1)
    ids = poses[:, 0].astype(int).tolist()
    poses = poses[:, 2:]
    poses = poses.reshape((-1, 4, 4))
    poses = dict(zip(ids, poses))
    return ids, poses
    # return dict(zip(ids, poses))


class Dataset(object):

    def __init__(self, name=dataset_names[0], path=None):
        self.name = name
        if path is None:
            path = os.path.join(data_dir, name)
        self.path = path
        self.ids, self.poses = read_poses(self.cloud_poses_path())
        # self.poses = read_poses(self.cloud_poses_path())

    def local_cloud_path(self, id):
        return os.path.join(self.path, 'csv_local', 'Hokuyo_%s.csv' % id)

    def global_cloud_path(self, id):
        return os.path.join(self.path, 'csv_global', 'PointCloud%s.csv' % id)

    def cloud_poses_path(self):
        return os.path.join(self.path, 'csv_global', 'pose_scanner_leica.csv')

    def __len__(self):
        return len(self.ids)

    def local_cloud(self, id):
        return read_points(self.local_cloud_path(id))

    def global_cloud(self, id):
        return read_points(self.global_cloud_path(id))

    def cloud_pose(self, id):
        return self.poses[id]


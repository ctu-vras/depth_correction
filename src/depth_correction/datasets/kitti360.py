#!/usr/bin/python

import os
import re
import numpy as np
import matplotlib.cm
try:
    import open3d
except:
    print('KITTI-360: Open3d is not installed')
from copy import copy
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
import scipy.spatial
# pip install git+https://github.com/autonomousvision/kitti360Scripts.git
from kitti360scripts.helpers.annotation  import Annotation3DPly, global2local
from kitti360scripts.helpers.labels      import id2label
from kitti360scripts.helpers.ply         import read_ply
from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationRigid


prefix = 'kitti360'
if 'KITTI360_DATASET' in os.environ:
    data_dir = os.environ['KITTI360_DATASET']
else:
    data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', prefix))

dataset_names = [
    '00_start_102_end_152_step_1',
    '03_start_102_end_152_step_1',
    '05_start_102_end_152_step_1',
    '06_start_102_end_152_step_1',
    '07_start_52_end_102_step_1',
    '09_start_102_end_152_step_1',
    '10_start_102_end_152_step_1',
    '04_start_2997_end_3047_step_1',
]


class Sequence(object):
    def __init__(self, path=None, seq=0, filtered_scans=False):
        if not path:
            path = data_dir
        self.path = path
        if isinstance(seq, str):
            parts = seq.split('/')
            assert 1 <= len(parts) <= 2
            if len(parts) == 2:
                assert parts[0] == prefix
                seq = parts[1].split('_')[0]
        self.seq = '2013_05_28_drive_%04d_sync' % int(seq)
        if filtered_scans:
            self.cloud_dir = os.path.join(self.path, 'data_3d_filtered', self.seq, 'velodyne_points', 'data')
        else:
            self.cloud_dir = os.path.join(self.path, 'data_3d_raw', self.seq, 'velodyne_points', 'data')
        self.T_cam2lidar = self.read_calibration()
        self.T_lidar2cam = np.linalg.inv(self.T_cam2lidar)
        self.poses, self.ids = self.read_poses()

    def get_poses_path(self):
        return os.path.join(self.path, 'data_poses', self.seq, 'cam0_to_world.txt')

    def read_poses(self):
        path = self.get_poses_path()
        data = np.loadtxt(path, dtype=np.float32)
        ids = np.asarray([int(i) for i in data[:, 0]])  # index of poses
        poses = data[:, 1:].reshape((-1, 4, 4)) @ self.T_lidar2cam
        # ensure that there are corresponding point clouds for ids
        clouds_ids = [int(i[:-4]) for i in os.listdir(self.cloud_dir)]
        mask = [True if i in clouds_ids else False for i in ids]
        ids = ids[mask]
        poses = poses[mask]
        return poses, ids

    def read_calibration(self):
        fileCameraToLidar = os.path.join(self.path, 'calibration', 'calib_cam_to_velo.txt')
        cam2lidar = loadCalibrationRigid(fileCameraToLidar)
        return cam2lidar

    def get_cloud_path(self, i):
        fpath = os.path.join(self.cloud_dir, '%010d.bin' % int(i))
        return fpath

    def local_cloud(self, i, filter_ego_pts=False):
        file = self.get_cloud_path(i)
        cloud = np.fromfile(file, dtype=np.float32)
        cloud = cloud.reshape((-1, 4))

        if filter_ego_pts:
            valid_indices = cloud[:, 0] < -3.
            valid_indices = valid_indices | (cloud[:, 0] > 3.)
            valid_indices = valid_indices | (cloud[:, 1] < -3.)
            valid_indices = valid_indices | (cloud[:, 1] > 3.)
            cloud = cloud[valid_indices]

        cloud = unstructured_to_structured(cloud, names=['x', 'y', 'z', 'i'])
        return cloud

    def cloud_pose(self, i, dtype=np.float32):
        pose = self.poses[self.ids.index(i)].astype(dtype=dtype)
        return pose

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item):
        if isinstance(item, int):
            id = self.ids[item]
            cloud = self.local_cloud(id)
            pose = self.poses[item]
            return cloud, pose

        ds = copy(self)
        if isinstance(item, (list, tuple)):
            ds.ids = [self.ids[i] for i in item]
            ds.poses = [self.poses[i] for i in item]
        else:
            assert isinstance(item, slice)
            ds.ids = self.ids[item]
            ds.poses = self.poses[item]
        return ds

    def __len__(self):
        return len(self.ids)


class ColoredCloud(object):
    # Constructor
    def __init__(self, path=None, seq=0):
        if not path:
            path = data_dir
        self.path = path

        self.downSampleEvery = -1
        # show visible point clouds only
        self.showVisibleOnly = False
        # colormap for instances
        self.cmap = matplotlib.cm.get_cmap('Set1')
        self.cmap_length = 9
        # colormap for confidence
        self.cmap_conf = matplotlib.cm.get_cmap('plasma')

        sequence = '2013_05_28_drive_%04d_sync' % seq
        self.label3DPcdPath = os.path.join(self.path, 'data_3d_semantics')
        # self.annotation3D = Annotation3D(self.label3DBboxPath, sequence)
        self.annotation3DPly = Annotation3DPly(self.label3DPcdPath, sequence)
        self.sequence = sequence

        self.pointClouds = {}

    def getColor(self, idx):
        if idx == 0:
            return np.array([0, 0, 0])
        return np.asarray(self.cmap(idx % self.cmap_length)[:3]) * 255.

    def assignColor(self, globalIds, gtType='semantic'):
        if not isinstance(globalIds, (np.ndarray, np.generic)):
            globalIds = np.array(globalIds)[None]
        color = np.zeros((globalIds.size, 3))
        for uid in np.unique(globalIds):
            semanticId, instanceId = global2local(uid)
            if gtType == 'semantic':
                color[globalIds == uid] = id2label[semanticId].color
            elif instanceId > 0:
                color[globalIds == uid] = self.getColor(instanceId)
            else:
                color[globalIds == uid] = (96, 96, 96)  # stuff objects in instance mode
        color = color.astype(float) / 255.0
        return color

    def assignColorConfidence(self, confidence):
        color = self.cmap_conf(confidence)[:, :3]
        return color

    def loadWindow(self, pcdFile, colorType='semantic'):
        window = pcdFile.split(os.sep)[-2]

        print('Loading %s ' % pcdFile)

        # load ply data using open3d for visualization
        if window in self.pointClouds.keys():
            pcd = self.pointClouds[window]
        else:
            # pcd = open3d.io.read_point_cloud(pcdFile)
            data = read_ply(pcdFile)
            points = np.vstack((data['x'], data['y'], data['z'])).T
            color = np.vstack((data['red'], data['green'], data['blue'])).T
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(points)
            pcd.colors = open3d.utility.Vector3dVector(color.astype(float) / 255.)

        # assign color
        if colorType == 'semantic' or colorType == 'instance':
            globalIds = data['instance']
            ptsColor = self.assignColor(globalIds, colorType)
            pcd.colors = open3d.utility.Vector3dVector(ptsColor)
        elif colorType == 'confidence':
            confidence = data[:, -1]
            ptsColor = self.assignColorConfidence(confidence)
            pcd.colors = open3d.utility.Vector3dVector(ptsColor)
        elif colorType != 'rgb':
            raise ValueError("Color type can only be 'rgb', 'semantic', 'instance'!")

        if self.showVisibleOnly:
            isVisible = data['visible']
            pcd = pcd.select_by_index(np.where(isVisible)[0])

        if self.downSampleEvery > 1:
            print(np.asarray(pcd.points).shape)
            pcd = pcd.uniform_down_sample(self.downSampleEvery)
            print(np.asarray(pcd.points).shape)
        return pcd


class Dataset(Sequence):
    def __init__(self, name, path=None, poses_path=None, zero_origin=True, filtered_scans=True):
        """ KITTI-360 dataset or a dataset in that format.

        :param name: Dataset name in format NN_start_SS_end_EE_step_ss
        :param path: Dataset path, takes precedence over name.
        :param zero_origin: Whether to move data set coordinate system origin to (0, 0, 0).
        :param filtered_scans: Whether to use point clouds with filtered dynamic objects
                               (if True 'data_3d_filtered' folder should be generated in advance).
        """
        super(Dataset, self).__init__(seq=name, path=path, filtered_scans=filtered_scans)
        assert isinstance(name, str)
        parts = name.split('/')
        assert 1 <= len(parts) <= 2
        if len(parts) == 2:
            assert parts[0] == prefix
            name = parts[1]
        self.name = name

        # Use slice specification from name.
        start = re.search('start_(\d+)', name)
        end = re.search('end_(\d+)', name)
        step = re.search('step_(\d+)', name)
        self.start = int(start.group(1)) if start else None
        self.end = int(end.group(1)) if end else None
        self.step = int(step.group(1)) if step else None
        sub_seq = slice(self.start, self.end, self.step)
        self.ids = self.ids[sub_seq]
        self.poses = self.poses[sub_seq]

        assert os.path.exists(self.cloud_dir), 'Path %s does not exist' % self.cloud_dir

        # move poses to zero-origin
        if zero_origin:
            Tr_inv = np.linalg.inv(self.poses[0])
            self.poses = np.asarray([np.matmul(Tr_inv, pose) for pose in self.poses])

    def __str__(self):
        return prefix + '/' + self.name

    def get_dynamic_points(self):
        pcd_path = os.path.join(self.path, 'data_3d_semantics', 'train', self.seq, 'dynamic')
        dynamic_points = []
        for p in os.listdir(pcd_path):
            pcd_file = os.path.join(pcd_path, p)
            data = read_ply(pcd_file)
            points = structured_to_unstructured(data[['x', 'y', 'z']])
            dynamic_points.append(points)
        if len(dynamic_points) > 0:
            dynamic_points = np.concatenate(dynamic_points)
        return dynamic_points

    def __str__(self):
        return f'kitti360/{self.name}'


def visualize_colored_submaps():

    mode = 'semantic'  # ['rgb', 'semantic', 'instance', 'confidence', 'bbox']

    ds = ColoredCloud(seq=0)

    pcdFileList = ds.annotation3DPly.pcdFileList
    for idx, pcdFile in enumerate(pcdFileList):
        pcd = ds.loadWindow(pcdFile, mode)
        if len(np.asarray(pcd.points)) == 0:
            print('Warning: skipping empty point cloud!')
            continue
        open3d.visualization.draw_geometries([pcd])

    exit()


def visualize_datasets():
    from depth_correction.config import Config
    from depth_correction.visualization import visualize_dataset

    cfg = Config(min_depth=1, max_depth=15, grid_res=0.4, nn_r=0.4)
    for name in dataset_names:
        ds = Dataset(name='%s/%s' % (prefix, name), zero_origin=True)

        visualize_dataset(ds, cfg)


def visualize_colored_datasets():
    import open3d as o3d
    from depth_correction.config import Config
    from depth_correction.preproc import filtered_cloud

    cfg = Config(min_depth=1, max_depth=15, grid_res=0.4, nn_r=0.4)
    for name in dataset_names:
        ds = Dataset(name='%s/%s' % (prefix, name), zero_origin=False)

        pcd_path = os.path.join(ds.path, 'data_3d_semantics', 'train', ds.seq, 'static')
        pcdFile = os.path.join(pcd_path, np.sort(os.listdir(pcd_path))[0])
        data = read_ply(pcdFile)
        points = np.vstack((data['x'], data['y'], data['z'])).T
        # color = np.vstack((data['red'], data['green'], data['blue'])).T
        globalIds = data['instance']
        colored_cloud = ColoredCloud(seq=int(name.split('_')[0]))
        color = colored_cloud.assignColor(globalIds=globalIds, gtType='semantic')

        pcd = o3d.geometry.PointCloud()
        for k, (cloud, pose) in enumerate(ds):
            cloud = structured_to_unstructured(cloud[['x', 'y', 'z']])
            # filter cloud
            cloud = filtered_cloud(cloud, cfg)

            # transform cloud to common map coordinate frame
            cloud_map = np.matmul(cloud[:, :3], pose[:3, :3].T) + pose[:3, 3:].T

            pcd.points.extend(o3d.utility.Vector3dVector(cloud_map))

        # choose nearest point from colored batch cloud
        pcd_colored = o3d.geometry.PointCloud()
        tree = scipy.spatial.cKDTree(np.asarray(pcd.points))
        dists, idxs = tree.query(points, k=1)
        mask = np.logical_and(dists >= 0, dists <= 0.1)
        cloud_filtered = np.asarray(points[mask], dtype=np.float)
        color_filtered = np.asarray(color[mask], dtype=np.float) / color.max()
        pcd_colored.points = open3d.utility.Vector3dVector(cloud_filtered)
        pcd_colored.colors = open3d.utility.Vector3dVector(color_filtered)

        # o3d.visualization.draw_geometries([pcd, pcd_colored])
        o3d.visualization.draw_geometries([pcd_colored])


def stack_colored_submaps():
    from tqdm import tqdm

    fileCameraToLidar = os.path.join(data_dir, 'calibration', 'calib_cam_to_velo.txt')
    T_cam2lidar = loadCalibrationRigid(fileCameraToLidar)
    T_lidar2cam = np.linalg.inv(T_cam2lidar)

    for name in dataset_names:
        seq = int(name.split('_')[0])
        print('Sequence: %i' % seq)

        poses_path = os.path.join(data_dir, 'data_poses', '2013_05_28_drive_%04d_sync' % seq, 'cam0_to_world.txt')
        pcd_path = os.path.join(data_dir, 'data_3d_semantics', 'train', '2013_05_28_drive_%04d_sync' % seq, 'static')
        # pcd_path = os.path.join(data_dir, 'data_3d_semantics', 'train', '2013_05_28_drive_%04d_sync' % seq, 'dynamic')
        data_poses = np.loadtxt(poses_path, dtype=np.float32)
        poses = data_poses[:, 1:].reshape((-1, 4, 4)) @ T_lidar2cam
        ids = [int(i) for i in data_poses[:, 0]]

        pcd = open3d.geometry.PointCloud()
        poses_pcd = open3d.geometry.PointCloud()

        for pcdFile in tqdm(os.listdir(pcd_path)):
            # print('Opening point cloud from %s ...' % pcdFile)
            pcdFile = os.path.join(pcd_path, pcdFile)
            start_i, end_i = pcdFile.split('/')[-1].split('_')
            start_i = int(start_i)
            end_i = int(end_i.replace('.ply', ''))
            print('Poses indexes from %i to %i' % (start_i, end_i))

            # try:
            start_i = ids.index(start_i - 1)
            end_i = ids.index(end_i - 1)
            # except:
            #     print('Cannot find start %i or end %i indexes in list: %s' % (start_i, end_i, ids))
            #     continue

            data = read_ply(pcdFile)
            points = np.vstack((data['x'], data['y'], data['z'])).T
            color = np.vstack((data['red'], data['green'], data['blue'])).T
            pcd.points.extend(open3d.utility.Vector3dVector(points))
            pcd.colors.extend(open3d.utility.Vector3dVector(color.astype(np.float) / 255.))

            poses_pcd.points.extend(open3d.utility.Vector3dVector(poses[start_i:end_i, :3, 3]))
            color = np.zeros_like(poses[start_i:end_i, :3, 3]) + np.asarray([0, 1, 0])
            poses_pcd.colors.extend(open3d.utility.Vector3dVector(color.astype(float)))

            # pcd = pcd.voxel_down_sample(voxel_size=0.5)
            # open3d.visualization.draw_geometries([pcd, poses_pcd])

        pcd = pcd.voxel_down_sample(voxel_size=0.5)
        open3d.visualization.draw_geometries([pcd, poses_pcd])

    exit()


def create_semantic_kitti360():
    from depth_correction.config import Config
    from depth_correction.preproc import filtered_cloud
    from tqdm import tqdm

    def transform_cloud(cloud, Tr):
        cloud_transformed = np.matmul(cloud[:, :3], Tr[:3, :3].T) + Tr[:3, 3:].T
        return cloud_transformed

    cfg = Config(min_depth=0.01, max_depth=150, grid_res=0.0)
    for name in dataset_names:
        ds = Dataset(name='%s/%s' % (prefix, name), zero_origin=False)

        # folders to save clouds and semantic labels
        pts_folder = os.path.join(data_dir, 'SemanticKITTI-360', ds.seq, 'velodyne')
        os.makedirs(pts_folder, exist_ok=True)
        labels_folder = os.path.join(data_dir, 'SemanticKITTI-360', ds.seq, 'labels')
        os.makedirs(labels_folder, exist_ok=True)

        # save poses
        poses_file = os.path.join(data_dir, 'SemanticKITTI-360', ds.seq, 'poses.txt')
        np.savetxt(poses_file, ds.poses.reshape([-1, 16])[:, :12])

        pcd_path = os.path.join(ds.path, 'data_3d_semantics', 'train', ds.seq, 'static')
        pcdFile = os.path.join(pcd_path, np.sort(os.listdir(pcd_path))[0])
        data = read_ply(pcdFile)
        points = np.vstack((data['x'], data['y'], data['z'])).T
        # color = np.vstack((data['red'], data['green'], data['blue'])).T
        globalIds = data['instance']
        semanticIds, _ = global2local(globalIds)
        color = np.zeros((semanticIds.size, 3))
        for uid in np.unique(semanticIds):
            color[semanticIds == uid] = id2label[uid].color

        for k, (cloud, pose) in tqdm(enumerate(ds)):
            cloud = structured_to_unstructured(cloud[['x', 'y', 'z']])
            # filter cloud
            cloud = filtered_cloud(cloud, cfg)

            # transform cloud to common map coordinate frame
            cloud_map = transform_cloud(cloud, pose)

            # choose nearest point from colored batch cloud
            tree = scipy.spatial.cKDTree(cloud_map)
            dists, idxs = tree.query(points, k=1)
            mask = np.logical_and(dists >= 0, dists <= 0.05)
            cloud_sampled = np.asarray(points[mask], dtype=float)
            # transform cloud back to lidar frame
            cloud_sampled = transform_cloud(cloud_sampled, np.linalg.inv(pose))
            ids_sampled = np.asarray(semanticIds[mask], dtype=np.uint8)

            # save sampled point clouds and labels
            with open(os.path.join(pts_folder, '%010d.bin' % ds.ids[k]), 'wb') as f:
                np.save(f, cloud_sampled)
            with open(os.path.join(labels_folder, '%010d.label' % ds.ids[k]), 'wb') as f:
                np.save(f, ids_sampled)


def inspect_semantic_kitti360():
    import open3d as o3d

    for seq in ['2013_05_28_drive_0000_sync']:
        pts_folder = os.path.join(data_dir, 'SemanticKITTI-360', seq, 'velodyne')

        for pts_file in os.listdir(pts_folder):
            pts_file = os.path.join(pts_folder, pts_file)
            ids_file = pts_file.replace('velodyne', 'labels').replace('.bin', '.label')

            points = np.load(pts_file)
            ids = np.load(ids_file)

            color = np.zeros((ids.size, 3))
            for uid in np.unique(ids):
                color[ids == uid] = id2label[uid].color

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = open3d.utility.Vector3dVector(color / color.max())

            o3d.visualization.draw_geometries([pcd])


def main():
    # stack_colored_submaps()
    visualize_datasets()
    # visualize_colored_submaps()
    # visualize_colored_datasets()
    # create_semantic_kitti360()
    # inspect_semantic_kitti360()


if __name__ == '__main__':
    main()

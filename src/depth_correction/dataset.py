from __future__ import absolute_import, division, print_function
from .config import Config
from .depth_cloud import DepthCloud
from .model import BaseModel
from .utils import cached
import numpy as np
from numpy.lib.recfunctions import merge_arrays, unstructured_to_structured

default_rng = np.random.default_rng(135)


def box_point_cloud(size=(1.0, 1.0, 0.0), density=100.0, rng=default_rng):
    size = np.asarray(size).reshape((1, 3))
    measure = np.prod([s for s in size.flatten() if s])
    n_pts = int(np.ceil(measure * density))
    x = size * rng.uniform(-0.5, 0.5, (n_pts, 3))
    return x


class GroundPlaneDataset(object):
    def __init__(self, name=None, n=10, size=(5.0, 5.0, 0.0), step=1.0, height=1.0, density=100.0, model=None,
                 **kwargs):
        """Dataset composed of multiple measurements of ground plane.

        :param n: Number of viewpoints.
        :param step: Distance between neighboring viewpoints.
        :param height: Sensor height above ground plane.
        :param density: Point density in unit volume/area.
        :param model: Ground-truth correction model; inverse will be applied to the points.
        """
        if name:
            parts = name.split('/')
            if len(parts) == 2:
                assert parts[0] == 'ground_plane'
                name = parts
            # TODO: Parse other params from name.
            if isinstance(name, str):
                n = int(name)

        self.model = model

        self.n = n
        self.size = size
        self.step = step
        self.height = height
        self.density = density
        self.ids = list(range(self.n))

    def local_cloud(self, id):
        rng = np.random.default_rng(id)
        pts = box_point_cloud(size=self.size, density=self.density, rng=rng)
        vps = np.zeros_like(pts)
        vps[:, 2] = self.height
        normals = np.zeros_like(pts)
        normals[:, 2] = 1.0

        if self.model is not None:
            assert isinstance(self.model, BaseModel)
            dc = DepthCloud.from_points(pts, vps=vps)
            assert isinstance(dc, DepthCloud)
            dc.normals = normals
            dc.update_incidence_angles()
            # dc = self.model(dc)
            dc = self.model.inverse(dc)
            pts = dc.to_points().detach().numpy()
            # print(pts.shape)

        pts = unstructured_to_structured(pts, names=['x', 'y', 'z'])
        vps = unstructured_to_structured(vps, names=['vp_%s' % f for f in 'xyz'])
        cloud = merge_arrays([pts, vps], flatten=True)

        return cloud

    def cloud_pose(self, id):
        pose = np.eye(4)
        pose[0, 3] = id * self.step
        return pose

    def __getitem__(self, i):
        if isinstance(i, int):
            id = self.ids[i]
            cloud = self.local_cloud(id)
            pose = self.cloud_pose(id)
            return cloud, pose

        ds = GroundPlaneDataset(n=self.n, size=self.size, step=self.step, height=self.height, density=self.density,
                                model=self.model)
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


def dataset_by_name(name):
    parts = name.split('/')
    if len(parts) == 2:
        name = parts[0]

    if name == 'ground_plane':
        return GroundPlaneDataset
    if name == 'asl_laser':
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

from __future__ import absolute_import, division, print_function
import numpy as np
from numpy.lib.recfunctions import merge_arrays, structured_to_unstructured, unstructured_to_structured
from ros_numpy import msgify
from sensor_msgs.msg import PointCloud2
import torch

__all__ = [
    'PointCloud',
]


class PointCloud(object):
    """Generic point cloud with multiple fields."""

    def __init__(self, **kwargs):
        """Create point cloud with multiple fields.

        Fields can be of type torch.Tensor, with the same size in the first dimension,
        or lists of arbitrary objects.

        >>> PointCloud(x=np.ones((100, 3)), vp=np.zeros((100, 3))).size()
        100
        >>> PointCloud(x=np.ones((5, 3)), label=[1, 2, 3, 4, 5]).size()
        5
        """
        for k, v in kwargs.items():
            if isinstance(v, tuple):
                v = list(v)
            elif isinstance(v, np.ndarray):
                v = torch.as_tensor(v)
            self[k] = v
        self.check()

    def __getitem__(self, item):
        """
        :param item:
        :return:

        >>> PointCloud(x=[1, 2, 3]).x
        [1, 2, 3]
        """
        if isinstance(item, str):
            return getattr(self, item)
        # Filter fields if item is a non-empty list of strings.
        if isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], str):
            other = PointCloud(**{k: self[k] for k in item})
        # Filter points using an index (slice, list, mask).
        else:
            other = PointCloud(**{k: v[item] for k, v in self.fields().items()})
        return other

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __add__(self, other):
        return PointCloud.concatenate([self, other])

    def __len__(self):
        return self.size()

    def __str__(self):
        return 'PointCloud(%i)' % self.size()

    def fields(self):
        return vars(self)

    def field_names(self):
        return self.fields().keys()

    def copy(self):
        """Create shallow copy of the cloud.

        >>> list(PointCloud(x=np.ones((10, 3))).copy().field_names())
        ['x']
        """
        other = PointCloud(**self.fields())
        return other

    def clone(self):
        """Create deep copy of the cloud.
        Gradients are still propagated if detach is not called."""
        other = PointCloud(**{k: v.clone() for k, v in self.fields().items()})
        return other

    def any_value(self):
        for v in self.fields().values():
            if v is not None:
                return v
        return None

    def size(self):
        v = self.any_value()
        if v is None:
            return 0
        # return v.shape[0]
        return len(v)

    def empty(self):
        """Is point cloud empty?

        :return: Bool, whethe

        >>> PointCloud().empty()
        True
        """
        return self.size() == 0

    def check(self):
        for v in self.fields().values():
            # assert v.shape[0] == self.size()
            assert len(v) == self.size()

    def to_structured_array(self):
        arrs = []
        for k, v in self.fields():
            arr = v.detach().cpu().numpy()
            arr = unstructured_to_structured(arr, names=[k])
            arrs.append(arr)
        merge_arrays(arrs, flatten=True)
        return None

    @staticmethod
    def concatenate(clouds, fields=None):
        """Concatenate multiple point clouds.

        :param clouds:
        :return:

        >>> PointCloud.concatenate([PointCloud(x=np.ones((10, 3))), PointCloud(x=np.ones((5, 3)))]).size()
        15
        """
        if clouds is None or len(clouds) == 0:
            return PointCloud()
        if not fields:
            fields = clouds[0].field_names()
        kwargs = {}
        for k in fields:
            kwargs[k] = torch.cat([c[k] for c in clouds])
        cld = PointCloud(**kwargs)
        return cld

    @staticmethod
    def from_structured_array(arr):
        """Create point cloud from structured array."""
        kwargs = {k: structured_to_unstructured(arr[[k]]) for k in arr.dtype.names}
        cloud = PointCloud(**kwargs)
        return cloud

    def to_msg(self):
        return msgify(PointCloud2, self.to_structured_array())

    def to(self, device=None, dtype=None, strict=True):
        """Move to device or change type.

        :param device:
        :param dtype:
        :param strict: Move only compatible type (floating-point / integer).
        :return: A copy with modified fields.

        >>> tuple(PointCloud(x=np.ones((100, 3))).to(dtype=torch.float32)['x'].shape)
        (100, 3)
        """
        kv = {}
        for k, v in self.fields().items():
            if v is None or (device is None and dtype is None):
                kv[k] = None
            elif not dtype or (strict and dtype.is_floating_point != v.dtype.is_floating_point):
                kv[k] = v.to(device=device)
            else:
                kv[k] = v.to(dtype=dtype, device=device)
        cloud = PointCloud(**kv)
        return cloud

    def cpu(self):
        return self.to(device=torch.device('cpu'))

    def gpu(self):
        return self.to(device=torch.device('cuda:0'))

    def device(self):
        if self.empty():
            return None
        return self.any_value().device

    def type(self, dtype=None):
        return self.to(dtype=dtype)

    def float(self):
        return self.type(torch.float32)

    def double(self):
        return self.type(torch.float64)

    def detach(self):
        kwargs = {k: v.detach() for k, v in self.fields().items()}
        return PointCloud(**kwargs)


def test():
    import doctest
    doctest.testmod()


def main():
    test()


if __name__ == '__main__':
    main()

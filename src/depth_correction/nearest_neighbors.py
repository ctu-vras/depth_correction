from __future__ import absolute_import, division, print_function
from scipy.spatial import cKDTree
import torch

__all__ = [
    'nearest_neighbors'
]


def nearest_neighbors(points, query, k=None, r=None):
    """Find nearest neighbors of query in points.

    :param points:
    :param query:
    :param k: Number of neighbors.
    :param r: Radius in which to find neighbors.
    :return:
    """
    assert isinstance(points, torch.Tensor)
    assert isinstance(query, torch.Tensor)
    assert k or r

    # Convert to numpy and squeeze leading dimensions.
    points = points.detach().numpy().reshape([None, points.shape[-1]])
    query = query.detach().numpy().reshape([None, query.shape[-1]])

    # Create index and query points.
    index = cKDTree(points)
    if k is not None:
        dist, ind = index.query(query, k)
    elif r is not None:
        dist, ind = index.query_ball_point(query, r)

    return dist, ind

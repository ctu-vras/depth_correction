from __future__ import absolute_import, division, print_function
from scipy.spatial import cKDTree
import torch
# from pytorch3d.ops.knn import knn_points

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

    points = points.reshape([-1, points.shape[-1]])
    query = query.reshape([-1, points.shape[-1]])
    # if points.device == torch.device('cpu'):
    # Convert to numpy and squeeze leading dimensions.
    points = points.detach().numpy()
    query = query.detach().numpy()

    # Create index and query points.
    index = cKDTree(points)
    if k is not None:
        dist, ind = index.query(query, k)
    elif r is not None:
        dist, ind = None, index.query_ball_point(query, r)

    # else:
    #     # TODO: currently doesn't support neighbors in a radius vicinity
    #     assert isinstance(k, int)
    #
    #     if points.shape[1] != query.shape[1]:
    #         raise ValueError("points and querry must have the same point dimension.")
    #
    #     with torch.no_grad():
    #         dist, ind, _ = knn_points(query[None], points[None], K=k)
    #         dist = torch.sqrt(dist).squeeze()
    #         ind = ind.squeeze()

    return dist, ind


# if __name__ == '__main__':
#     # torch.manual_seed(0)
#     points = 5*torch.rand(size=(1, 5, 3), dtype=torch.float32)
#     query = 6*torch.rand(size=(1, 4, 3), dtype=torch.float32)
#     print(nearest_neighbors(points, query, k=1))
#
#     points = points.to(torch.device('cuda:0'))
#     query = query.to(torch.device('cuda:0'))
#     print(nearest_neighbors(points, query, k=1))

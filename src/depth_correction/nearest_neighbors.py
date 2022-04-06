from __future__ import absolute_import, division, print_function
import numpy as np
# from pytorch3d.ops.knn import knn_points
from scipy.spatial import cKDTree
import torch

__all__ = [
    'nearest_neighbors'
]


def nearest_neighbors(points, query, k=None, r=None, n_jobs=-1):
    """Find nearest neighbors of query in points.

    :param points: Reference points in rows.
    :param query: Query points in rows.
    :param k: Number of neighbors.
    :param r: Radius in which to find neighbors.
    :return: Tuple with distances and indices. Distances may be None. Missing neighbors are indicated by -1.
    """
    assert isinstance(points, torch.Tensor)
    assert isinstance(query, torch.Tensor)
    assert k or r

    points = points.reshape([-1, points.shape[-1]])
    query = query.reshape([-1, points.shape[-1]])
    # if points.device == torch.device('cpu'):
    device = points.device
    # Convert to numpy and squeeze leading dimensions.
    points = points.detach().cpu().numpy()
    query = query.detach().cpu().numpy()

    # Create index and query points.
    invalid_dist = float('nan')
    invalid_index = -1
    index = cKDTree(points)
    if k:
        dist, ind = index.query(query, k, n_jobs=n_jobs, **({'distance_upper_bound': r} if r else {}))
        ind[ind == index.n] = invalid_index
    elif r:
        dist, ind = None, index.query_ball_point(query, r, n_jobs=n_jobs)
    else:
        raise ValueError('Invalid neighborhood definition.')

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

    # Convert distances and indices to fixed size array if using cuda.
    # if device.type == 'cuda':
    if ind.dtype == np.object:
        n = max([len(x) for x in ind])
        if dist is not None:
            dist = np.array([x + (n - len(x)) * [invalid_dist] for x in dist])
        ind = np.array([x + (n - len(x)) * [invalid_index] for x in ind])

    # Move nearest neighbor output to input device.
    if dist is not None:
        dist = torch.from_numpy(dist).to(device)
    ind = torch.from_numpy(ind).to(device)

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

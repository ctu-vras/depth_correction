from __future__ import absolute_import, division, print_function
from .nearest_neighbors import nearest_neighbors
import torch

__all__ = [
    'min_eig_loss',
    'neighbor_cov',
    'neighbor_fun',
    'reduce',
    'trace_loss',
]


def reduce(x, reduction='mean'):
    assert reduction in ('none', 'mean', 'sum')

    if reduction == 'mean':
        x = x.mean()
    elif reduction == 'sum':
        x = x.sum()

    return x


def neighbor_fun(points, fun, query=None, k=None, r=None):
    assert isinstance(points, torch.Tensor)
    assert isinstance(query, torch.Tensor)
    assert callable(fun)
    assert k is None or (isinstance(k, int) and k >= 1)
    assert r is None or (isinstance(r, float) and r > 0.0)

    dist, ind = nearest_neighbors(points, query, k=k, r=r)

    # TODO: Allow batch dimension.
    n = query.shape[-1]
    result = []
    for i in range(n):
        nn = torch.index_select(points, 0, ind[i])
        q = query[i]
        result.append(fun(nn, q))

    return result


def neighbor_cov(points, query=None, k=None, r=None, correction=1):
    fun = lambda p, q: torch.cov(p.transpose(-1, -2), correction=correction)
    cov = neighbor_fun(points, fun, query=query, k=k, r=r)
    cov = torch.stack(cov)
    return cov


def min_eig_loss(points, query=None, k=None, r=None, reduction='mean'):
    # fun = lambda x: torch.linalg.eigvalsh(torch.cov(x.transpose(-1, -2)))[0]
    # loss = neighbor_fun(points, fun, query=query, k=k, r=r)
    # loss = torch.stack(loss)
    # loss = reduce(loss, reduction=reduction)

    # Parallelize eigvals.
    cov = neighbor_cov(points, query=query, k=k, r=r)
    loss = torch.linalg.eigvalsh(cov)[..., 0]
    loss = reduce(loss, reduction=reduction)
    return loss


def trace_loss(points, query=None, k=None, r=None, reduction='mean'):
    fun = lambda p, q: torch.cov(p.transpose(-1, -2)).trace()
    loss = neighbor_fun(points, fun, query=query, k=k, r=r)
    loss = torch.stack(loss)
    loss = reduce(loss, reduction=reduction)
    return loss

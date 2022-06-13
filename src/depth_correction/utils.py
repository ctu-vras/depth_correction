from __future__ import absolute_import, division, print_function
from matplotlib import cm
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from timeit import default_timer as timer
import torch
from pytorch3d.io import load_ply, load_obj
from pytorch3d.structures import Meshes


__all__ = [
    'covs',
    'map_colors',
    'timer',
    'timing',
    'trace',
]


def map_colors(values, colormap=cm.gist_rainbow, min_value=None, max_value=None):
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)
    # if not isinstance(colormap, torch.Tensor):
    #     colormap = torch.tensor(colormap, dtype=torch.float64)
    # assert colormap.shape[1] == (2, 3)
    # assert callable(colormap)
    assert callable(colormap) or isinstance(colormap, torch.Tensor)
    if min_value is None:
        min_value = values.min()
    if max_value is None:
        max_value = values.max()
    scale = max_value - min_value
    a = (values - min_value) / scale if scale > 0.0 else values - min_value
    if callable(colormap):
        colors = colormap(a.squeeze())[:, :3]
        return colors
    # TODO: Allow full colormap with multiple colors.
    assert isinstance(colormap, torch.Tensor)
    num_colors = colormap.shape[0]
    a = a.reshape([-1, 1])
    if num_colors == 2:
        # Interpolate the two colors.
        colors = (1 - a) * colormap[0:1] + a * colormap[1:]
    else:
        # Select closest based on scaled value.
        i = torch.round(a * (num_colors - 1))
        colors = colormap[i]
    return colors


def timing(f):
    def timing_wrapper(*args, **kwargs):
        t0 = timer()
        try:
            ret = f(*args, **kwargs)
            return ret
        finally:
            t1 = timer()
            print('%s %.6f s' % (f.__name__, t1 - t0))

    return timing_wrapper


def hashable(obj):
    if isinstance(obj, (list, tuple)):
        obj = tuple(hashable(o) for o in obj)
    elif isinstance(obj, dict):
        obj = hashable(sorted(obj.items()))
    elif isinstance(obj, slice):
        obj = obj.start, obj.stop, obj.step
    elif isinstance(obj, np.ndarray):
        obj = hashable(obj.tolist())
    return obj


cache = {}


def cached(f):
    """Create caching wrapper of the function.
    Function and its inputs are used as keys to store or retrieve outputs."""
    def cached_wrapper(*args, **kwargs):
        key = (hashable(f), hashable(args), hashable(kwargs))
        if key not in cache:
            print('Evaluating key %s.' % (key,))
            try:
                ret = f(*args, **kwargs)
                cache[key] = ret, None
            except Exception as ex:
                cache[key] = None, ex
        else:
            print('Using cached key %s.' % (key,))
        ret, ex = cache[key]
        if ex is not None:
            raise ex
        return ret

    return cached_wrapper


def covs(x, obs_axis=-2, var_axis=-1, center=True, correction=True, weights=None):
    """Create covariance matrices from multiple samples."""
    assert isinstance(x, torch.Tensor)
    assert obs_axis != var_axis
    assert weights is None or isinstance(weights, torch.Tensor)

    # Use sum of provided weights or number of observation for normalization.
    if weights is not None:
        w = weights.sum(dim=obs_axis, keepdim=True)
    else:
        w = x.shape[obs_axis]

    # Center the points if requested.
    if center:
        if weights is not None:
            xm = (weights * x).sum(dim=obs_axis, keepdim=True) / w
        else:
            xm = x.mean(dim=obs_axis, keepdim=True)
        xc = x - xm
    else:
        xc = x

    # Construct possibly weighted xx = x * x^T.
    var_axis_2 = var_axis + 1 if var_axis >= 0 else var_axis - 1
    xx = xc.unsqueeze(var_axis) * xc.unsqueeze(var_axis_2)
    if weights is not None:
        xx = weights.unsqueeze(var_axis) * xx

    # Compute weighted average of x * x^T to get cov.
    if obs_axis < var_axis and obs_axis < 0:
        obs_axis -= 1
    elif obs_axis > var_axis and obs_axis > 0:
        obs_axis += 1
    xx = xx.sum(dim=obs_axis)
    if correction:
        w = w - 1
    w = w.clamp(1e-6, None)
    xx = xx / w

    return xx


def trace(x, dim1=-2, dim2=-1):
    tr = x.diagonal(dim1=dim1, dim2=dim2).sum(dim=-1)
    return tr


def nearest_orthonormal(A):
    U, _, Vt = np.linalg.svd(A)
    A = U @ Vt
    return A


def fix_transform(T):
    T_fixed = T.copy()
    T_fixed[:-1, :-1] = nearest_orthonormal(T[:-1, :-1])
    # print('fix:\n', T_fixed - T)
    return T


def rotation_angle(T):
    R = T[:-1, :-1]
    angle = np.arccos((np.trace(R) - 1.0) / 2.0).item()
    return angle


def translation_norm(T):
    t = T[:-1, -1:]
    norm = np.linalg.norm(t).item()
    return norm


def transform_inv(T):
    T_inv = np.eye(T.shape[0])
    R = T[:-1, :-1]
    t = T[:-1, -1:]
    T_inv[:-1, :-1] = R.T
    T_inv[:-1, -1:] = -R.T @ t
    return T_inv


def delta_transform(T_0, T_1):
    """Delta transform D s.t. T_1 = T_0 * D."""
    delta = np.linalg.solve(T_0, T_1)
    # delta = transform_inv(T_0) @ T_1
    return delta


def e2p(x, axis=-1):
    assert isinstance(x, np.ndarray)
    assert isinstance(axis, int)
    h_size = list(x.shape)
    h_size[axis] = 1
    h = np.ones(h_size, dtype=x.dtype)
    xh = np.concatenate((x, h), axis=axis)
    return xh


def p2e(xh, axis=-1):
    assert isinstance(xh, np.ndarray)
    assert isinstance(axis, int)
    if axis != -1:
        xh = xh.swapaxes(axis, -1)
    x = xh[..., :-1]
    if axis != -1:
        x = x.swapaxes(axis, -1)
    return x


def transform(T, x_struct):
    assert isinstance(T, np.ndarray)
    assert T.shape == (4, 4)
    assert isinstance(x_struct, np.ndarray)
    x_struct = x_struct.copy()
    fields_op = []
    for fs, op in [[['x', 'y', 'z'], 'Rt'],
                   [['vp_x', 'vp_y', 'vp_z'], 'Rt'],
                   [['normal_x', 'normal_y', 'normal_z'], 'R']]:
        if fs[0] in x_struct.dtype.names:
            fields_op.append((fs, op))
    for fs, op in fields_op:
        x = structured_to_unstructured(x_struct[fs])
        if op == 'Rt':
            x = p2e(np.matmul(e2p(x), T.T))
        elif op == 'R':
            x = np.matmul(x, T[:-1, :-1].T)
        x_struct[fs] = unstructured_to_structured(x, names=fs)
    return x_struct


def load_mesh(mesh_path):
    if mesh_path[-3:] == 'obj':
        pts, faces, _ = load_obj(mesh_path)
        mesh = Meshes(verts=[pts], faces=[faces.verts_idx])
    elif mesh_path[-3:] == 'ply':
        pts, faces = load_ply(mesh_path)
        mesh = Meshes(verts=[pts], faces=[faces])
    else:
        raise ValueError('Supported mesh formats are *.obj or *.ply')
    return mesh

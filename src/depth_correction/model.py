from __future__ import absolute_import, division, print_function
from .config import Config
from .depth_cloud import DepthCloud
import numpy as np
import torch

__all__ = [
    'BaseModel',
    'InvCos',
    'Linear',
    'load_model',
    'model_by_name',
    'Polynomial',
    'ScaledPolynomial',
    'ScaledInvCos',
]


def load_model(class_name: str=None,
               model_args=None,
               model_kwargs=None,
               state_dict: (dict, str)=None,
               device: (str, torch.device)=None,
               cfg: Config=None,
               eval_mode: bool=True):

    if cfg is not None:
        if class_name is None:
            class_name = cfg.model_class
        if model_args is None:
            model_args = cfg.model_args[:] if cfg.model_args else []
        if model_kwargs is None:
            model_kwargs = cfg.model_kwargs.copy() if cfg.model_kwargs else {}
        if state_dict is None:
            state_dict = cfg.model_state_dict
        if device is None:
            device = cfg.device

    if model_args is None:
        model_args = []
    if model_kwargs is None:
        model_kwargs = {}

    if isinstance(state_dict, str) and state_dict:
        print('Loading model state from %s.' % state_dict)
        state_dict = torch.load(state_dict)

    if isinstance(device, str):
        device = torch.device(device)

    if 'device' not in model_kwargs:
        model_kwargs['device'] = device

    Class = model_by_name(class_name)
    model = Class(*model_args, **model_kwargs)
    assert isinstance(model, BaseModel)

    if state_dict:
        model.load_state_dict(state_dict)
    # print('Using model: %s.' % model)

    if eval_mode:
        model.eval()

    model.to(device)
    # print('Model moved to %s.' % device)
    return model


class BaseModel(torch.nn.Module):

    def __init__(self, device=torch.device('cpu')):
        super(BaseModel, self).__init__()
        self.device = device

    def forward(self, dc: DepthCloud) -> DepthCloud:
        dc = self.correct_depth(dc, dc.mask)
        return dc

    def correct_depth(self, dc: DepthCloud, mask=None) -> DepthCloud:
        return dc

    def inverse(self, dc: DepthCloud, mask=None) -> DepthCloud:
        return dc

    def __str__(self):
        return 'BaseModel()'

    def construct(self, *args, **kwargs):
        return type(self)(*args, **kwargs)

    def detach(self):
        kwargs = {k: v.detach() for k, v in self.named_parameters()}
        return self.construct(**kwargs)

    def clone(self):
        kwargs = {k: v.clone() for k, v in self.named_parameters()}
        return self.construct(**kwargs)

    def plot(self, ax, max_angle=np.deg2rad(89.0), **kwargs):
        n_pts = 100
        if 'label' not in kwargs:
            kwargs['label'] = str(self)
        with torch.no_grad():
            cloud = DepthCloud.from_points(torch.ones((n_pts, 3)) / torch.sqrt(torch.tensor(3.0)))
            cloud.inc_angles = torch.as_tensor(np.linspace(0, max_angle, n_pts))[:, None]
            ax.plot(np.rad2deg(cloud.inc_angles.numpy()).flatten(), self(cloud).depth.numpy().flatten(), **kwargs)
            ax.set_xlabel('Incidence Angle [deg]')
            ax.set_ylabel('Depth [m]')


class Linear(BaseModel):

    def __init__(self, w0=1.0, w1=0.0, b=0.0, uniform_weights=False, device=torch.device('cpu')):
        super(Linear, self).__init__(device=device)

        assert isinstance(w0, (float, torch.Tensor))
        assert isinstance(w1, (float, torch.Tensor))
        assert isinstance(b, (float, torch.Tensor))

        if uniform_weights:
            w0 = torch.nn.init.uniform(torch.as_tensor(w0), -0.95, 1.05)
            w1 = torch.nn.init.uniform(torch.as_tensor(w1), -0.05, 0.05)
            b = torch.nn.init.uniform(torch.as_tensor(b), -0.05, 0.05)

        self.w0 = torch.nn.Parameter(torch.as_tensor(w0, device=self.device))
        self.w1 = torch.nn.Parameter(torch.as_tensor(w1, device=self.device))
        self.b = torch.nn.Parameter(torch.as_tensor(b, device=self.device))

    def correct_depth(self, dc: DepthCloud, mask=None) -> DepthCloud:
        assert dc.inc_angles is not None
        dc_corr = dc.copy()
        if mask is None:
            dc_corr.depth = self.w0 * dc_corr.depth + self.w1 * dc_corr.inc_angles + self.b
        else:
            # Avoid modifying depth in-place.
            dc_corr.depth = dc_corr.depth.clone()
            dc_corr.depth[mask] = self.w0 * dc_corr.depth[mask] + self.w1 * dc_corr.inc_angles[mask] + self.b
        return dc_corr

    def inverse(self, dc: DepthCloud, mask=None) -> DepthCloud:
        raise NotImplementedError()

    def __str__(self):
        return 'Linear(%.6g, %.6g, %.6g)' % (self.w0.item(), self.w1.item(), self.b.item())


class Polynomial(BaseModel):

    def __init__(self, p0=None, p1=None, w=None, exponent=None, learnable_exponents=False,
                 device=torch.device('cpu')):
        super().__init__(device=device)

        if exponent is None:
            assert w is None
            self.legacy = True
            exponent = [2.0, 4.0]
            w = [p0 or 0.0, p1 or 0.0]
        else:
            self.legacy = False

        if w is None:
            w = [0.0] * len(exponent)
        w = torch.as_tensor(w, dtype=torch.float64, device=device).view((1, -1))
        assert w.numel() == len(exponent)
        self.w = torch.nn.Parameter(w)

        exponent = torch.as_tensor(exponent, dtype=torch.float64, device=device).view((1, -1))
        self.exponent = torch.nn.Parameter(exponent) if learnable_exponents else exponent

    def bias(self, inc_angles):
        assert inc_angles.dim() == 2
        assert inc_angles.shape[1] == 1
        x = torch.pow(inc_angles, self.exponent)
        bias = torch.matmul(x, self.w.t()).view((-1, 1))
        return bias

    def correct_depth(self, dc: DepthCloud, mask=None) -> DepthCloud:
        assert dc.inc_angles is not None
        dc_corr = dc.copy()
        if mask is None:
            bias = self.bias(dc.inc_angles)
            dc_corr.depth = dc_corr.depth - bias
        else:
            bias = self.bias(dc.inc_angles[mask])
            # Avoid modifying depth in-place.
            dc_corr.depth = dc_corr.depth.clone()
            dc_corr.depth[mask] = dc_corr.depth[mask] - bias
        return dc_corr

    def inverse(self, dc: DepthCloud, mask=None) -> DepthCloud:
        assert dc.inc_angles is not None
        dc_corr = dc.copy()
        if mask is None:
            bias = self.bias(dc.inc_angles)
            dc_corr.depth = dc_corr.depth / (1. - bias)
        else:
            # Avoid modifying depth in-place.
            bias = self.bias(dc.inc_angles[mask])
            dc_corr.depth = dc_corr.depth.clone()
            dc_corr.depth[mask] = dc_corr.depth[mask] + bias
        return dc_corr

    def to(self, *args, **kwargs):
        ret = super().to(*args, **kwargs)
        if not isinstance(ret.exponent, torch.nn.Parameter):
            ret.exponent = ret.exponent.to(*args, **kwargs)
        return ret

    def __str__(self):
        if self.legacy:
            return 'Polynomial(%.6g, %.6g)' % (self.p0.item(), self.p1.item())
        return 'Polynomial(%s)' % ', '.join('%.6gx^%.6g' % (w, e)
                                                  for w, e in zip(self.w.flatten(), self.exponent.flatten()))


class ScaledPolynomial(BaseModel):

    def __init__(self, p0=None, p1=None, w=None, exponent=None, learnable_exponents=False,
                 device=torch.device('cpu')):
        super().__init__(device=device)

        if exponent is None:
            assert w is None, w
            self.legacy = True
            exponent = [2.0, 4.0]
            w = [p0 or 0.0, p1 or 0.0]
        else:
            self.legacy = False

        if w is None:
            w = [0.0] * len(exponent)
        elif isinstance(w, float):
            w = [w]
        w = torch.as_tensor(w, dtype=torch.float64, device=device).view((1, -1))
        assert w.numel() == len(exponent), (w, exponent)
        self.w = torch.nn.Parameter(w)

        exponent = torch.as_tensor(exponent, dtype=torch.float64, device=device).view((1, -1))
        self.exponent = torch.nn.Parameter(exponent) if learnable_exponents else exponent

    def bias(self, inc_angles):
        assert inc_angles.dim() == 2
        assert inc_angles.shape[1] == 1
        x = torch.pow(inc_angles, self.exponent)
        bias = torch.matmul(x, self.w.t()).view((-1, 1))
        return bias

    def correct_depth(self, dc: DepthCloud, mask=None) -> DepthCloud:
        assert dc.inc_angles is not None
        dc_corr = dc.copy()
        if mask is None:
            bias = self.bias(dc.inc_angles)
            dc_corr.depth = dc_corr.depth * (1. - bias)
        else:
            bias = self.bias(dc.inc_angles[mask])
            # Avoid modifying depth in-place.
            dc_corr.depth = dc_corr.depth.clone()
            dc_corr.depth[mask] = dc_corr.depth[mask] * (1. - bias)
        return dc_corr

    def inverse(self, dc: DepthCloud, mask=None) -> DepthCloud:
        assert dc.inc_angles is not None
        dc_corr = dc.copy()
        if mask is None:
            bias = self.bias(dc.inc_angles)
            dc_corr.depth = dc_corr.depth / (1. - bias)
        else:
            # Avoid modifying depth in-place.
            bias = self.bias(dc.inc_angles[mask])
            dc_corr.depth = dc_corr.depth.clone()
            dc_corr.depth[mask] = dc_corr.depth[mask] / (1. - bias)
        return dc_corr

    def to(self, *args, **kwargs):
        ret = super().to(*args, **kwargs)
        if not isinstance(ret.exponent, torch.nn.Parameter):
            ret.exponent = ret.exponent.to(*args, **kwargs)
        return ret

    def __str__(self):
        if self.legacy:
            return 'ScaledPolynomial(%.6g, %.6g)' % (self.p0.item(), self.p1.item())
        return 'ScaledPolynomial(%s)' % ', '.join('%.6gx^%.6g' % (w, e)
                                                  for w, e in zip(self.w.flatten(), self.exponent.flatten()))


class InvCos(BaseModel):

    def __init__(self, p0=0.0, device=torch.device('cpu')):
        super(InvCos, self).__init__(device=device)
        p0 = torch.as_tensor(p0, device=device)
        self.p0 = torch.nn.Parameter(p0)

    def correct_depth(self, dc: DepthCloud, mask=None) -> DepthCloud:
        assert dc.inc_angles is not None
        dc_corr = dc.copy()
        if mask is None:
            bias = self.p0 / torch.cos(dc.inc_angles)
            dc_corr.depth = dc_corr.depth - bias
        else:
            bias = self.p0 / torch.cos(dc.inc_angles[mask])
            # Avoid modifying depth in-place.
            dc_corr.depth = dc_corr.depth.clone()
            dc_corr.depth[mask] = dc_corr.depth[mask] - bias
        return dc_corr

    def inverse(self, dc: DepthCloud, mask=None) -> DepthCloud:
        raise NotImplementedError()

    def __str__(self):
        return 'InvCos(%.6g)' % (self.p0.item(),)


class ScaledInvCos(BaseModel):

    def __init__(self, p0=0.0, device=torch.device('cpu')):
        super(ScaledInvCos, self).__init__(device=device)
        self.p0 = torch.nn.Parameter(torch.as_tensor(p0, device=device))

    def correct_depth(self, dc: DepthCloud, mask=None) -> DepthCloud:
        assert dc.inc_angles is not None
        dc_corr = dc.copy()
        if mask is None:
            bias = self.p0 / torch.cos(dc.inc_angles).abs()
            dc_corr.depth = dc_corr.depth * (1. - bias)
        else:
            bias = self.p0 / torch.cos(dc.inc_angles[mask]).abs()
            # Avoid modifying depth in-place.
            dc_corr.depth = dc_corr.depth.clone()
            dc_corr.depth[mask] = dc_corr.depth[mask] * (1. - bias)
        return dc_corr

    def inverse(self, dc: DepthCloud, mask=None) -> DepthCloud:
        assert dc.inc_angles is not None
        dc_corr = dc.copy()
        if mask is None:
            bias = self.p0 / torch.cos(dc.inc_angles).abs()
            dc_corr.depth = dc_corr.depth / (1. - bias)
        else:
            bias = self.p0 / torch.cos(dc.inc_angles[mask]).abs()
            # Avoid modifying depth in-place.
            dc_corr.depth = dc_corr.depth.clone()
            dc_corr.depth[mask] = dc_corr.depth[mask] / (1. - bias)
        return dc_corr

    def __str__(self):
        return 'ScaledInvCos(%.6g)' % (self.p0.item(),)


def model_by_name(name):
    assert name in ('BaseModel', 'InvCos', 'Linear', 'Polynomial', 'ScaledInvCos', 'ScaledPolynomial')
    return globals()[name]

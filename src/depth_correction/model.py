from __future__ import absolute_import, division, print_function
from .config import Config
from .depth_cloud import DepthCloud
from .utils import timing
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
               state_dict: (dict, str)=None,
               device: (str, torch.device)=None,
               cfg: Config=None,
               eval_mode: bool=True):

    if cfg is not None:
        if class_name is None:
            class_name = cfg.model_class
        if state_dict is None:
            state_dict = cfg.model_state_dict
        if device is None:
            device = cfg.device

    if isinstance(state_dict, str) and state_dict:
        print('Loading model state from %s.' % state_dict)
        state_dict = torch.load(state_dict)
    # elif state_dict:
    #     print('Using provided state.')

    if isinstance(device, str):
        device = torch.device(device)

    Class = model_by_name(class_name)
    model = Class()
    assert isinstance(model, BaseModel)

    # if state_dict is not None:
    if state_dict:
        model.load_state_dict(state_dict)
    print('Using model: %s.' % model)

    if eval_mode:
        model.eval()

    model.to(device)
    print('Model moved to %s.' % device)
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

    def __str__(self):
        return 'Linear(%.6g, %.6g, %.6g)' % (self.w0.item(), self.w1.item(), self.b.item())


class Polynomial(BaseModel):

    def __init__(self, p0=0.0, p1=0.0, uniform_weights=False, device=torch.device('cpu')):
        super(Polynomial, self).__init__(device=device)

        assert isinstance(p0, (float, torch.Tensor))
        assert isinstance(p1, (float, torch.Tensor))

        if uniform_weights:
            p0 = torch.nn.init.uniform_(torch.as_tensor(p0), -0.05, 0.05)
            p1 = torch.nn.init.uniform_(torch.as_tensor(p1), -0.05, 0.05)

        self.p0 = torch.nn.Parameter(torch.as_tensor(p0, device=self.device))
        self.p1 = torch.nn.Parameter(torch.as_tensor(p1, device=self.device))

    def correct_depth(self, dc: DepthCloud, mask=None) -> DepthCloud:
        assert dc.inc_angles is not None
        dc_corr = dc.copy()
        if mask is None:
            gamma = dc.inc_angles
            bias = self.p0 * gamma ** 2 + self.p1 * gamma ** 4
            dc_corr.depth = dc_corr.depth - bias
        else:
            gamma = dc.inc_angles[mask]
            bias = self.p0 * gamma ** 2 + self.p1 * gamma ** 4
            # Avoid modifying depth in-place.
            dc_corr.depth = dc_corr.depth.clone()
            dc_corr.depth[mask] = dc_corr.depth[mask] - bias
        return dc_corr

    def __str__(self):
        return 'Polynomial(%.6g, %.6g)' % (self.p0.item(), self.p1.item())


class ScaledPolynomial(BaseModel):

    def __init__(self, p0=0.0, p1=0.0, device=torch.device('cpu')):
        super(ScaledPolynomial, self).__init__(device=device)

        assert isinstance(p0, (float, torch.Tensor))
        assert isinstance(p1, (float, torch.Tensor))

        self.p0 = torch.nn.Parameter(torch.as_tensor(p0, device=self.device))
        self.p1 = torch.nn.Parameter(torch.as_tensor(p1, device=self.device))

    def correct_depth(self, dc: DepthCloud, mask=None) -> DepthCloud:
        assert dc.inc_angles is not None
        dc_corr = dc.copy()
        if mask is None:
            gamma = dc.inc_angles
            bias = self.p0 * gamma ** 2 + self.p1 * gamma ** 4
            dc_corr.depth = dc_corr.depth * (1. - bias)
        else:
            gamma = dc.inc_angles[mask]
            bias = self.p0 * gamma ** 2 + self.p1 * gamma ** 4
            # Avoid modifying depth in-place.
            dc_corr.depth = dc_corr.depth.clone()
            dc_corr.depth[mask] = dc_corr.depth[mask] * (1. - bias)
        return dc_corr

    def __str__(self):
        return 'ScaledPolynomial(%.6g, %.6g)' % (self.p0.item(), self.p1.item())


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

    def __str__(self):
        return 'InvCos(%.6g)' % (self.p0.item(),)


class ScaledInvCos(BaseModel):

    def __init__(self, p0=0.0, device=torch.device('cpu')):
        super(ScaledInvCos, self).__init__(device=device)
        p0 = torch.as_tensor(p0, device=device)
        self.p0 = torch.nn.Parameter(p0)

    def correct_depth(self, dc: DepthCloud, mask=None) -> DepthCloud:
        assert dc.inc_angles is not None
        dc_corr = dc.copy()
        if mask is None:
            bias = self.p0 / torch.cos(dc.inc_angles)
            dc_corr.depth = dc_corr.depth * (1. - bias)
        else:
            bias = self.p0 / torch.cos(dc.inc_angles[mask])
            # Avoid modifying depth in-place.
            dc_corr.depth = dc_corr.depth.clone()
            dc_corr.depth[mask] = dc_corr.depth[mask] * (1. - bias)
        return dc_corr

    def __str__(self):
        return 'ScaledInvCos(%.6g)' % (self.p0.item(),)


def model_by_name(name):
    assert name in ('BaseModel', 'InvCos', 'Linear', 'Polynomial', 'ScaledInvCos', 'ScaledPolynomial')
    return globals()[name]

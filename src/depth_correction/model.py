from __future__ import absolute_import, division, print_function
from .config import Config
from .depth_cloud import DepthCloud
from .utils import timing
import torch

__all__ = [
    'BaseModel',
    'Linear',
    'load_model',
    'Polynomial',
    'ScaledPolynomial',
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

    if isinstance(state_dict, str):
        print('Loading model state from %s.' % state_dict)
        state_dict = torch.load(state_dict)
    elif state_dict is not None:
        print('Using provided state.')

    if isinstance(device, str):
        device = torch.device(device)

    Class = eval(class_name)
    model = Class()
    assert isinstance(model, BaseModel)

    if state_dict is not None:
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

    # @timing
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
        # copy = type(self)()
        # kwargs = {}
        # for k, v in self.named_parameters():
            # copy[k] = v.detach()
            # setattr(self, k, v.detach())
            # kwargs[k] = v.detach()
        # return type(self)(**kwargs)
        kwargs = {k: v.detach() for k, v in self.named_parameters()}
        return self.construct(**kwargs)

    def clone(self):
        # copy = type(self)()
        # kwargs = {}
        # for k, v in self.named_parameters():
            # copy[k] = v.clone()
            # setattr(self, k, v.clone())
            # kwargs[k] = v.clone()
        # return copy
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



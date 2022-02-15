from __future__ import absolute_import, division, print_function
import torch
from torch import nn
from .utils import timing
from .depth_cloud import DepthCloud

__all__ = [
    'BaseModel',
    'Linear',
    'Polynomial',
    'ScaledPolynomial'
]


class BaseModel(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(BaseModel, self).__init__()
        self.device = device

    # @timing
    def forward(self, dc: DepthCloud) -> DepthCloud:
        dc = self.correct_depth(dc)
        return dc

    def correct_depth(self, dc: DepthCloud) -> DepthCloud:
        return dc


class Linear(BaseModel):

    def __init__(self, w0=1.0, w1=0.0, b=0.0, uniform_weights=False, device=torch.device('cpu')):
        super(Linear, self).__init__(device=device)

        assert isinstance(w0, (float, torch.Tensor))
        assert isinstance(w1, (float, torch.Tensor))
        assert isinstance(b, (float, torch.Tensor))

        if uniform_weights:
            w0 = torch.nn.init.uniform(torch.tensor(w0), -0.95, 1.05)
            w1 = torch.nn.init.uniform(torch.tensor(w1), -0.05, 0.05)
            b = torch.nn.init.uniform(torch.tensor(b), -0.05, 0.05)

        self.w0 = nn.Parameter(torch.tensor(w0, device=self.device))
        self.w1 = nn.Parameter(torch.tensor(w1, device=self.device))
        self.b = nn.Parameter(torch.tensor(b, device=self.device))

    def correct_depth(self, dc: DepthCloud) -> DepthCloud:
        assert dc.inc_angles is not None
        dc_corr = dc.copy()
        dc_corr.depth = self.w0 * dc_corr.depth + self.w1 * dc_corr.inc_angles + self.b
        return dc_corr


class Polynomial(BaseModel):

    def __init__(self, p0=0.0, p1=0.0, uniform_weights=False, device=torch.device('cpu')):
        super(Polynomial, self).__init__(device=device)

        assert isinstance(p0, (float, torch.Tensor))
        assert isinstance(p1, (float, torch.Tensor))

        if uniform_weights:
            p0 = torch.nn.init.uniform_(torch.as_tensor(p0), -0.05, 0.05)
            p1 = torch.nn.init.uniform_(torch.as_tensor(p1), -0.05, 0.05)

        self.p0 = nn.Parameter(torch.as_tensor(p0, device=self.device))
        self.p1 = nn.Parameter(torch.as_tensor(p1, device=self.device))

    def correct_depth(self, dc: DepthCloud) -> DepthCloud:
        assert dc.inc_angles is not None
        dc_corr = dc.copy()
        gamma = dc.inc_angles
        bias = self.p0 * gamma ** 2 + self.p1 * gamma ** 4
        dc_corr.depth = dc_corr.depth - bias
        return dc_corr


class ScaledPolynomial(BaseModel):

    def __init__(self, p0=0.0, p1=0.0, device=torch.device('cpu')):
        super(ScaledPolynomial, self).__init__(device=device)

        assert isinstance(p0, (float, torch.Tensor))
        assert isinstance(p1, (float, torch.Tensor))

        self.p0 = nn.Parameter(torch.tensor(p0, device=self.device))
        self.p1 = nn.Parameter(torch.tensor(p1, device=self.device))

    def correct_depth(self, dc: DepthCloud) -> DepthCloud:
        assert dc.inc_angles is not None
        dc_corr = dc.copy()
        gamma = dc.inc_angles
        bias = self.p0 * gamma ** 2 + self.p1 * gamma ** 4
        dc_corr.depth = dc_corr.depth * (1. - bias)
        return dc_corr


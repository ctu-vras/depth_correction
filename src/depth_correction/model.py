from __future__ import absolute_import, division, print_function
import torch
from torch import nn
from .utils import timing
from .depth_cloud import DepthCloud


class BaseModel(nn.Module):
    def __init__(self,):
        super(BaseModel, self).__init__()
        pass

    # @timing
    def forward(self, dc: DepthCloud) -> DepthCloud:
        dc = self.correct_depth(dc)
        return dc

    def correct_depth(self, dc: DepthCloud) -> DepthCloud:
        return dc


class Linear(BaseModel):

    def __init__(self, w0=1.0, w1=0.0, b=0.0):
        super(Linear, self).__init__()

        assert isinstance(w0, (float, torch.Tensor))
        assert isinstance(w1, (float, torch.Tensor))
        assert isinstance(b, (float, torch.Tensor))

        self.w0 = nn.Parameter(torch.tensor(w0))
        self.w1 = nn.Parameter(torch.tensor(w1))
        self.b = nn.Parameter(torch.tensor(b))

    def correct_depth(self, dc: DepthCloud) -> DepthCloud:
        assert dc.inc_angles is not None
        dc_corr = dc.copy()
        dc_corr.depth = self.w0 * dc.depth + self.w1 * dc.inc_angles + self.b
        return dc_corr


class Polynomial(BaseModel):

    def __init__(self, p0=0.0, p1=0.0):
        super(Polynomial, self).__init__()

        assert isinstance(p0, (float, torch.Tensor))
        assert isinstance(p1, (float, torch.Tensor))

        self.p0 = nn.Parameter(torch.tensor(p0))
        self.p1 = nn.Parameter(torch.tensor(p1))

    def correct_depth(self, dc: DepthCloud) -> DepthCloud:
        assert dc.inc_angles is not None
        dc_corr = dc.copy()
        gamma = dc.inc_angles
        bias = self.p0 * gamma ** 2 + self.p1 * gamma ** 4
        dc_corr.depth = dc.depth - bias
        return dc_corr

from __future__ import absolute_import, division, print_function
import torch
from torch import nn
from .depth_cloud import DepthCloud


class Linear(nn.Module):
    def __init__(self, ):
        super(Linear, self).__init__()
        self.w0 = nn.Parameter(torch.rand(1, ))
        self.w1 = nn.Parameter(torch.rand(1, ))
        self.b = nn.Parameter(torch.rand(1, ))

    def forward(self, dc: DepthCloud) -> DepthCloud:
        assert dc.depth.dim() == 2
        assert dc.depth.shape[1] == 1  # depth.shape == (N, 1)
        assert dc.dirs.dim() == 2
        assert dc.dirs.shape[1] == 3  # depth.shape == (N, 3)
        assert dc.inc_angles is not None
        depth_corr = self.w0 * dc.depth + self.w1 * dc.inc_angles + self.b
        dirs = dc.dirs
        vps = dc.vps
        dc_corr = DepthCloud(vps=vps, depth=depth_corr, dirs=dirs)
        return dc_corr


class Polynomial(nn.Module):
    def __init__(self, ):
        super(Polynomial, self).__init__()
        self.p0 = nn.Parameter(torch.rand(1, ))
        self.p1 = nn.Parameter(torch.rand(1, ))

    def forward(self, dc: DepthCloud) -> DepthCloud:
        assert dc.depth.dim() == 2
        assert dc.depth.shape[1] == 1  # depth.shape == (N, 1)
        assert dc.dirs.dim() == 2
        assert dc.dirs.shape[1] == 3  # depth.shape == (N, 3)
        assert dc.inc_angles is not None
        gamma = dc.inc_angles
        bias = self.p0 * gamma ** 2 + self.p1 * gamma ** 4
        depth_corr = dc.depth - bias
        dirs = dc.dirs
        vps = dc.vps
        dc_corr = DepthCloud(vps=vps, depth=depth_corr, dirs=dirs)
        return dc_corr

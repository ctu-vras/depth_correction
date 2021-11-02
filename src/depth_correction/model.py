from __future__ import absolute_import, division, print_function
import torch
from torch import nn
from .depth_cloud import DepthCloud


class Linear(nn.Module):

    def __init__(self,):
        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.rand(1, 2))
        self.b = nn.Parameter(torch.rand(1,))

    def forward(self, dc: DepthCloud):
        assert dc.depth.dim() == 2
        assert dc.depth.shape[1] == 1  # depth.shape == (N, 1)
        assert dc.dirs.dim() == 2
        assert dc.dirs.shape[1] == 3  # depth.shape == (N, 3)
        # TODO: depth_corr = w0 * depths + w1 * incidence_angles + b
        dc_corr = dc
        return dc_corr

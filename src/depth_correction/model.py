from __future__ import absolute_import, division, print_function
import torch
from torch import nn
from .depth_cloud import DepthCloud, depth_cloud_from_points


class Linear(nn.Module):

    def __init__(self,):
        super(Linear, self).__init__()
        self.w0 = nn.Parameter(torch.rand(1,))
        self.w1 = nn.Parameter(torch.rand(1,))
        self.b = nn.Parameter(torch.rand(1,))

    def forward(self, dc: DepthCloud):
        assert dc.depth.dim() == 2
        assert dc.depth.shape[1] == 1  # depth.shape == (N, 1)
        assert dc.dirs.dim() == 2
        assert dc.dirs.shape[1] == 3  # depth.shape == (N, 3)
        assert dc.inc_angles is not None
        # depth_corr = w0 * depths + w1 * incidence_angles + b
        pts_corr = self.w0 * dc.depth + self.w1 * dc.inc_angles + self.b
        dc_corr = depth_cloud_from_points(pts_corr)
        return dc_corr

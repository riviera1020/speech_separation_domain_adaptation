# Created on 2018/12
# Author: Kaituo XU

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.trans_norm import TransNorm1d
from src.sep_utils import overlap_and_add
from src.conv_tasnet import ConvTasNet

EPS = 1e-8

class NormLayer(nn.Module):
    def __init__(self, num_features, eps = 1e-5):
        super(NormLayer, self).__init__()
        self.register_buffer('mean', torch.zeros(num_features))
        self.register_buffer('var', torch.ones(num_features))
        self.eps = eps

    def set_mv(self, m, v):
        self.mean.copy_(m)
        self.var.copy_(v)

    def forward(self, spec):
        """
        Args:
            spec: [ M, N, T ]
        Return:
            normed spec: [M, N, T]
        """
        #input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))

        m = self.mean.unsqueeze(0).unsqueeze(-1)
        v = self.var.unsqueeze(0).unsqueeze(-1)
        spec = (spec - m) / (torch.sqrt(v + self.eps))
        return spec

class CMVNConvTasNet(ConvTasNet):
    def __init__(self, config):
        super(CMVNConvTasNet, self).__init__(config)
        self.norm_layer = NormLayer(self.N)

    def set_mv(self, m, v):
        self.norm_layer.set_mv(m, v)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
        mixture_w = self.encoder(mixture)
        sep_input = self.norm_layer(mixture_w)
        est_mask = self.separator(sep_input)
        est_source = self.decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        return est_source


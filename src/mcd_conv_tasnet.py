
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from src.utils import DEV
from src.conv_tasnet import ConvTasNet, TemporalConvNet

class Separator(nn.Module):
    def __init__(self, N, B, H, P, X, R, C, norm_type="gLN", causal=False,
                 mask_nonlinear='relu', dropout = 0.0):
        super(Separator, self).__init__()
        # Hyper-parameter
        self.C = C
        self.mask_nonlinear = mask_nonlinear
        # Components
        # [M, N, K] -> [M, N, K]
        layer_norm = ChannelwiseLayerNorm(N)
        # [M, N, K] -> [M, B, K]
        bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)
        # [M, B, K] -> [M, B, K]
        repeats = []
        for r in range(R):
            blocks = []
            for x in range(X):
                dilation = 2**x
                padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
                blocks += [TemporalBlock(B, H, P, stride=1,
                                         padding=padding,
                                         dilation=dilation,
                                         norm_type=norm_type,
                                         causal=causal,
                                         dropout=dropout)]
            repeats += [nn.Sequential(*blocks)]
        temporal_conv_net = nn.Sequential(*repeats)
        # Put together
        self.network = nn.Sequential(layer_norm,
                                     bottleneck_conv1x1,
                                     temporal_conv_net)
        # [M, B, K] -> [M, C*N, K]
        self.F1 = nn.Conv1d(B, C*N, 1, bias=False)
        self.F2 = nn.Conv1d(B, C*N, 1, bias=False)

    def apply_act(self, score):
        score = score.view(M, self.C, N, K) # [M, C*N, K] -> [M, C, N, K]
        if self.mask_nonlinear == 'softmax':
            est_mask = F.softmax(score, dim=1)
        elif self.mask_nonlinear == 'relu':
            est_mask = F.relu(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask

    def forward(self, mixture_w):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [M, N, K], M is batch size
        returns:
            est_mask: [M, C, N, K]
        """
        M, N, K = mixture_w.size()
        score = self.network(mixture_w)  # [M, N, K] -> [M, N, K]

        f1 = self.F1(score)
        f2 = self.F2(score)

        m1 = self.apply_act(f1)
        m2 = self.apply_act(f2)
        return m1, m2

class MCDConvTasNet(ConvTasNet):
    def __init__(self, config):
        super(MCDConvTasNet, self).__init__(config)

        del self.separator
        self.separator = Separator(self.N, self.B, self.H, self.P, self.X, self.R, self.C,
                self.norm_type, self.causal, self.mask_nonlinear, self.dropout)
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def decode(self, mixture_w, est_mask):
        est_source = self.decoder(mixture_w, est_mask)
        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        return est_source

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
        mixture_w = self.encoder(mixture)
        m1, m2 = self.separator(mixture_w)

        est_s1 = self.decode(mixture_w, m1)
        est_s2 = self.decode(mixture_w, m2)

        return est_s1, est_s2, m1, m2

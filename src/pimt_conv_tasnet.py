
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from src.misc import apply_norm
from src.utils import DEV
from src.conv_tasnet import ConvTasNet, TemporalConvNet
from src.specaugm import SpecAugm

class AddNoise(nn.Module):
    def __init__(self, config):
        super(AddNoise, self).__init__()
        self.scale = config['scale']

    def forward(self, x):
        std = x.std()
        mean = x.mean()

        noise = torch.normal(mean = mean, std = std, size = x.size()).to(DEV)
        return x + noise

class SpecTransform(nn.Module):
    def __init__(self, config):
        super(SpecTransform, self).__init__()

        methods = config['methods']

        self.trans = nn.ModuleList()
        for m in methods:
            if m == 'specaugm':
                self.trans.append(SpecAugm(config['specaugm']))
            if m == 'noise':
                self.trans.append(AddNoise(config['noise']))

    def forward(self, x):
        for m in self.trans:
            x = m(x)
        return x

class Separator(TemporalConvNet):
    def __init__(self, N, B, H, P, X, R, C, norm_type="gLN", causal=False,
                 mask_nonlinear='relu'):
        super(Separator, self).__init__(N, B, H, P, X, R, C,
                norm_type, causal, mask_nonlinear)

    def forward(self, mixture_w):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [M, N, K], M is batch size
        returns:
            est_mask: [M, C, N, K]
        """
        M, N, K = mixture_w.size()
        score = self.network(mixture_w)  # [M, N, K] -> [M, C*N, K]
        score = score.view(M, self.C, N, K) # [M, C*N, K] -> [M, C, N, K]
        if self.mask_nonlinear == 'softmax':
            est_mask = F.softmax(score, dim=1)
        elif self.mask_nonlinear == 'relu':
            est_mask = F.relu(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask, score

class PiMtConvTasNet(ConvTasNet):
    def __init__(self, config):
        super(PiMtConvTasNet, self).__init__(config)

        del self.separator
        self.separator = Separator(self.N, self.B, self.H, self.P, self.X, self.R, self.C,
                self.norm_type, self.causal, self.mask_nonlinear)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
        mixture_w = self.encoder(mixture)

        est_mask, score = self.separator(mixture_w)
        est_source = self.decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))

        return est_source

    def consistency_forward(self, mixture, transform):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
        mixture_w = self.encoder(mixture)

        mixture_w_purb = mixture_w.clone().detach()
        mixture_w_purb = transform(mixture_w_purb)

        est_mask1, score_clean = self.separator(mixture_w)
        est_mask2, score_noise = self.separator(mixture_w_purb)

        est_source_clean = self.decoder(mixture_w, est_mask1)
        T_origin = mixture.size(-1)
        T_conv = est_source_clean.size(-1)
        est_source_clean = F.pad(est_source_clean, (0, T_origin - T_conv))

        est_source_noise = self.decoder(mixture_w, est_mask2)
        T_origin = mixture.size(-1)
        T_conv = est_source_noise.size(-1)
        est_source_noise = F.pad(est_source_noise, (0, T_origin - T_conv))

        return est_source_clean, est_source_noise, score_clean, score_noise


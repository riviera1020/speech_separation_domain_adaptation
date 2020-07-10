# Created on 2018/12
# Author: Kaituo XU

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sep_utils import overlap_and_add
from src.conv_tasnet import Encoder, Decoder, ChannelwiseLayerNorm, TemporalBlock

EPS = 1e-8

class MyConvTasNet(nn.Module):
    def __init__(self, config, mode = 'ori'):
        """
        Args:
            N: Number of filters in autoencoder
            L: Length of the filters (in samples)
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
            mode: ori, da, pimt
        """
        super(MyConvTasNet, self).__init__()
        # Hyper-parameter

        # use to do alter forward in training baseline, dagan, pimt
        self.mode = mode

        self.N = config['N']
        self.L = config['L']
        self.B = config['B']
        self.H = config['H']
        self.P = config['P']
        self.X = config['X']
        self.R = config['R']
        self.preC = config['preC']
        self.C = config['C']

        self.norm_type = config['norm_type']
        self.causal = config['causal']
        self.mask_nonlinear = config['mask_nonlinear']
        self.dropout = config.get('dropout', 0.0)
        self.enc_dropout = config.get('enc_dropout', 0.0)
        self.sep_in_dropout = config.get('sep_in_dropout', 0.0)
        self.sep_out_dropout = config.get('sep_out_dropout', 0.0)

        print(f'Dropout: {self.dropout}')
        print(f'Enc Dropout: {self.enc_dropout}')
        print(f'Sep Input Dropout: {self.sep_in_dropout}')
        print(f'Sep Output Dropout: {self.sep_out_dropout}')

        # Components
        self.encoder = Encoder(self.L, self.N, dropout = self.enc_dropout)
        self.separator = TemporalConvNet(self.N, self.B, self.H, self.P, self.X, self.R, self.C, self.preC,
                self.norm_type, self.causal, self.mask_nonlinear, self.dropout, self.sep_in_dropout, self.sep_out_dropout)
        self.decoder = Decoder(self.N, self.L)
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture):
        if self.mode == 'pimt':
            return
        elif self.mode == 'dagan':
            return self.dagan_forward(mixture)
        else:
            return self.naive_forward(mixture)

    def naive_forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
        mixture_w = self.encoder(mixture)
        est_mask = self.separator(mixture_w)
        est_source = self.decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        return est_source

    def dict_forward(self, mixture, consider_mask = False):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
        mixture_w = self.encoder(mixture)
        est_mask, feature = self.separator.dict_forward(mixture_w)

        feature['enc'] = mixture_w
        if consider_mask:
            feature['mask'] = est_mask

        est_source = self.decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))

        return est_source, feature

class TemporalConvNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, C, preC, norm_type="gLN", causal=False,
            mask_nonlinear='relu', dropout = 0.0,
            sep_in_dropout = 0.0, sep_out_dropout = 0.0):
        """
        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super(TemporalConvNet, self).__init__()
        # Hyper-parameter
        self.C = C
        self.X = X
        self.R = R
        self.preC = preC
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
        # [M, B, K] -> [M, C*N, K]
        mask_conv1x1 = nn.Conv1d(B, preC*N, 1, bias=False)
        self.pred = nn.Linear(preC, C, bias=False)
        # Put together
        self.network = nn.Sequential(layer_norm,
                                     bottleneck_conv1x1,
                                     temporal_conv_net,
                                     mask_conv1x1)

        self.sep_in_d = sep_in_dropout
        if self.sep_in_d > 0:
            self.sep_in_dropout = nn.Dropout(self.sep_in_d)

        self.sep_out_d = sep_out_dropout
        if self.sep_out_d > 0:
            self.sep_out_dropout = nn.Dropout(self.sep_out_d)

    def forward(self, mixture_w):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [M, N, K], M is batch size
        returns:
            est_mask: [M, C, N, K]
        """
        if self.sep_out_d > 0:
            return self.d_forward(mixture_w)

        M, N, K = mixture_w.size()

        if self.sep_in_d > 0:
            mixture_w = self.sep_in_dropout(mixture_w)

        score = self.network(mixture_w)  # [M, N, K] -> [M, C*N, K]
        score = score.view(M, self.preC, N, K) # [M, C*N, K] -> [M, C, N, K]

        score = self.pred(score.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        if self.mask_nonlinear == 'softmax':
            est_mask = F.softmax(score, dim=1)
        elif self.mask_nonlinear == 'relu':
            est_mask = F.relu(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask

    def d_forward(self, mixture_w):
        M, N, K = mixture_w.size()

        if self.sep_in_d > 0:
            mixture_w = self.sep_in_dropout(mixture_w)

        score = mixture_w
        for i, l in enumerate(self.network):
            score = l(score)
            if i == 2:
                score = self.sep_out_dropout(score)

        score = score.view(M, self.C, N, K) # [M, C*N, K] -> [M, C, N, K]
        if self.mask_nonlinear == 'softmax':
            est_mask = F.softmax(score, dim=1)
        elif self.mask_nonlinear == 'relu':
            est_mask = F.relu(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask

    def dict_forward(self, mixture_w):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [M, N, K], M is batch size
        returns:
            est_mask: [M, C, N, K]
        """
        M, N, K = mixture_w.size()

        feature = {}
        score = mixture_w
        for i, layer in enumerate(self.network):
            if i == len(self.network) - 2:
                for r, repeat in enumerate(layer):
                    for x, block in enumerate(repeat):
                        score = block(score)
                        idx = r * self.X + x
                        feature[idx] = score
            else:
                score = layer(score)

        score = score.view(M, self.preC, N, K) # [M, C*N, K] -> [M, C, N, K]
        feature['emb'] = score

        score = self.pred(score.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        if self.mask_nonlinear == 'softmax':
            est_mask = F.softmax(score, dim=1)
        elif self.mask_nonlinear == 'relu':
            est_mask = F.relu(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask, feature

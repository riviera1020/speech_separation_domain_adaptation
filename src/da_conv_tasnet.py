
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from operator import itemgetter

from src.misc import apply_norm
from src.conv_tasnet import Encoder, Decoder, ChannelwiseLayerNorm
from src.conv_tasnet import DepthwiseSeparableConv, chose_norm

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DAConvTasNet(nn.Module):
    def __init__(self, config):
        """Domain adversarial conv tasnet
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
        """
        super(DAConvTasNet, self).__init__()
        # Hyper-parameter

        self.N = config['N']
        self.L = config['L']
        self.B = config['B']
        self.H = config['H']
        self.P = config['P']
        self.X = config['X']
        self.R = config['R']
        self.C = config['C']

        self.norm_type = config['norm_type']
        self.causal = config['causal']
        self.mask_nonlinear = config['mask_nonlinear']
        self.locs = config.get('locs', [(self.R-1, self.X-1)])
        self.feature_dim = len(self.locs) * self.B

        # Components
        self.encoder = Encoder(self.L, self.N)
        self.separator = TemporalConvNet(self.N, self.B, self.H, self.P, self.X, self.R, self.C,
                self.norm_type, self.causal, self.mask_nonlinear, locs = self.locs)
        self.decoder = Decoder(self.N, self.L)

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
        mixture_w = self.encoder(mixture)
        est_mask, feature = self.separator(mixture_w)
        est_source = self.decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))

        return est_source, feature

class TemporalConvNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, C, norm_type="gLN", causal=False,
                 mask_nonlinear='relu', locs = None):
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
                                         causal=causal)]
            repeats += [nn.Sequential(*blocks)]
        temporal_conv_net = nn.Sequential(*repeats)
        # [M, B, K] -> [M, C*N, K]
        mask_conv1x1 = nn.Conv1d(B, C*N, 1, bias=False)
        # Put together
        self.network = nn.Sequential(layer_norm,
                                     bottleneck_conv1x1,
                                     temporal_conv_net,
                                     mask_conv1x1)

        self.locs = []
        for loc in locs:
            r, x = loc
            assert x < X and r < R
            idx = r * X + x
            self.locs.append(idx)
        self.locs.sort()

    def forward(self, mixture_w):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [M, N, K], M is batch size
        returns:
            est_mask: [M, C, N, K]
        """
        M, N, K = mixture_w.size()

        feature = []
        score = mixture_w
        for i, layer in enumerate(self.network):
            if i == len(self.network) - 2:
                for x, repeat in enumerate(layer):
                    for r, block in enumerate(repeat):
                        score, feat = block(score)
                        feature.append(feat)
            else:
                score = layer(score)

        feature = itemgetter(*self.locs)(feature)
        if len(self.locs) > 1:
            feature = torch.cat(feature, dim = 1)

        score = score.view(M, self.C, N, K) # [M, C*N, K] -> [M, C, N, K]
        if self.mask_nonlinear == 'softmax':
            est_mask = F.softmax(score, dim=1)
        elif self.mask_nonlinear == 'relu':
            est_mask = F.relu(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask, feature

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, norm_type="gLN", causal=False, dropout=0.0, loc = -1, feat_loc = 'residual'):
        super(TemporalBlock, self).__init__()
        # [M, B, K] -> [M, H, K]
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, out_channels)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size,
                                        stride, padding, dilation, norm_type,
                                        causal)

        self.drop_loc = 3
        self.dropout = dropout
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

        if feat_loc == 'residual':
            self.feat_loc = -1
        elif feat_loc == 'conv1x1':
            self.feat_loc = 0
        elif feat_loc == 'dsconv':
            self.feat_loc = 3

    def forward(self, x):
        """
        Args:
            x: [M, B, K]
        Returns:
            [M, B, K]
        """
        feature = None
        residual = x
        for i, layer in enumerate(self.net):
            if i == self.drop_loc and self.dropout > 0:
                x = F.dropout(x, p=self.dropout)
            x = layer(x)

            if i == self.feat_loc:
                feature = x

        ret = x + residual
        if self.feat_loc == -1:
            feature = ret
        return ret, feature

class AvgLayer(nn.Module):
    def __init__(self):
        super(AvgLayer, self).__init__()

    def forward(self, x):
        """
        x : B, C, T
        """
        x = x.mean(dim = -1)
        return x

class DomainClassifier(nn.Module):
    def __init__(self, B, config):
        super(DomainClassifier, self).__init__()

        self.B = B
        mtype = config['type']
        self.mtype = mtype

        acts = {
            "hardtanh": nn.Hardtanh,
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
        }
        act = acts[config['act']]
        norm_type = config['norm_type']

        layers = config['layers']

        if mtype == 'conv':
            in_channel = B
            network = []

            for l, lconf in enumerate(layers):
                out_channel = lconf['filters']
                kernel = lconf['kernel']
                stride = lconf['stride']
                padding = 0

                layer = nn.Conv1d(in_channel, out_channel,
                        kernel_size = kernel,
                        stride = stride,
                        padding = padding)

                layer = apply_norm(layer, norm_type)
                network.append(layer)
                if l != len(layers) - 1:
                    network.append(act())
                in_channel = out_channel

            network.append(AvgLayer())
            final = nn.Linear(in_channel, 2)
            network.append(final)
            self.network = nn.Sequential(*network)

        elif mtype == 'conv-patch':
            in_channel = B
            network = []

            for l, lconf in enumerate(layers):
                out_channel = lconf['filters']
                kernel = lconf['kernel']
                stride = lconf['stride']
                padding = 0

                layer = nn.Conv1d(in_channel, out_channel,
                        kernel_size = kernel,
                        stride = stride,
                        padding = padding)

                layer = apply_norm(layer, norm_type)
                network.append(layer)
                if l != len(layers) - 1:
                    network.append(act())
                in_channel = out_channel

            self.network = nn.Sequential(*network)

        elif mtype == 'linear':
            raise NotImplementedError

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, feature, alpha):

        feature = ReverseLayerF.apply(feature, alpha)
        x = self.network(feature)
        return x

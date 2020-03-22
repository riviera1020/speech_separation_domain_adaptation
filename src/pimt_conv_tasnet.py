
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from src.misc import apply_norm
from src.utils import DEV
from src.conv_tasnet import ConvTasNet, TemporalConvNet
from src.specaugm import SpecAugm
from src.pit_criterion import cal_loss
from src.pimt_utils import PITMSELoss

class AddNoise(nn.Module):
    def __init__(self, config):
        super(AddNoise, self).__init__()
        self.scale = config['scale']

        # if no batchwise, compute mean, std on diff sample separatly
        self.batchwise = config.get('batchwise', True)

    def forward(self, x):
        if not self.batchwise:
            if len(x.size()) == 3:
                B, M, N = x.size()
                x_flat = x.view(B, -1)
                std = x_flat.std(dim = -1, keepdim = True)
                std = std.unsqueeze(-1).expand(B, M, N)
                mean = x_flat.mean(dim = -1, keepdim = True)
                mean = std.unsqueeze(-1).expand(B, M, N)
            elif len(x.size()) == 2:
                B, T = x.size()
                std = x.std(dim = -1, keepdim = True).expand(B, T)
                mean = x.mean(dim = -1, keepdim = True).expand(B, T)
            std = self.scale * std
            noise = torch.normal(mean = mean, std = std).to(DEV)
        else:
            std = x.std().item()
            std = self.scale * std
            mean = x.mean().item()
            noise = torch.normal(mean = mean, std = std, size = x.size()).to(DEV)
        return x + noise

class InputTransform(nn.Module):
    def __init__(self, config):
        super(InputTransform, self).__init__()
        methods = config['methods']

        # where = spec or wav
        self.where = config.get('where', 'spec')
        self.trans = nn.ModuleList()
        for m in methods:
            if m == 'specaugm':
                if self.where == 'wav':
                    print('No specaugm on waveform')
                    exit()
                self.trans.append(SpecAugm(**config['specaugm']))
            if m == 'noise':
                self.trans.append(AddNoise(config['noise']))

    def forward(self, x):
        for m in self.trans:
            x = m(x)
        return x

class Separator(TemporalConvNet):
    def __init__(self, N, B, H, P, X, R, C, norm_type="gLN", causal=False,
                 mask_nonlinear='relu', dropout = 0.0):
        super(Separator, self).__init__(N, B, H, P, X, R, C,
                norm_type, causal, mask_nonlinear, dropout)

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
                self.norm_type, self.causal, self.mask_nonlinear, self.dropout)
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def get_layer(self, loc):
        """
        loc:
            a       -> idx = a in separator.network
            a|b     -> idx = b in TemporalConvNet R
            a|b|c   -> idx = c in TemporalConvNet X
            a|b|c|d -> a, b, c as following and idx = c in TemporalBlock

            a should always be 2 in a|*... setting

        Range of loc:
            a: 0 to 3 (layer_norm, bottleneck_conv1x1, temporal_conv_net, mask_conv1x1)
            b: 0 to R - 1 ( layers in TemporalConvNet R )
            c: 0 to X - 1 ( layers in TemporalConvNet X )
            d: 0 to 3 (conv1x1, prelu, norm, dsconv)

            a = -1 equal to 'mask'
            a|b == a|b|-1

        I am so idiot, it seems that loc would be 2 XD
        """
        ll = [ int(l) for l in loc.split('|') ]
        assert len(ll) in [ 1, 2, 3, 4 ]

        if len(ll) == 1:
            a = ll[0]
            layer = self.separator.network[a]
        if len(ll) == 2:
            a, b = ll
            layer = self.separator.network[2][b]
        if len(ll) == 3:
            a, b, c = ll
            layer = self.separator.network[2][b][c]
        if len(ll) == 4:
            a, b, c, d = ll
            layer = self.separator.network[2][b][c][d]
        return layer

    def clean_hook_tensor(self, feat):
        locs = list(feat.keys())
        for loc in locs:
            del feat[loc]

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

    def noise_forward(self, mixture, transform):

        if transform.where == 'wav':
            mixture = transform(mixture)
            mixture_w_purb = self.encoder(mixture)
            mixture_w = mixture_w_purb
        elif transform.where == 'spec':
            mixture_w = self.encoder(mixture)
            mixture_w_purb = transform(mixture_w)

        est_mask, score_noise = self.separator(mixture_w_purb)

        est_source_noise = self.decoder(mixture_w, est_mask)
        T_origin = mixture.size(-1)
        T_conv = est_source_noise.size(-1)
        est_source_noise = F.pad(est_source_noise, (0, T_origin - T_conv))
        return est_source_noise

    def fetch_forward(self, mixture, locs, transform = None):
        """
        loc: a, a|b, a|b|c, mask
        mask can fetch in forward, so no need to do register
        """
        store_mask = False
        store_score = False
        if 'mask' in locs:
            store_mask = True
        if 'score' in locs:
            store_score = True

        feat = {}
        handles = []
        def get_feat(loc):
            def fetch(module, feat_in, feat_out):
                feat[loc] = feat_out
            return fetch
        for loc in locs:
            if loc not in [ 'mask', 'score']:
                layer = self.get_layer(loc = loc)
                h = layer.register_forward_hook(hook = get_feat(loc))
                handles.append(h)

        if transform != None:
            if transform.where == 'wav':
                mixture = transform(mixture)
                mixture_w_purb = self.encoder(mixture)
                mixture_w = mixture_w_purb
            elif transform.where == 'spec':
                mixture_w = self.encoder(mixture)
                mixture_w_purb = transform(mixture_w)
            est_mask, score = self.separator(mixture_w_purb)
        else:
            mixture_w = self.encoder(mixture)
            est_mask, score = self.separator(mixture_w)

        if store_mask:
            feat['mask'] = est_mask
        if store_score:
            feat['score'] = score

        est_source = self.decoder(mixture_w, est_mask)
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))

        # remove hooks
        for h in handles:
            h.remove()

        return est_source, feat

class ConsistencyLoss(nn.Module):
    def __init__(self, loss_type):
        super(ConsistencyLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, wav_clean, wav_noise, mixture_lengths, feat_clean, feat_noise):
        if self.loss_type == 'sisnr':
            loss, max_snr, estimate_source, reorder_estimate_source = \
                    cal_loss(wav_clean, wav_noise, mixture_lengths)
        else:
            loss = 0.
            for loc in feat_clean:
                c = feat_clean[loc]
                n = feat_noise[loc]

                if loc != 'mask' or loc != '3':
                    c = feat_clean[loc]
                    n = feat_noise[loc]
                    loss += ((c - n) ** 2).mean()
                else:
                    loss += PITMSELoss(c, n)
        return loss

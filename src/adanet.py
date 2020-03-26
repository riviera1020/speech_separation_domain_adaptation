import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

from src.sep_utils import overlap_and_add
from src.conv_tasnet import Encoder, Decoder, TemporalBlock
from src.conv_tasnet import DepthwiseSeparableConv, chose_norm
from src.conv_tasnet import ChannelwiseLayerNorm, Chomp1d

EPS = 1e-8

class ADANet(nn.Module):
    def __init__(self, config):
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
            K: Number of anchors
            emb_size: embedding size of attractor/anchor
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super(ADANet, self).__init__()
        # Hyper-parameter

        self.N = config['N']
        self.L = config['L']
        self.B = config['B']
        self.H = config['H']
        self.P = config['P']
        self.X = config['X']
        self.R = config['R']
        self.C = config['C']
        self.K = config['K']
        self.emb_size = config['emb_size']

        self.norm_type = config['norm_type']
        self.causal = config['causal']
        self.mask_nonlinear = config['mask_nonlinear']
        self.dropout = config.get('dropout', 0.0)
        # Components
        self.encoder = Encoder(self.L, self.N)
        self.separator = AttractorConvNet(self.N, self.B, self.H, self.P, self.X, self.R, self.C, self.K,
                self.emb_size, self.norm_type, self.causal, self.mask_nonlinear, self.dropout)
        self.decoder = Decoder(self.N, self.L)
        # init
        init_type = config.get('init', 'xavier_normal')
        print(f'Use {init_type} init')
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if init_type == 'xavier_normal':
                    nn.init.xavier_normal_(p)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(p)
                else:
                    print('Specify init method')
                    exit()

    def forward(self, mixture):
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

class AttractorConvNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, C, K, emb_size, norm_type="gLN", causal=False,
                 mask_nonlinear='relu', dropout = 0.0):
        """
        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            K: Number of anchors
            emb_size: embedding size of attractor/anchor
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super(AttractorConvNet, self).__init__()
        # Hyper-parameter
        self.C = C
        self.emb_size = emb_size
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

        # [M, B, K] -> [M, C*emb_size, K]
        mask_conv1x1 = nn.Conv1d(B, emb_size*N, 1, bias=False)
        # Put together
        self.network = nn.Sequential(layer_norm,
                                     bottleneck_conv1x1,
                                     temporal_conv_net,
                                     mask_conv1x1)

        v_anchors = torch.FloatTensor(K, emb_size)
        self.v_anchors = nn.Parameter(v_anchors)

        # parameter for indexing, no need on gpu
        c_combs = list(itertools.combinations(range(K), C))
        self.register_buffer('c_combs', torch.tensor(c_combs).long())
        self.register_buffer('diag_mask', torch.eye(self.C).bool())

    def cluster(self, embedding):
        """
        Keep shape logic same with ConvTasNet
        Args:
            embedding: [M, emb_size, N, K]
        returns:
            score: [M, C, N, K]
        """
        v_anchors = self.v_anchors.unsqueeze(0).expand(self.c_combs.size(0), -1, -1)
        # [ P, C, emb_size ], P is the size of combination
        s_anchor_sets = torch.gather(v_anchors, dim = 1, index = self.c_combs.unsqueeze(-1).expand(-1, -1, v_anchors.size(-1)))

        # [ M, P, N, K, C ]
        s_anchor_assigment = torch.einsum('menk,pce->mpnkc', embedding, s_anchor_sets)
        s_anchor_assigment = F.softmax(s_anchor_assigment, dim = -1)

        # [ M, P, C, E ]
        s_attractor_sets = torch.einsum('mpnkc,menk->mpce', s_anchor_assigment, embedding)
        s_attractor_sets = s_attractor_sets / s_anchor_assigment.sum(dim = (2, 3)).unsqueeze(-1)

        # [ M, P, C, C ]
        sp = torch.matmul(s_attractor_sets, s_attractor_sets.permute(0, 1, 3, 2))
        sp = sp.masked_fill(self.diag_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # [ M, P ]
        M, P, C, C = sp.size()
        s_in_set_similarities, _ = sp.view(M, P, -1).max(dim = -1)

        # [ M, 1 ] -> [ M, 1, C, E ]
        s_subset_choice = s_in_set_similarities.argmin(dim = -1, keepdim = True)
        s_subset_choice = s_subset_choice.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.C, self.emb_size)

        # [ M, C, E ]
        s_attractors = torch.gather(s_attractor_sets, dim = 1, index = s_subset_choice).squeeze(dim = 1)

        # [ M, C, N, K ]
        s_logits = torch.einsum('menk,mce->mcnk', embedding, s_attractors)
        return s_logits

    def forward(self, mixture_w):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [M, N, K], M is batch size
        returns:
            est_mask: [M, C, N, K]
        """
        M, N, K = mixture_w.size()
        feat = self.network(mixture_w)  # [M, N, K] -> [M, N*emb_size, K]
        embedding = feat.view(M, self.emb_size, N, K) # [M, N*emb_size, K] -> [M, emb_size, N, K]
        score = self.cluster(embedding) # [M, emb_size, N, K ] -> [M, C, N, K ]
        if self.mask_nonlinear == 'softmax':
            est_mask = F.softmax(score, dim=1)
        elif self.mask_nonlinear == 'relu':
            est_mask = F.relu(score)
        elif self.mask_nonlinear == 'sigmoid':
            est_mask = F.sigmoid(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask


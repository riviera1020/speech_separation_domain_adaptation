import torch
import torch.nn as nn
from src.misc import apply_norm
from src.conv_tasnet import TemporalConvNet

class AvgLayer(nn.Module):
    def __init__(self):
        super(AvgLayer, self).__init__()

    def forward(self, x):
        """
        x : B, C, T
        """
        x = x.mean(dim = -1)
        return x

class LSTMClassifer(nn.Module):
    def __init__(self, B, config):
        super(LSTMClassifer, self).__init__()

        self.B = B
        self.num_layers = config['layers']
        self.hidden_size = config['hidden_size']
        self.dropout = config['dropout']

        self.rnns = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drops = nn.ModuleList()

        for l in range(self.num_layers):

            if l == 0:
                in_dim = self.B
            else:
                in_dim = self.hidden_size * 2

            self.rnns.append(
                    nn.LSTM(in_dim, self.hidden_size, 1, batch_first = True, bidirectional = True))
            self.drops.append(nn.Dropout(self.dropout))
            self.norms.append(nn.LayerNorm(self.hidden_size*2))

        self.out = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        for l in range(self.num_layers):
            x, _ = self.rnns[l](x)
            x = self.norms[l](x)
            x = self.drops[l](x)

        x = x[:, -1, :]
        x = self.out(x)
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
            final = nn.Linear(in_channel, 1)
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

        elif mtype == 'LSTM':
            in_channel = B
            self.network = LSTMClassifer(B, config)

        elif mtype == 'linear':
            raise NotImplementedError

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, feature):

        x = self.network(feature)
        return x

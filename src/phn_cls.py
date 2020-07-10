
import torch
import torch.nn as nn

from torchaudio.transforms import MFCC
from src.specaugm import SpecAugm
from src.mish import Mish

def get_model(config, vocab_size):
    if config['type'] == 'linear':
        return DNN(config, vocab_size)
    elif config['type'] == 'rnn':
        return RNN(config, vocab_size)
    elif config['type'] == 'cnn':
        return CNN(config, vocab_size)

class FrameSlide(nn.Module):

    def __init__(self, w):
        super(FrameSlide, self).__init__()
        assert w % 2 == 1
        self.i = (w - 1) // 2

    def pad(self, p, x, i, pre):
        x = x.unsqueeze(1)
        x = x.expand(x.size(0), i, x.size(2))

        if pre:
            p = torch.cat([x, p], dim = 1)
        else:
            p = torch.cat([p, x], dim = 1)
        return p

    def forward(self, x):

        pres = []
        posts = []
        for i in range(1, self.i + 1):
            pre = x[:, :-i:, :]
            post = x[:, i:, :]

            pre = self.pad(pre, x[:, 0, :], i, pre = True)
            post = self.pad(post, x[:, -1, :], i, pre = False)

            pres.insert(0, pre)
            posts.append(post)

        x = torch.cat([*pres, x, *posts], dim = 2)
        return x

class FeatExt(nn.Module):

    def __init__(self, sample_rate, n_mfcc, mel_args, w):
        super(FeatExt, self).__init__()

        self.mfcc = MFCC(sample_rate = sample_rate,
                n_mfcc = n_mfcc,
                melkwargs = mel_args)
        self.frame_slide = FrameSlide(w)
        self.spec_augm = SpecAugm(F = -1,
                T = 20,
                tm_num = 1)

    def norm(self, x):
        mean = x.mean(dim = 1, keepdim = True)
        var = x.var(dim = 1, keepdim = True)

        x = (x - mean) / (var + 1e-5).sqrt()
        return x

    def forward(self, x):
        x = self.mfcc(x).permute(0, 2, 1)
        x = self.norm(x)
        x = self.frame_slide(x)
        x = self.spec_augm(x)
        return x

class DNN(nn.Module):

    def __init__(self, config, vocab_size):

        super(DNN, self).__init__()

        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']

        self.net = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, vocab_size)
            )

    def forward(self, x):

        return self.net(x)

class RNN(nn.Module):

    def __init__(self, config, vocab_size):

        super(RNN, self).__init__()

        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']

        self.net = nn.LSTM(
                input_size = self.input_size,
                hidden_size = self.hidden_size,
                num_layers = 2,
                batch_first = True,
                bidirectional = True)
        self.out = nn.Linear(self.hidden_size * 2, vocab_size)

    def forward(self, x):

        x, _ = self.net(x)
        x = self.out(x)
        return x

class CNN(nn.Module):

    def __init__(self, config, vocab_size):
        super(CNN, self).__init__()

        self.input_size = config['input_size']
        archs = config['archs']

        self.layers = len(archs)
        self.cnns = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_channels = self.input_size
        for i, conf in enumerate(archs):
            out_channels = conf['channels']
            kernel_size = conf['kernel_size']
            stride = 1
            dilation = conf['dilation']
            padding = self.get_padding(kernel_size, dilation)
            net = nn.Conv1d(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = padding,
                    dilation = dilation)
            self.cnns.append(net)
            self.acts.append(nn.LeakyReLU())
            self.norms.append(nn.InstanceNorm1d(out_channels, eps=1e-8))

            in_channels = out_channels

        self.out = nn.Linear(in_channels, vocab_size)

    def get_padding(self, k, d):
        #return (k - 1) // 2
        return ( d * (k - 1) ) // 2

    def forward(self, x):

        x = x.permute(0, 2, 1)
        for i in range(self.layers):
            x = self.cnns[i](x)
            x = self.acts[i](x)
            x = self.norms[i](x)
        x = x.permute(0, 2, 1)
        x = self.out(x)
        return x

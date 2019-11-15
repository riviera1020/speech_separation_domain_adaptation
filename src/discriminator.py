"""
Borrow code from https://github.com/yanggeng1995/GAN-TTS/blob/master/models/modules.py
"""
import random

import torch
import torch.nn as nn
from src.misc import apply_norm

class DBlock(nn.Module):
    def __init__(self, in_channel, out_channel, factor, norm_type):
        """
        Args:
            in_channel: input channel
            out_channel: output channel
            factor: downsample factor
        """
        super(DBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.AvgPool1d(factor, stride=factor),
            nn.ReLU(),
            apply_norm(nn.Conv1d(in_channel, out_channel,
                                    kernel_size=3, padding=1), norm_type),
            nn.ReLU(),
            apply_norm(nn.Conv1d(out_channel, out_channel, kernel_size=3,
                                    dilation=2, padding=2 * (3 - 1) // 2), norm_type)
        )
        self.residual = nn.Sequential(
            apply_norm(nn.Conv1d(in_channel, out_channel, kernel_size=1), norm_type),
            nn.AvgPool1d(factor, stride=factor)
        )

    def forward(self, inputs):
        outputs = self.layers(inputs) + self.residual(inputs)
        return outputs

class UnConditionalDBlocks(nn.Module):
    def __init__(self,
                 in_channel,
                 factors=(5, 3),
                 out_channel=(128, 256),
                 norm_type = None):
        super(UnConditionalDBlocks, self).__init__()

        self.in_channel = in_channel
        self.factors = factors
        self.out_channel = out_channel

        self.layers = nn.ModuleList()
        self.layers.append(DBlock(in_channel, 64, 1, norm_type))
        in_channel = 64
        for (i, factor) in enumerate(factors):
            self.layers.append(DBlock(in_channel, out_channel[i], factor, norm_type))
            in_channel = out_channel[i]
        self.layers.append(DBlock(in_channel, in_channel, 1, norm_type))
        self.layers.append(DBlock(in_channel, in_channel, 1, norm_type))
        #self.layers.append(DBlock(in_channel, 1, 1, norm_type))
        self.out = nn.Linear(in_channel, 1)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        outputs = inputs.view(batch_size, self.in_channel, -1)
        for layer in self.layers:
            outputs = layer(outputs)

        outputs = outputs.mean(dim = -1)
        outputs = self.out(outputs)
        return outputs

class RWD(nn.Module):
    def __init__(self, config):
        """
        Args:
            w: window_size (default: 240)
            ks: downsample_factor (default: (1, 2, 4, 8, 15))
            factorss: factors (default: ((5, 3), (5, 3), (5, 3), (5, 3), (2, 2)))
        """
        super(RWD, self).__init__()

        self.w = config['w']
        self.ks = config['ks']
        factorss = config['factorss']
        norm_type = config['norm_type']

        self.ensembles = nn.ModuleList()
        for k, factors in zip(self.ks, factorss):
            uDis = UnConditionalDBlocks(in_channel = k, factors = factors,
                    out_channel = (128, 256), norm_type = norm_type)
            self.ensembles.append(uDis)

    def forward(self, inputs):
        B, T = inputs.size()

        outputs = []
        for (k, uDis) in zip(self.ks, self.ensembles):
            w_len = k * self.w
            index = random.randint(0, T - w_len - 1)

            rand_inputs = inputs[:, index: index + w_len]
            output = uDis(rand_inputs)
            outputs.append(output)
        return outputs

if __name__ == '__main__':

    window_length = 240
    in_channel = 2

    model = RWD()
    audio = torch.rand(4, 32000)
    outputs = model(audio)


"""
Only implement freq & time mask
"""
import torch
import torch.nn as nn
import random

def freq_mask(spec, F=27, num_masks=1, replace_with_zero=False):
    num_mel_channels = spec.size(2)

    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f):
            return spec

        mask_end = random.randrange(f_zero, f_zero + f)
        if replace_with_zero:
            spec[:, f_zero:mask_end] = 0
        else:
            spec[:, f_zero:mask_end] = spec.mean()

    return spec

def time_mask(spec, T=100, num_masks=1, p=1.0, replace_with_zero=False):
    len_spectro = spec.size(1)
    T = min([T, int(p * T)])

    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t):
            return spec

        mask_end = random.randrange(t_zero, t_zero + t)
        if (replace_with_zero):
            spec[:, t_zero:mask_end] = 0
        else:
            spec[:, t_zero:mask_end] = spec.mean()
    return spec

class SpecAugm(nn.Module):
    def __init__(self, F=27, fm_num=1, T=100, tm_num=1, p=1.0, replace_with_zero = False):
        super(SpecAugm, self).__init__()

        self.F = F
        self.fm_num = fm_num
        self.T = T
        self.tm_num = tm_num
        self.p = p
        self.replace_with_zero = replace_with_zero

        self.apply_F = (F > 0)
        self.apply_T = (T > 0)

    def forward(self, audio):
        """
        audio: tensor
        """
        if self.apply_F:
            audio = freq_mask(audio, self.F, self.fm_num, self.replace_with_zero)

        if self.apply_T:
            audio = time_mask(audio, self.T, self.tm_num, self.p, self.replace_with_zero)

        return audio

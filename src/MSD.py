import torch
import torch.nn as nn
import torch.nn.functional as F
from src.misc import apply_norm

class Discriminator(nn.Module):
    def __init__(self, norm_type):
        super(Discriminator, self).__init__()

        self.discriminator = nn.ModuleList([
            nn.Sequential(
                apply_norm(nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7), norm_type),
                nn.LeakyReLU(),
            ),
            nn.Sequential(
                apply_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4), norm_type),
                nn.LeakyReLU(),
            ),
            nn.Sequential(
                apply_norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16), norm_type),
                nn.LeakyReLU(),
            ),
            nn.Sequential(
                apply_norm(nn.Conv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64), norm_type),
                nn.LeakyReLU(),
            ),
            nn.Sequential(
                apply_norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256), norm_type),
                nn.LeakyReLU(),
            ),
            nn.Sequential(
                apply_norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2), norm_type),
                nn.LeakyReLU(),
            ),
            apply_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1), norm_type),
        ])

    def forward(self, x):
        '''
            returns: (list of 6 features, discriminator score)
            we directly predict score without last sigmoid function
            since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        '''
        for module in self.discriminator:
            x = module(x)
        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, config):
        super(MultiScaleDiscriminator, self).__init__()

        norm_type = config['norm_type']

        self.discriminators = nn.ModuleList(
            [Discriminator(norm_type) for _ in range(3)]
        )

        self.pooling = nn.ModuleList([
            Identity(),
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
            nn.AvgPool1d(kernel_size=4, stride=4, padding=2)
        ])

    def forward(self, x):
        ret = list()

        x = x.unsqueeze(1)
        for pool, disc in zip(self.pooling, self.discriminators):
            x = pool(x)
            d = disc(x)
            ret.append(d)
        return ret

if __name__ == '__main__':
    model = Discriminator()

    x = torch.randn(3, 1, 22050)
    print(x.shape)

    features, score = model(x)
    for feat in features:
        print(feat.shape)
    print(score.shape)

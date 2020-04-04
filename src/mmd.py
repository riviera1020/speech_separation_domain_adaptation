import torch
import torch.nn as nn
from functools import partial

def pairwise_distance(x, y):
    # [ B, F, 1]
    x = x.view(x.shape[0], x.shape[1], 1)
    # [ F, B ]
    y = torch.transpose(y, 0, 1)
    # [ B, B ]
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output

def gaussian_kernel_matrix(x, y, sigmas):

    # [ S, 1 ]
    sigmas = sigmas.view(sigmas.shape[0], 1)
    # [ S, 1 ]
    beta = 1. / (2. * sigmas)
    # [ B, B ]
    dist = pairwise_distance(x, y).contiguous()
    # [ 1, B*B ]
    dist_ = dist.view(1, -1)
    # [ S, B*B ]
    s = torch.matmul(beta, dist_)
    # [ B, B ]
    return torch.sum(torch.exp(-s), 0).view_as(dist)

def maximum_mean_discrepancy(x, y, kernel= gaussian_kernel_matrix):

    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    return cost

class MMDLoss(nn.Module):
    def __init__(self, sigmas):
        super(MMDLoss, self).__init__()
        sigmas = torch.FloatTensor(sigmas)
        self.register_buffer('sigmas', sigmas)

    def forward(self, source_features, target_features):
        """
        source_features: [ B, F, T ]
        target_features: [ B, F, T ]
        """
        source_features = source_features.mean(dim = -1)
        target_features = target_features.mean(dim = -1)
        gaussian_kernel = partial(gaussian_kernel_matrix, sigmas = self.sigmas)
        loss_value = maximum_mean_discrepancy(source_features, target_features, kernel= gaussian_kernel)
        loss_value = loss_value

        return loss_value

if __name__ == '__main__':

    B = 4
    F = 256

    s = torch.rand(B, F)
    t = torch.rand(B, F)

    loss = mmd_loss(s, t)


import torch
import torch.nn as nn
from itertools import permutations

class PITMSELoss(nn.Module):
    def __init__(self):
        super(PITMSELoss, self).__init__()

    def forward(self, x1, x2):
        """
        x1: B x C x N x K
        x2: B x C x N x K
        """

        B, C, _, _ = x1.size()

        x1 = x1.view(B, C, -1).unsqueeze(dim = 1)
        x2 = x2.view(B, C, -1).unsqueeze(dim = 2)

        # [B, C, C]
        mse = ((x1 - x2) ** 2).mean(dim = -1)

        perms = x1.new_tensor(list(permutations(range(C))), dtype=torch.long)
        # one-hot, [C!, C, C]
        index = torch.unsqueeze(perms, 2)
        perms_one_hot = x1.new_zeros((*perms.size(), C)).scatter_(2, index, 1)

        # [B, C!] <- [B, C, C] einsum [C!, C, C]
        mse_set = torch.einsum('bij,pij->bp', [mse, perms_one_hot])
        min_mse_idx = torch.argmin(mse_set, dim=1)  # [B]

        min_mse, _ = torch.min(mse_set, dim=1, keepdim=True)
        min_mse /= C

        #return min_mse.mean(), min_mse_idx
        return min_mse.mean()

if __name__ == '__main__':

    B = 2
    C = 2
    N = 2
    K = 2

    l = PITMSELoss()

    #x1 = torch.rand(B, C, N, K)
    #x2 = torch.rand(B, C, N, K)

    x1 = torch.ones(B, C, N, K)
    x2 = torch.ones(B, C, N, K)

    x1[0, 0, :, :] *= 2
    x2[0, 0, :, :] *= 2

    x1[1, 0, :, :] *= 2
    x2[1, 1, :, :] *= 3

    r = l(x1, x2)


import torch
import torch.autograd as autograd

from src.utils import DEV

def calc_gradient_penalty(D, real_data, fake_data):

    B, T = real_data.size()
    alpha = torch.rand(B, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(DEV)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.requires_grad = True

    disc_interpolates = D(interpolates)

    gp = 0.
    for d_out in disc_interpolates:
        gradients = autograd.grad(outputs = d_out,
                                  inputs=interpolates,
                                  grad_outputs = torch.ones(d_out.size()).to(DEV),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        gp = gp + gradient_penalty
    return gp


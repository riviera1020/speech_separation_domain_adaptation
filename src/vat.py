
import torch
import torch.nn as nn
import torch.nn.functional as F

class EntMin(nn.Module):

    def __init__(self):
        super(EntMin, self).__init__()

    def forward(self, ul_y):
        p = F.softmax(ul_y, dim = 2)
        logp = F.log_softmax(ul_y, dim = 2)
        ent = - (p * logp).sum(dim = 2).mean()
        return ent

class VAT(nn.Module):

    def __init__(self, xi, eps, num_iters):
        super(VAT, self).__init__()

        self.xi = xi
        self.eps = eps
        self.num_iters = num_iters

    def l2_normalize(self, x):
        xn = x.norm(p = 2, dim = -1, keepdim = True)
        x = x / (xn + 1e-16)
        return x

    def kl_div_with_logit(self, q_logit, p_logit):

        q = F.softmax(q_logit, dim=2)
        logq = F.log_softmax(q_logit, dim=2)
        logp = F.log_softmax(p_logit, dim=2)

        qlogq = ( q *logq).sum(dim=2).mean()
        qlogp = ( q *logp).sum(dim=2).mean()

        return qlogq - qlogp

    def l2_normalize(self, x):
        xn = x.norm(p = 2, dim = -1, keepdim = True)
        x = x / (xn + 1e-16)
        return x

    def forward(self, model, ul_data):

        # gen r_adv
        with torch.no_grad():
            ul_phn = model(ul_data)

        d = ul_data.clone().detach().normal_()
        for i in range(self.num_iters):
            d = self.xi * self.l2_normalize(d)
            d.requires_grad_()
            pert_phn = model(ul_data + d)

            delta_kl = self.kl_div_with_logit(ul_phn, pert_phn)
            delta_kl.backward()
            d = d.grad.data.clone().detach()
            model.zero_grad()

        d = self.l2_normalize(d)
        r_adv = self.eps * d

        # cal loss
        adv_phn = model(ul_data + r_adv)
        delta_kl = self.kl_div_with_logit(ul_phn, adv_phn)
        return delta_kl

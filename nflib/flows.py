"""
Implements various flows.
Each flow is invertible so it can be forward()ed and backward()ed.
Notice that backward() is not backward as in backprop but simply inversion.
Each flow also outputs its log det J "regularization"

Reference:

NICE: Non-linear Independent Components Estimation, Dinh et al. 2014
https://arxiv.org/abs/1410.8516

Variational Inference with Normalizing Flows, Rezende and Mohamed 2015
https://arxiv.org/abs/1505.05770

Density estimation using Real NVP, Dinh et al. May 2016
https://arxiv.org/abs/1605.08803
(Laurent's extension of NICE)

Improved Variational Inference with Inverse Autoregressive Flow, Kingma et al June 2016
https://arxiv.org/abs/1606.04934
(IAF)

Masked Autoregressive Flow for Density Estimation, Papamakarios et al. May 2017 
https://arxiv.org/abs/1705.07057
"The advantage of Real NVP compared to MAF and IAF is that it can both generate data and estimate densities with one forward pass only, whereas MAF would need D passes to generate data and IAF would need D passes to estimate densities."
(MAF)

Glow: Generative Flow with Invertible 1x1 Convolutions, Kingma and Dhariwal, Jul 2018
https://arxiv.org/abs/1807.03039

"Normalizing Flows for Probabilistic Modeling and Inference"
https://arxiv.org/abs/1912.02762
(review paper)
"""

#  Copyright: (C) ETAS GmbH 2019. All rights reserved.

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nflib.nets import LeafParam, MLP, ARMLP, JoinConditioner, ARMLPConditional
from common import helper

class AffineConstantFlow(nn.Module):
    """ 
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """
    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else None
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else None
        
    def forward(self, x):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


# class ActNorm(AffineConstantFlow):
#     """
#     Really an AffineConstantFlow but with a data-dependent initialization,
#     where on the very first batch we clever initialize the s,t so that the output
#     is unit gaussian. As described in Glow paper.
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.data_dep_init_done = False
#
#     def forward(self, x):
#         # first batch is used for init
#         if not self.data_dep_init_done:
#             assert self.s is not None and self.t is not None # for now
#             self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
#             self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
#             self.data_dep_init_done = True
#         return super().forward(x)


class ActNorm(nn.Module):
    """ Actnorm layer; cf Glow section 3.1 """
    def __init__(self, param_dim=(1, 2)):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(param_dim))
        self.bias = nn.Parameter(torch.zeros(param_dim))
        self.register_buffer('initialized', torch.tensor(0).byte())

    def forward(self, x):
        if not self.initialized:
            # mean and variance
            flattened_x = x.flatten(1)
            self.bias.squeeze().data.copy_(x.mean(0)).view_as(self.scale)  # .transpose(0,1).flatten(1)
            self.scale.squeeze().data.copy_(torch.log(x.std(0, False) + 1e-6)).view_as(self.bias)  # .transpose(0,1).flatten(1)
            self.initialized += 1

        z = (x - self.bias) * torch.exp(-self.scale)
        logdet = -self.scale.sum()
        return z, logdet

    def backward(self, z):
        return z * torch.exp(self.scale) + self.bias, self.scale.sum()  # .abs().log()


class ConActNorm(nn.Module):
    """ Conditional Actnorm layer """
    def __init__(self, con_dim, param_dim=2, nh=64):
        super().__init__()
        self.conditioner = nn.Sequential(
            nn.Linear(con_dim, nh),
            nn.LeakyReLU(0.2),
            # nn.Linear(nh, nh),
        )
        self.scale = nn.Linear(nh, param_dim)
        self.bias = nn.Linear(nh, param_dim)

    def forward(self, x, condition):
        cond = self.conditioner(condition)
        scale = self.scale(cond)
        bias = self.bias(cond)
        z = (x - bias) * torch.exp(-scale)
        logdet = -scale.sum(-1)
        return z, logdet

    def backward(self, z, condition):
        cond = self.conditioner(condition)
        scale = self.scale(cond)
        bias = self.bias(cond)
        return z * torch.exp(scale) + bias, scale.sum(-1)


class AffineHalfFlow(nn.Module):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the 
    dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """
    def __init__(self, dim, parity, con_dim=None, net_class=MLP, nh=24, scale=True, shift=True, device="cpu",
                 use_elu=True, use_tanh=False, use_cond_an=False):
        super().__init__()
        self.device = device
        if dim%2 != 0:
            dim = dim+1
            self.add_zeros = True
        else:
            self.add_zeros = False
        self.dim = dim
        self.parity = parity
        self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2).to(device)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2).to(device)
        if scale:
            self.s_cond = net_class(self.dim // 2, self.dim // 2, nh).to(device)
        if shift:
            self.t_cond = net_class(self.dim // 2, self.dim // 2, nh).to(device)
        if con_dim is not None:
            self.s_j_cond = JoinConditioner(self.dim, con_dim, nh=1).to(device)
            self.t_j_cond = JoinConditioner(self.dim, con_dim, nh=1).to(device)
        if con_dim is not None and use_cond_an:
            self.act_norm = ConActNorm(con_dim)
            self.use_cond_an = use_cond_an
        else:
            self.act_norm = ActNorm()
            self.use_cond_an = False
        self.elu = nn.ELU()
        self.use_elu = use_elu
        if use_tanh:
            self.tanh = nn.Tanh()
        else:
            self.tanh = None
        
    def forward(self, x, con_in=None):
        if self.add_zeros:
            zeros = torch.zeros((x.shape[0], 1)).to(self.device)
            x = torch.cat((x, zeros), dim=1)
        if self.use_cond_an:
            x, act_log_det = self.act_norm(x, con_in)
        else:
            x, act_log_det = self.act_norm(x)
        x0, x1 = x[:,::2], x[:,1::2]
        if self.parity:
            x0, x1 = x1, x0
        if con_in is not None:
            x0_s_cond = self.s_j_cond(x0, con_in)
            x0_t_cond = self.t_j_cond(x0, con_in)
        else:
            x0_s_cond = x0
            x0_t_cond = x0
        s = self.s_cond(x0_s_cond)
        t = self.t_cond(x0_t_cond)
        if self.tanh is not None:
            s = self.tanh(s) * 3
        z0 = x0 # untouched half
        if self.use_elu:
            z1 = (self.elu(s) + 1.) * x1 + t
        else:
            z1 = torch.exp(s) * x1 + t  # transform this half as a function of the other
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat((z0, z1), dim=1)
        if self.add_zeros:
            z = z[:, :-1]
        if self.use_elu:
            log_det = torch.sum(torch.log(self.elu(s) + 1.), dim=1) + act_log_det
        else:
            log_det = torch.sum(s, dim=1) + act_log_det
        return z, log_det
    
    def backward(self, z, con_in=None):
        if self.add_zeros:
            zeros = torch.zeros((z.shape[0], 1)).to(self.device)
            z = torch.cat((z, zeros), dim=1)
        z0, z1 = z[:,::2], z[:,1::2]
        if self.parity:
            z0, z1 = z1, z0
        if con_in is not None:
            z0_s_cond = self.s_j_cond(z0, con_in)
            z0_t_cond = self.t_j_cond(z0, con_in)
        else:
            z0_s_cond = z0
            z0_t_cond = z0
        s = self.s_cond(z0_s_cond)
        t = self.t_cond(z0_t_cond)
        if self.tanh is not None:
            s = self.tanh(s) * 3.
        x0 = z0  # this was the same
        if self.use_elu:
            x1 = (z1 - t) / (self.elu(s) + 1.)
        else:
            x1 = (z1 - t) * torch.exp(-s)  # reverse the transform on this half
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat((x0, x1), dim=1)
        if self.add_zeros:
            x = x[:, :-1]
        if self.use_cond_an:
            x, act_log_det = self.act_norm.backward(x, con_in)
        else:
            x, act_log_det = self.act_norm.backward(x)
        if self.use_elu:
            log_det = torch.sum(torch.log(1 / (self.elu(s) + 1.)), dim=1) + act_log_det
        else:
            log_det = torch.sum(-s, dim=1) + act_log_det
        return x, log_det


class SlowMAF(nn.Module):
    """ 
    Masked Autoregressive Flow, slow version with explicit networks per dim
    """
    def __init__(self, dim, parity, net_class=MLP, nh=24, device="cpu"):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleDict().to(device)
        self.layers[str(0)] = LeafParam(2).to(device)
        for i in range(1, dim):
            self.layers[str(i)] = net_class(i, 2, nh).to(device)
        self.order = list(range(dim)) if parity else list(range(dim))[::-1]
        
    def forward(self, x):
        z = torch.zeros_like(x)
        log_det = torch.zeros(x.size(0))
        for i in range(self.dim):
            st = self.layers[str(i)](x[:, :i])
            s, t = st[:, 0], st[:, 1]
            z[:, self.order[i]] = x[:, i] * torch.exp(s) + t
            log_det += s
        return z, log_det

    def backward(self, z):
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.size(0))
        for i in range(self.dim):
            st = self.layers[str(i)](x[:, :i])
            s, t = st[:, 0], st[:, 1]
            x[:, i] = (z[:, self.order[i]] - t) * torch.exp(-s)
            log_det += -s
        return x, log_det


class MAF(nn.Module):
    """ Masked Autoregressive Flow that uses a MADE-style network for fast forward """
    
    def __init__(self, dim, parity, cond_in, net_class=ARMLP, nh=24, device="cpu", writer=None, id=0,
                 use_tanh=True, big_hidden=True, use_elu=False, use_cond_an=False, use_act_norm=False, s_mp=5.):
        super().__init__()
        self.dim = dim
        self.device = device
        if cond_in is not None:
            self.net = ARMLPConditional(dim, dim * 2, nh, cond_in, big_hidden=big_hidden).to(device)
        else:
            self.net = net_class(dim, dim*2, nh).to(device)
        self.parity = parity
        self.writer = writer
        self.id = id
        self.internal_fw_counter = 0
        self.internal_bw_counter = 0
        self.elu = nn.ELU()
        self.use_elu = use_elu
        self.EPS = 1e-12
        if use_act_norm:
            if cond_in is not None and use_cond_an:
                self.act_norm = ConActNorm(cond_in, param_dim=dim)
                self.use_cond_an = use_cond_an
            else:
                self.act_norm = ActNorm(param_dim=(1, dim))
                self.use_cond_an = False
        else:
            self.act_norm = None
            self.use_cond_an = False
        if use_tanh:
            self.tanh = nn.Tanh()
        else:
            self.tanh = None
        self.s_mp = s_mp

    def forward(self, x, y=None):
        # here we see that we are evaluating all of z in parallel, so density estimation will be fast
        if self.act_norm is not None:
            if self.use_cond_an:
                x, act_log_det = self.act_norm(x, y)
            else:
                x, act_log_det = self.act_norm(x)
        else:
            act_log_det = 0
        # act_log_det = 0
        if y is not None:
            st = self.net(x, y)
        else:
            st = self.net(x)
        s, t = st.split(self.dim, dim=1)
        if self.tanh is not None:
            s = self.tanh(s)*self.s_mp
        if self.use_elu:
            z = x * (self.elu(s) + 1.) + t  # elu+1
        else:
            z = x * torch.exp(s) + t  # elu+1
        # reverse order, so if we stack MAFs correct things happen
        z = z.flip(dims=(1,)) if self.parity else z
        if self.use_elu:
            log_det = torch.sum(torch.log(self.elu(s) + 1.), dim=1) + act_log_det  # log(elu(s)+1)
        else:
            log_det = torch.sum(s, dim=1) + act_log_det
        if self.writer is not None:
            self.writer.add_histogram("s_fw_"+str(self.id), s, global_step=self.internal_fw_counter)
            self.writer.add_histogram("t_fw_" + str(self.id), t, global_step=self.internal_fw_counter)
            self.internal_fw_counter += 1
        return z, log_det
    
    def backward(self, z, y=None):
        # we have to decode the x one at a time, sequentially
        x = torch.zeros_like(z).to(self.device)
        log_det = torch.zeros(z.size(0)).to(self.device)
        z = z.flip(dims=(1,)) if self.parity else z
        for i in range(self.dim):
            if y is not None:
                st = self.net(x.clone(), y.clone())  # clone to avoid in-place op errors if using IAF
            else:
                st = self.net(x.clone())  # clone to avoid in-place op errors if using IAF
            s, t = st.split(self.dim, dim=1)
            if self.tanh is not None:
                s = self.tanh(s) * self.s_mp
            if torch.isnan(s).any():
                print("found NaN in s tensor! In Layer ", self.id)
                print("In Dimension ", helper.features[i])
            if torch.isnan(t).any():
                print("found NaN in t tensor! In Layer", self.id)
                print("In Dimension ", helper.features[i])
            if self.use_elu:
                x[:, i] = (z[:, i] - t[:, i]) / (self.elu(s[:, i]).to(self.device) + 1.)
            else:
                x[:, i] = (z[:, i] - t[:, i]) * torch.exp(-s[:, i]).to(self.device)
            if torch.isnan(x).any():
                print("found NaN in output tensor! In Layer", self.id)
                print("In Dimension ", helper.features[i])
            if self.use_elu:
                log_det += torch.log(1/(self.elu(s[:, i]).to(self.device) + 1.))  # -s[:, i]  # 1/log(elu(s)+1)
            else:
                log_det += -s[:, i]
            if self.writer is not None:
                self.writer.add_histogram("s_bw_" + str(self.id), s, global_step=self.internal_bw_counter)
                self.writer.add_histogram("t_bw_" + str(self.id), t, global_step=self.internal_bw_counter)
                self.internal_bw_counter += 1
        if self.act_norm is not None:
            if self.use_cond_an:
                x, act_log_det = self.act_norm.backward_sample(x, y)
            else:
                x, act_log_det = self.act_norm.backward_sample(x)
        else:
            act_log_det = 0
        log_det += act_log_det
        return x, log_det


class IAF(MAF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        reverse the flow, giving an Inverse Autoregressive Flow (IAF) instead, 
        where sampling will be fast but density estimation slow
        """
        self.forward, self.backward = self.backward, self.forward


class Invertible1x1Conv(nn.Module):
    """ 
    As introduced in Glow paper.
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
        P, L, U = torch.lu_unpack(*Q.lu())
        self.P = P # remains fixed during optimization
        self.L = nn.Parameter(L) # lower triangular portion
        self.S = nn.Parameter(U.diag()) # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1)) # "crop out" diagonal, stored in S

    def _assemble_W(self):
        """ assemble W from its pieces (P, L, U, S) """
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x):
        W = self._assemble_W()
        z = x @ W
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z, log_det

    def backward(self, z):
        W = self._assemble_W()
        W_inv = torch.inverse(W)
        x = z @ W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        return x, log_det

# ------------------------------------------------------------------------


class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows, device="cpu", prep_con=False, cond_in=49, nh=64):
        super().__init__()
        self.flows = nn.ModuleList(flows).to(device)
        if prep_con:
            self.cond_layer = MLP(nin=cond_in, nout=nh, nh=nh)
        else:
            self.cond_layer = None
        self.device = device

    def forward_sample(self, x, con_in=None):
        m, _ = x.shape
        log_det = torch.zeros(m).to(self.device)
        zs = [x]
        if con_in is not None and self.cond_layer is not None:
            con_in = self.cond_layer(con_in)
        for flow in self.flows:
            if con_in is not None:
                x, ld = flow.forward_sample(x, con_in)
            else:
                x, ld = flow.forward_sample(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward_sample(self, z, con_in=None):
        m, _ = z.shape
        log_det = torch.zeros(m).to(self.device)
        xs = [z]
        if con_in is not None and self.cond_layer is not None:
            con_in = self.cond_layer(con_in)
        for flow in self.flows[::-1]:
            if con_in is not None:
                z, ld = flow.backward_sample(z, con_in)
            else:
                z, ld = flow.backward_sample(z)
            log_det += ld
            xs.append(z)
        return xs, -log_det


class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (base, flow) pair """
    
    def __init__(self, base, flows, device="cpu", prep_con=True, cond_in=49, nh=64,
                 is_freia=False):
        super().__init__()
        self.device = device
        self.base = base
        if is_freia:
            self.flow = flows
        else:
            self.flow = NormalizingFlow(flows, device, prep_con, cond_in, nh).to(device)
        self.freia_inn = is_freia

    def forward(self, x, con_in=None):
        return self.forward_sample(x, con_in)

    def log_prob(self, x, con_in=None):
        if con_in is not None:
            zs, log_det = self.flow.forward_sample(x, con_in)
        else:
            zs, log_det = self.flow.forward_sample(x)
        base_logprob = self.base.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return base_logprob+log_det

    def forward_sample(self, x, con_in=None):
        if con_in is not None:
            zs, log_det = self.flow.forward_sample(x, con_in)
        else:
            zs, log_det = self.flow.forward_sample(x)
        base_logprob = self.base.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, base_logprob, log_det

    def backward_sample(self, z, con_in=None):
        if con_in is not None:
            xs, log_det = self.flow.backward_sample(z, con_in)
        else:
            xs, log_det = self.flow.backward_sample(z)
        return xs, log_det
    
    def sample(self, num_samples, con_in=None):
        z = self.base.sample((num_samples,))
        if con_in is not None:
            xs, _ = self.flow.backward_sample(z, con_in)
        else:
            xs, _ = self.flow.backward_sample(z)
        return xs

    def r_sample(self, num_samples, con_in=None):
        z = self.base.sample((num_samples,))
        if con_in is not None:
            xs, log_det = self.flow.backward_sample(z, con_in)
        else:
            xs, log_det = self.flow.backward_sample(z)
        base_logprob = self.base.log_prob(z)
        base_logprob = base_logprob.view(xs[-1].size(0), -1).sum(1)
        return xs, base_logprob, log_det

    def sample_base(self, sample_size):
        return self.base.sample((sample_size,))

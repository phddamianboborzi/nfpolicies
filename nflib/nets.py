"""
Various helper network modules
"""

#  Copyright: (C) ETAS GmbH 2019. All rights reserved.

import torch
import torch.nn.functional as F
from torch import nn

from nflib.made import MADE

class LeafParam(nn.Module):
    """ 
    just ignores the input and outputs a parameter tensor, lol 
    todo maybe this exists in PyTorch somewhere?
    """
    def __init__(self, n):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1,n))
    
    def forward(self, x):
        return self.p.expand(x.size(0), self.p.size(1))

class PositionalEncoder(nn.Module):
    """
    Each dimension of the input gets expanded out with sins/coses
    to "carve" out the space. Useful in low-dimensional cases with
    tightly "curled up" data.
    """
    def __init__(self, freqs=(.5,1,2,4,8)):
        super().__init__()
        self.freqs = freqs
        
    def forward(self, x):
        sines = [torch.sin(x * f) for f in self.freqs]
        coses = [torch.cos(x * f) for f in self.freqs]
        out = torch.cat(sines + coses, dim=1)
        return out


class JoinConditioner(nn.Module):
    """ Preprocesses a condition and ads it to the input"""
    def __init__(self, dim, con_dim, nh):
        super().__init__()
        self.dim = dim
        # change preprocessing depending on inpur complexity
        if nh == 1:
            self.cond_net = nn.Linear(con_dim, self.dim // 2)
        else:
            self.cond_net = nn.Sequential(
                nn.Linear(con_dim, nh),
                # nn.LeakyReLU(0.2),
                # nn.Linear(nh, nh),
                nn.LeakyReLU(0.2),
                nn.Linear(nh, self.dim // 2),
            )
        self.in_linear = nn.Linear(self.dim // 2, self.dim // 2)

    def forward(self, input, cond_input):
        output = self.in_linear(input)
        output += self.cond_net(cond_input)
        return output


class MLP(nn.Module):
    """ a simple 3-layer MLP """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            # nn.Linear(nh, nh),
            # nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x)


class PosEncMLP(nn.Module):
    """ 
    Position Encoded MLP, where the first layer performs position encoding.
    Each dimension of the input gets transformed to len(freqs)*2 dimensions
    using a fixed transformation of sin/cos of given frequencies.
    """
    def __init__(self, nin, nout, nh, freqs=(.5,1,2,4,8)):
        super().__init__()
        self.net = nn.Sequential(
            PositionalEncoder(freqs),
            MLP(nin * len(freqs) * 2, nout, nh),
        )
    def forward(self, x):
        return self.net(x)


class ARMLP(nn.Module):
    """ a 4-layer auto-regressive MLP, wrapper around MADE net """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = MADE(nin, [nh, nh, nh], nout, num_masks=1, natural_ordering=True)
        
    def forward(self, x):
        return self.net(x)


class ARMLPConditional(nn.Module):
    """ a 4-layer auto-regressive MLP, wrapper around MADE net """

    def __init__(self, nin, nout, nh, con_dim=None, big_hidden=True):
        super().__init__()
        if big_hidden:
            hidden_layer = [nh, nh, nh]
        else:
            hidden_layer = [nh, nh]
        if con_dim is not None:
            self.net = MADE(nin, hidden_layer, nout, con_f=con_dim, num_masks=1, natural_ordering=True)
        else:
            self.net = MADE(nin, hidden_layer, nout, num_masks=1, natural_ordering=True)

    def forward(self, x, con_in):
        if con_in is not None:
            return self.net(x, con_in)
        else:
            return self.net(x)


#  Copyright: (C) ETAS GmbH 2019. All rights reserved.

# Implementation of Block Neural Autoregressive Flow (https://arxiv.org/abs/1904.04676)
# Code originally from https://github.com/nicola-decao/BNAF


import torch
import torch.nn as nn
import math
import numpy as np


class BNAFINN(nn.Module):
    def __init__(self, n_blocks=5, x_dim=50, nh=64, n_layers=1, model_device="cpu"):
        super().__init__()

        self.model_device = model_device
        self.residual = 'gated'  # [None, 'normal', 'gated']
        self.init_scale = 0.075 / 2  # / 10
        self.inn = self.build_inn(n_blocks, n_layers, x_dim, nh, device=model_device)
        self.inn.to(model_device)

        self.trainable_parameters = [p for p in self.inn.parameters() if p.requires_grad]
        # for p in self.trainable_parameters:
        #     p.data = self.init_scale * torch.randn_like(p)

    def build_inn(self, n_flows, n_layers, n_dims, hidden_dim, device="cpu"):
        flows = []
        for f in range(n_flows):
            layers = []
            for _ in range(n_layers - 1):
                layers.append(MaskedWeight(n_dims * hidden_dim,
                                           n_dims * hidden_dim, dim=n_dims))
                layers.append(Tanh())

            flows.append(
                BNAF(*([MaskedWeight(n_dims, n_dims * hidden_dim, dim=n_dims), Tanh()] + \
                       layers + [MaskedWeight(n_dims * hidden_dim, n_dims, dim=n_dims)]), res=self.residual if f < n_flows - 1 else None
                     )
            )

            if f < n_flows - 1:
                flows.append(Permutation(n_dims, 'flip'))

        model = Sequential(*flows).to(device)

        return model

    def forward_sample(self, state, con_in=None):
        if con_in is not None:
            raise NotImplementedError()
        else:
            z, log_det = self.inn(state)
        return [z], log_det

    def compute_log_p_x(self, x_mb):
        y_mb, log_diag_j_mb = self.inn(x_mb)
        log_p_y_mb = torch.distributions.Normal(torch.zeros_like(y_mb), torch.ones_like(y_mb)).log_prob(y_mb).sum(-1)
        return log_p_y_mb + log_diag_j_mb

    def backward(self, z, con_in=None):
        print("Unfortunately this is not implemented for bnaf and I am to lazy to take a look into it...")
        raise NotImplementedError()

    def save(self, name, epoch=0):
        print('Saving model..')
        torch.save({
            'model': self.inn.state_dict(),
            'epoch': epoch
        }, name)

    def load(self, name):
        print("loads model from ", name)
        state_dicts = torch.load(name, map_location="cpu")
        self.inn.load_state_dict(state_dicts["model"])
        self.inn.to(self.model_device)


class Sequential(torch.nn.Sequential):
    """
    Class that extends ``torch.nn.Sequential`` for computing the output of 
    the function alongside with the log-det-Jacobian of such transformation.
    """
    
    def forward(self, inputs : torch.Tensor):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        Returns
        -------
        The output tensor and the log-det-Jacobian of this transformation.
        """
        
        log_det_jacobian = 0.
        for i, module in enumerate(self._modules.values()):
            inputs, log_det_jacobian_ = module(inputs)
            log_det_jacobian = log_det_jacobian + log_det_jacobian_
        return inputs, log_det_jacobian


class BNAF(torch.nn.Sequential):
    """
    Class that extends ``torch.nn.Sequential`` for constructing a Block Neural 
    Normalizing Flow.
    """
    
    def __init__(self, *args, res: str = None):
        """
        Parameters
        ----------
        *args : ``Iterable[torch.nn.Module]``, required.
            The modules to use.
        res : ``str``, optional (default = None).
            Which kind of residual connection to use. ``res = None`` is no residual 
            connection, ``res = 'normal'`` is ``x + f(x)`` and ``res = 'gated'`` is
            ``a * x + (1 - a) * f(x)`` where ``a`` is a learnable parameter.
        """
        
        super(BNAF, self).__init__(*args)
        
        self.res = res
        
        if res == 'gated':
            self.gate = torch.nn.Parameter(torch.nn.init.normal_(torch.Tensor(1)))
    
    def forward(self, inputs : torch.Tensor):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        Returns
        -------
        The output tensor and the log-det-Jacobian of this transformation.
        """
            
        outputs = inputs
        grad = None
        
        for module in self._modules.values():
            outputs, grad = module(outputs, grad)
                
            grad = grad if len(grad.shape) == 4 else grad.view(grad.shape + [1, 1])
        
        assert inputs.shape[-1] == outputs.shape[-1]
        
        if self.res == 'normal':
            return inputs + outputs, torch.nn.functional.softplus(grad.squeeze()).sum(-1)
        elif self.res == 'gated':
            return self.gate.sigmoid() * outputs + (1 - self.gate.sigmoid()) * inputs, \
                (torch.nn.functional.softplus(grad.squeeze() + self.gate) - \
                 torch.nn.functional.softplus(self.gate)).sum(-1)
        else:
            return outputs, grad.squeeze().sum(-1)
        
    def _get_name(self):
        return 'BNAF(res={})'.format(self.res)
        
        
class Permutation(torch.nn.Module):
    """
    Module that outputs a permutation of its input.
    """
    
    def __init__(self, in_features : int, p : list = None):
        """
        Parameters
        ----------
        in_features : ``int``, required.
            The number of input features.
        p : ``list`` or ``str``, optional (default = None)
            The list of indeces that indicate the permutation. When ``p`` is not a
            list, if ``p = 'flip'``the tensor is reversed, if ``p = None`` a random 
            permutation is applied.
        """
        
        super(Permutation, self).__init__()
        
        self.in_features = in_features
        
        if p is None:
            self.p = np.random.permutation(in_features)
        elif p == 'flip':
            self.p = list(reversed(range(in_features)))
        else:
            self.p = p
        
    def forward(self, inputs : torch.Tensor):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        Returns
        -------
        The permuted tensor and the log-det-Jacobian of this permutation.
        """
        
        return inputs[:,self.p], 0
    
    def __repr__(self):
        return 'Permutation(in_features={}, p={})'.format(self.in_features, self.p)
    
    
class MaskedWeight(torch.nn.Module):
    """
    Module that implements a linear layer with block matrices with positive diagonal blocks.
    Moreover, it uses Weight Normalization (https://arxiv.org/abs/1602.07868) for stability.
    """
    
    def __init__(self, in_features : int, out_features : int, dim : int, bias : bool = True):
        """
        Parameters
        ----------
        in_features : ``int``, required.
            The number of input features per each dimension ``dim``.
        out_features : ``int``, required.
            The number of output features per each dimension ``dim``.
        dim : ``int``, required.
            The number of dimensions of the input of the flow.
        bias : ``bool``, optional (default = True).
            Whether to add a parametrizable bias.
        """
            
        super(MaskedWeight, self).__init__()
        self.in_features, self.out_features, self.dim = in_features, out_features, dim

        weight = torch.zeros(out_features, in_features)
        for i in range(dim):
            weight[i * out_features // dim:(i + 1) * out_features // dim,
                   0:(i + 1) * in_features // dim] = torch.nn.init.xavier_uniform_(
                torch.Tensor(out_features // dim, (i + 1) * in_features // dim))
            
        self._weight = torch.nn.Parameter(weight)
        self._diag_weight = torch.nn.Parameter(torch.nn.init.uniform_(torch.Tensor(out_features, 1)).log())
        
        self.bias = torch.nn.Parameter(
            torch.nn.init.uniform_(torch.Tensor(out_features), 
                                   -1 / math.sqrt(out_features),
                                   1 / math.sqrt(out_features))) if bias else 0
        
        mask_d = torch.zeros_like(weight)
        for i in range(dim):
            mask_d[i * (out_features // dim):(i + 1) * (out_features // dim),
                   i * (in_features // dim):(i + 1) * (in_features // dim)] = 1

        self.register_buffer('mask_d', mask_d)
            
        mask_o = torch.ones_like(weight)
        for i in range(dim):
            mask_o[i * (out_features // dim):(i + 1) * (out_features // dim),
                   i * (in_features // dim):] = 0
            
        self.register_buffer('mask_o', mask_o)

    def get_weights(self):
        """
        Computes the weight matrix using masks and weight normalization.
        It also compute the log diagonal blocks of it.
        """
        
        w = torch.exp(self._weight) * self.mask_d + self._weight * self.mask_o

        w_squared_norm = (w ** 2).sum(-1, keepdim=True)
        
        w = self._diag_weight.exp() * w / w_squared_norm.sqrt()
        
        wpl = self._diag_weight + self._weight - 0.5 * torch.log(w_squared_norm) 

        w_t = w.t()

        wpl_t = wpl.t()[self.mask_d.bool().t()].view(  # .byte()
            self.dim, self.in_features // self.dim, self.out_features // self.dim)

        return w_t, wpl_t


    def forward(self, inputs, grad : torch.Tensor = None):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        grad : ``torch.Tensor``, optional (default = None).
            The log diagonal block of the partial Jacobian of previous transformations.
        Returns
        -------
        The output tensor and the log diagonal blocks of the partial log-Jacobian of previous 
        transformations combined with this transformation.
        """
        
        w, wpl = self.get_weights()
        
        g = wpl.transpose(-2, -1).unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
        
        return inputs.matmul(w) + self.bias, torch.logsumexp(
            g.unsqueeze(-2) + grad.transpose(-2, -1).unsqueeze(-3), -1) if grad is not None else g

    def __repr__(self):
        return 'MaskedWeight(in_features={}, out_features={}, dim={}, bias={})'.format(
            self.in_features, self.out_features, self.dim, not isinstance(self.bias, int))

    
class Tanh(torch.nn.Tanh):
    """
    Class that extends ``torch.nn.Tanh`` additionally computing the log diagonal
    blocks of the Jacobian.
    """

    def forward(self, inputs, grad : torch.Tensor = None):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        grad : ``torch.Tensor``, optional (default = None).
            The log diagonal blocks of the partial Jacobian of previous transformations.
        Returns
        -------
        The output tensor and the log diagonal blocks of the partial log-Jacobian of previous 
        transformations combined with this transformation.
        """
        
        g = - 2 * (inputs - math.log(2) + torch.nn.functional.softplus(- 2 * inputs))
        return torch.tanh(inputs), (g.view(grad.shape) + grad) if grad is not None else g
    
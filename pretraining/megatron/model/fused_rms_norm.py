# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""This code is copied fron NVIDIA apex:
      https://github.com/NVIDIA/apex
   with some changes. """

import numbers
import torch
from torch.nn.parameter import Parameter
from torch.nn import init
import importlib

from apex.normalization.fused_layer_norm import FusedRMSNormAffineFunction


def _kernel_make_viewless_tensor(inp, requires_grad):
    '''Make a viewless tensor.

    View tensors have the undesirable side-affect of retaining a reference
    to the originally-viewed tensor, even after manually setting the '.data'
    field. This method creates a new tensor that links to the old tensor's
    data, without linking the viewed tensor, referenced via the '._base'
    field.
    '''
    out = torch.empty((1,), dtype=inp.dtype, device=inp.device, requires_grad=requires_grad,)
    out.data = inp.data
    return out


class MakeViewlessTensor(torch.autograd.Function):
    '''
    Autograd function to make a viewless tensor.

    This function should be used in cases where the computation graph needs
    to be propagated, but we only want a viewless tensor (e.g.,
    ParallelTransformer's hidden_states). Call this function by passing
    'keep_graph = True' to 'make_viewless_tensor()'.
    '''

    @staticmethod
    def forward(ctx, inp, requires_grad):
        return _kernel_make_viewless_tensor(inp, requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def make_viewless_tensor(inp, requires_grad, keep_graph):
    '''
    Entry-point for creating viewless tensors.

    This method should be used, rather than calling 'MakeViewlessTensor'
    or '_kernel_make_viewless_tensor' directly. This method acts as a
    switch for determining if an autograd function or a regular method
    should be used to create the tensor.
    '''

    # return tensor as-is, if not a 'view'
    if inp._base is None:
        return inp

    # create viewless tensor
    if keep_graph:
        return MakeViewlessTensor.apply(inp, requires_grad)
    else:
        return _kernel_make_viewless_tensor(inp, requires_grad)


class MixedFusedRMSNorm(torch.nn.Module):

    def __init__(self, normalized_shape, eps=1e-5,
                no_persist_layer_norm=True,
                sequence_parallel=False,
                apply_layernorm_1p=False,
                init_weight=None,
                freeze=False,
                init_method=None,
                hf_checkpoint=None):
        super(MixedFusedRMSNorm, self).__init__()

        assert apply_layernorm_1p == False, "1p layernorm not supported in mixed precision"

        self.init_weight = init_weight
        assert self.init_weight is None or isinstance(self.init_weight, float), \
            "Cannot init_weight of None or of non-float"


        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()
        self.sequence_parallel = sequence_parallel

        if init_method is not None:
            init_method(self.weight)

        if hf_checkpoint is not None:
            hf_checkpoint = hf_checkpoint.to(self.weight)
            assert hf_checkpoint.shape == self.weight.shape, f"checkpoint shape {hf_checkpoint.shape} does not match weight shape {self.weight.shape}"
            self.weight.data = hf_checkpoint

        # set sequence parallelism flag on weight and bias parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)

        self.freeze = freeze


    def reset_parameters(self):
        if self.init_weight:
            init.constant_(self.weight, self.init_weight)
        else:
            init.ones_(self.weight)


    def forward(self, x):
        # don't track gradient of the whole layer
        if self.freeze:
            for p in self.parameters():
                p.requires_grad = False

        weight = self.weight.to(x.dtype)

        return FusedRMSNormAffineFunction.apply(x, weight, self.normalized_shape, self.eps)

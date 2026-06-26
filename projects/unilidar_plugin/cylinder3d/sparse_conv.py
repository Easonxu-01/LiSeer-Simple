'''
Author: EASON XU
Date: 2025-05-31 09:20:42
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2025-06-03 01:36:44
Description: 头部注释
FilePath: /UniLiDAR/projects/unilidar_plugin/cylinder3d/sparse_conv.py
'''
import math
from typing import Optional, Union, List, Tuple
from torch import nn
import numpy as np
import torch
from torch.nn import init
from torch.nn.parameter import Parameter
from spconv.pytorch import ops
from typing import Any
import torch
from torch.autograd import Function


def scatter_nd(indices: torch.Tensor, updates: torch.Tensor,
               shape: torch.Tensor) -> torch.Tensor:
    """pytorch edition of tensorflow scatter_nd.

    this function don't contain except handle code. so use this carefully when
    indice repeats, don't support repeat add which is supported in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret



class SparseConvFunction(Function):
    """Sparse Convolution.

    Please refer to `SECOND <https://www.mdpi.com/1424-8220/18/10/3337>`_ for
    more details.
    """

    @staticmethod
    def forward(ctx: Any, features: torch.Tensor, filters: torch.nn.Parameter,
                indice_pairs: torch.Tensor, indice_pair_num: torch.Tensor,
                num_activate_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Features that needs to convolute.
            filters (torch.nn.parameter.Parameter): Convolution filters.
            indice_pairs (torch.Tensor): Indice pairs between inputs locations
                and outputs locations.
            indice_pair_num (torch.Tensor): Indice pairs num.
            num_activate_out (torch.Tensor): Output channels num.

        Returns:
            torch.Tensor: Output features from gather-gemm-scatter.
        """
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters)
        return indice_conv(features, filters, indice_pairs,
                               indice_pair_num, num_activate_out, False)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        input_bp, filters_bp = ops.indice_conv_backward(
            features, filters, grad_output, indice_pairs, indice_pair_num,
            False)

        return input_bp, filters_bp, None, None, None


class SparseInverseConvFunction(Function):

    @staticmethod
    def forward(ctx: Any, features: torch.Tensor, filters: torch.nn.Parameter,
                indice_pairs: torch.Tensor, indice_pair_num: torch.Tensor,
                num_activate_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Features that needs to convolute.
            filters (torch.nn.parameter.Parameter): Convolution filters.
            indice_pairs (torch.Tensor): Indice pairs between inputs locations
                and outputs locations.
            indice_pair_num (torch.Tensor): Indice pairs num.
            num_activate_out (torch.Tensor): Output channels num.

        Returns:
            torch.Tensor: Output features from gather-gemm-scatter.
        """
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters)
        return indice_conv(features, filters, indice_pairs,
                               indice_pair_num, num_activate_out, True, False)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        input_bp, filters_bp = ops.indice_conv_backward(
            features, filters, grad_output, indice_pairs, indice_pair_num,
            True, False)

        return input_bp, filters_bp, None, None, None


class SubMConvFunction(Function):

    @staticmethod
    def forward(ctx: Any, features: torch.Tensor, filters: torch.nn.Parameter,
                indice_pairs: torch.Tensor, indice_pair_num: torch.Tensor,
                num_activate_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Features that needs to convolute.
            filters (torch.nn.parameter.Parameter): Convolution filters.
            indice_pairs (torch.Tensor): Indice pairs between inputs locations
                and outputs locations.
            indice_pair_num (torch.Tensor): Indice pairs num.
            num_activate_out (torch.Tensor): Output channels num.

        Returns:
            torch.Tensor: Output features from gather-gemm-scatter.
        """
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters)
        return indice_conv(features, filters, indice_pairs,
                               indice_pair_num, num_activate_out, False, True)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        input_bp, filters_bp = ops.indice_conv_backward(
            features, filters, grad_output, indice_pairs, indice_pair_num,
            False, True)

        return input_bp, filters_bp, None, None, None


class SparseMaxPoolFunction(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, indice_pairs: torch.Tensor,
                indice_pair_num: torch.Tensor,
                num_activate_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Features that needs to convolute.
            indice_pairs (torch.Tensor): Indice pairs between inputs locations
                and outputs locations.
            indice_pair_num (torch.Tensor): Indice pairs num.
            num_activate_out (torch.Tensor): Output channels num.

        Returns:
            torch.Tensor: Output features from sparse maxpooling.
        """
        out = indice_maxpool(features, indice_pairs, indice_pair_num,
                                 num_activate_out)
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, out)
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        indice_pairs, indice_pair_num, features, out = ctx.saved_tensors
        input_bp = ops.indice_maxpool_backward(features, out, grad_output,
                                               indice_pairs, indice_pair_num)
        return input_bp, None, None, None


indice_conv = SparseConvFunction.apply
indice_inverse_conv = SparseInverseConvFunction.apply
indice_subm_conv = SubMConvFunction.apply
indice_maxpool = SparseMaxPoolFunction.apply


def scatter_nd(indices: torch.Tensor, updates: torch.Tensor,
               shape: torch.Tensor) -> torch.Tensor:
    """pytorch edition of tensorflow scatter_nd.

    this function don't contain except handle code. so use this carefully when
    indice repeats, don't support repeat add which is supported in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret

class SparseModule(nn.Module):
    """place holder, All module subclass from this will take sptensor in
    SparseSequential."""
    pass

class SparseConvTensor:
    def __init__(self,
                 features: torch.Tensor,
                 indices: torch.Tensor,
                 spatial_shape: Union[List, Tuple],
                 batch_size: int,
                 grid: Optional[torch.Tensor] = None):
        self.features = features
        self.indices = indices
        if self.indices.dtype != torch.int32:
            self.indices.int()
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
        self.indice_dict: dict = {}
        self.grid = grid

    @property
    def spatial_size(self):
        return np.prod(self.spatial_shape)

    def find_indice_pair(self, key):
        if key is None:
            return None
        if key in self.indice_dict:
            return self.indice_dict[key]
        return None

    def dense(self, channels_first: bool = True) -> torch.Tensor:
        output_shape = [self.batch_size] + list(
            self.spatial_shape) + [self.features.shape[1]]
        res = scatter_nd(self.indices.long(), self.features, output_shape)
        if not channels_first:
            return res
        ndim = len(self.spatial_shape)
        trans_params = list(range(0, ndim + 1))
        trans_params.insert(1, ndim + 1)
        return res.permute(*trans_params).contiguous()

    @property
    def sparity(self):
        return (self.indices.shape[0] / np.prod(self.spatial_shape) /
                self.batch_size)
        
def _calculate_fan_in_and_fan_out_hwio(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError('fan in and fan out can not be computed for tensor'
                         'with fewer than 2 dimensions')

    if dimensions == 2:  # Linear
        fan_in = tensor.size(-2)
        fan_out = tensor.size(-1)
    else:
        num_input_fmaps = tensor.size(-2)
        num_output_fmaps = tensor.size(-1)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[..., 0, 0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def get_conv_output_size(input_size, kernel_size, stride, padding, dilation):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        size = (input_size[i] + 2 * padding[i] - dilation[i] *
                (kernel_size[i] - 1) - 1) // stride[i] + 1
        if kernel_size[i] == -1:
            output_size.append(1)
        else:
            output_size.append(size)
    return output_size


def get_deconv_output_size(input_size, kernel_size, stride, padding, dilation,
                           output_padding):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        if kernel_size[i] == -1:
            raise ValueError("deconv don't support kernel_size < 0")
        size = (input_size[i] - 1) * stride[i] - 2 * padding[i] + kernel_size[
            i] + output_padding[i]
        output_size.append(size)
    return output_size

def indice_conv(features,
                filters,
                indice_pairs,
                indice_pair_num,
                num_activate_out,
                inverse=False,
                subm=False):
    if filters.dtype == torch.float32 or filters.dtype == torch.half:
        return ops.indice_conv_forward(features, filters, indice_pairs,
                                     indice_pair_num, num_activate_out,
                                     int(inverse), int(subm))
    else:
        raise NotImplementedError


def fused_indice_conv(features, filters, bias, indice_pairs, indice_pair_num,
                      num_activate_out, inverse, subm):
    if features.dtype == torch.half or filters.dtype == torch.float32:
        return ops.fused_indice_conv_forward(features, filters, bias,
                                           indice_pairs, indice_pair_num,
                                           num_activate_out, int(inverse),
                                           int(subm))
    else:
        raise NotImplementedError

        
class SparseConvolution(SparseModule):
    
    def __init__(self,
                 ndim,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 subm=False,
                 output_padding=0,
                 transposed=False,
                 inverse=False,
                 indice_key=None,
                 fused_bn=False):
        super().__init__()
        assert groups == 1
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size] * ndim
        if not isinstance(stride, (list, tuple)):
            stride = [stride] * ndim
        if not isinstance(padding, (list, tuple)):
            padding = [padding] * ndim
        if not isinstance(dilation, (list, tuple)):
            dilation = [dilation] * ndim
        if not isinstance(output_padding, (list, tuple)):
            output_padding = [output_padding] * ndim

        for d, s in zip(dilation, stride):
            assert any([s == 1, d == 1]), "don't support this."

        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv1x1 = np.prod(kernel_size) == 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.inverse = inverse
        self.output_padding = output_padding
        self.groups = groups
        self.subm = subm
        self.indice_key = indice_key
        self.fused_bn = fused_bn

        self.weight = Parameter(
            torch.Tensor(*kernel_size, in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out_hwio(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        assert isinstance(input, SparseConvTensor)
        features = input.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        if not self.subm:
            if self.transposed:
                out_spatial_shape = get_deconv_output_size(
                    spatial_shape, self.kernel_size, self.stride, self.padding,
                    self.dilation, self.output_padding)
            else:
                out_spatial_shape = get_conv_output_size(
                    spatial_shape, self.kernel_size, self.stride, self.padding,
                    self.dilation)

        else:
            out_spatial_shape = spatial_shape

        if self.conv1x1:
            features = torch.mm(
                input.features,
                self.weight.view(self.in_channels, self.out_channels))
            if self.bias is not None:
                features += self.bias
            out_tensor = SparseConvTensor(features, input.indices,
                                          input.spatial_shape,
                                          input.batch_size)
            out_tensor.indice_dict = input.indice_dict
            out_tensor.grid = input.grid
            return out_tensor
        data = input.find_indice_pair(self.indice_key)
        if self.inverse:
            assert data is not None and self.indice_key is not None
            _, outids, indice_pairs, indice_pair_num, out_spatial_shape = data
            assert indice_pairs.shape[0] == np.prod(
                self.kernel_size
            ), 'inverse conv must have same kernel size as its couple conv'
        else:
            if self.indice_key is not None and data is not None:
                outids, _, indice_pairs, indice_pair_num, _ = data
            else:
                outids, indice_pairs, indice_pair_num = ops.get_indice_pairs(
                    indices,
                    batch_size,
                    spatial_shape,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.output_padding,
                    self.subm,
                    self.transposed,
                    )
                input.indice_dict[self.indice_key] = (outids, indices,
                                                      indice_pairs,
                                                      indice_pair_num,
                                                      spatial_shape)
        if self.fused_bn:
            assert self.bias is not None
            out_features = fused_indice_conv(features, self.weight,
                                                 self.bias,
                                                 indice_pairs.to(device),
                                                 indice_pair_num,
                                                 outids.shape[0], self.inverse,
                                                 self.subm)
        else:
            if self.subm:
                out_features = indice_subm_conv(features, self.weight,
                                                    indice_pairs.to(device),
                                                    indice_pair_num,
                                                    outids.shape[0])
            else:
                if self.inverse:
                    out_features = indice_inverse_conv(
                        features, self.weight, indice_pairs.to(device),
                        indice_pair_num, outids.shape[0])
                else:
                    out_features = indice_conv(features, self.weight,
                                                   indice_pairs.to(device),
                                                   indice_pair_num,
                                                   outids.shape[0])

            if self.bias is not None:
                out_features += self.bias
        out_tensor = SparseConvTensor(out_features, outids, out_spatial_shape,
                                      batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor

class SparseConv2d(SparseConvolution):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None):
        super().__init__(
            2,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            indice_key=indice_key)


class SparseConv3d(SparseConvolution):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None):
        super().__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            indice_key=indice_key)

class SubMConv3d(SparseConvolution):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None):
        super().__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            True,
            indice_key=indice_key)

class SparseInverseConv3d(SparseConvolution):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 indice_key=None,
                 bias=True):
        super().__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            bias=bias,
            inverse=True,
            indice_key=indice_key)

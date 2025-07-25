# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FP8 Padding API"""

from typing import List, Optional, Tuple

import torch

import transformer_engine_torch as tex

from ..fp8 import FP8GlobalStateManager
from ..jit import no_torch_dynamo
from ..tensor.quantized_tensor import QuantizedTensor
from ..tensor.float8_blockwise_tensor import Float8BlockwiseQTensor
from ..tensor.mxfp8_tensor import MXFP8Tensor

__all__ = ["Fp8Padding"]


class _Fp8Padding(torch.autograd.Function):
    """functional FP8 padding"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        m_splits: List[int],
        padded_m_splits: Optional[List[int]],
        padded_scale_splits: Optional[List[int]],
        is_grad_enabled: bool,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring

        if isinstance(inp, QuantizedTensor):
            # Each m in m_splits indicates a tensor. So for tensor-wise scaled tensors,
            # we should always pad the high precision tensors and then do multi-quantize.
            # For MXFP8, padding doesn't make sense.
            assert (
                isinstance(inp, MXFP8Tensor) or
                (isinstance(inp, Float8BlockwiseQTensor) and not inp._is_2D_scaled and inp._data_format == tex.Float8BlockScaleTensorFormat.COMPACT)
            ), (
                "Fp8Padding only supports"
                "1. mxfp8"
                "2. blockwise 1D scaled tensor with compact data and scales."
            )

            if padded_m_splits is not None:
                total_row = sum(padded_m_splits)
                in_features = inp._rowwise_data.shape[-1]
                rowwise_data = inp._rowwise_data.view(-1, in_features)
                out_data = torch.empty(
                    [total_row, in_features], dtype=inp._rowwise_data.dtype, device=inp.device
                )
                tex.fused_multi_row_padding(rowwise_data, out_data, m_splits, padded_m_splits)
            else:
                out_data = inp._rowwise_data

            if padded_scale_splits is not None:
                total_row = sum(padded_scale_splits)
                in_scale_features = inp._rowwise_scale_inv.shape[-1]
                rowwise_scale_inv = inp._rowwise_scale_inv.view(-1, in_scale_features).contiguous()
                out_scale_inv = torch.empty(
                    [total_row, in_scale_features],
                    dtype=inp._rowwise_scale_inv.dtype,
                    device=inp.device,
                )
                tex.fused_multi_row_padding(rowwise_scale_inv, out_scale_inv, m_splits, padded_scale_splits)
            else:
                out_scale_inv = inp._rowwise_scale_inv

            if isinstance(inp, MXFP8Tensor):
                out = MXFP8Tensor(
                    shape=out_data.shape,
                    dtype=inp.dtype,
                    fp8_dtype=inp._fp8_dtype,
                    rowwise_data=out_data,
                    rowwise_scale_inv=out_scale_inv.contiguous(),
                    columnwise_data=None,
                    columnwise_scale_inv=None,
                    quantizer=inp._get_quantizer(),
                    requires_grad=inp.requires_grad,
                )
            elif isinstance(inp, Float8BlockwiseQTensor):
                out = Float8BlockwiseQTensor(
                    shape=out_data.shape,
                    dtype=inp.dtype,
                    rowwise_data=out_data,
                    rowwise_scale_inv=out_scale_inv.contiguous(),
                    columnwise_data=None,
                    columnwise_scale_inv=None,
                    fp8_dtype=inp._fp8_dtype,
                    quantizer=inp._get_quantizer(),
                    is_2D_scaled=False,
                    requires_grad=inp.requires_grad,
                    data_format=tex.Float8BlockScaleTensorFormat.COMPACT,
                )
        else:
            if padded_m_splits is not None:
                total_row = sum(padded_m_splits)
                # Make sure input dimensions are compatible
                in_features = inp.shape[-1]

                # Allocate cast and transpose output tensor
                out = torch.empty([total_row, in_features], dtype=inp.dtype, device=inp.device)

                tex.fused_multi_row_padding(inp.view(-1, in_features), out, m_splits, padded_m_splits)

        if is_grad_enabled:
            ctx.m_splits = m_splits
            ctx.padded_m_splits = padded_m_splits
            ctx.requires_dgrad = inp.requires_grad

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # pylint: disable=missing-function-docstring

        grad_input = None
        if ctx.requires_dgrad and ctx.padded_m_splits is not None:
            grad_output = grad_output.contiguous()

            in_features = grad_output.shape[-1]

            # Allocate cast and transpose output tensor
            total_row = sum(ctx.m_splits)
            grad_input = torch.empty(
                [total_row, in_features], dtype=grad_output.dtype, device=grad_output.device
            )

            tex.fused_multi_row_unpadding(
                grad_output.view(-1, in_features), grad_input, ctx.padded_m_splits, ctx.m_splits
            )

        return (grad_input, None, None, None, None)


class Fp8Padding(torch.nn.Module):
    """
    Apply the padding for Grouped GEMM input.

    Parameters
    ----------
    num_gemms : int
                number of GEMMs to be performed simultaneously.
    align_size : int, optional
                 the alignment size for the input tensor. If not provided, the alignment size will
                 be determined by the FP8 recipe (32 for MXFP8 and 16 for others) in the first
                 forward pass.
    """

    def __init__(
        self,
        num_gemms: int,
        align_size: Optional[int] = None,
        scale_align_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.num_gemms = num_gemms
        self.align_size = align_size
        self.scale_align_size = scale_align_size

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        m_splits: List[int],
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Apply the padding to the input.

        Parameters
        ----------
        inp : torch.Tensor
                Input tensor.
        m_splits : List[int]
                    List of integers representing the split of the input tensor.
        """

        assert len(m_splits) == self.num_gemms, "Number of splits should match number of GEMMs."
        if self.align_size is None:
            self.align_size = 32 if FP8GlobalStateManager.get_fp8_recipe().mxfp8() else 16
        if self.scale_align_size is None:
            self.scale_align_size = 128 if FP8GlobalStateManager.get_fp8_recipe().mxfp8() else self.align_size
        
        # FP8 padding calculate
        padded_m_splits = [
            (m + self.align_size - 1) // self.align_size * self.align_size for m in m_splits
        ]
        padded_scale_splits = [
            (m + self.scale_align_size - 1) // self.scale_align_size * self.scale_align_size for m in m_splits
        ]
        # no padding needed
        if m_splits == padded_m_splits and m_splits == padded_scale_splits:
            return inp, m_splits

        if torch.is_grad_enabled():
            fn = _Fp8Padding.apply
            args = []
        else:
            fn = _Fp8Padding.forward
            args = [None]

        args += (
            inp,
            m_splits,
            padded_m_splits if m_splits != padded_m_splits else None,
            padded_scale_splits if m_splits != padded_scale_splits else None,
            torch.is_grad_enabled(),
        )
        out = fn(*args)

        return out, padded_m_splits
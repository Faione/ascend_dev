"""MXFP8 quantize -> dequantize -> add_pipeline teaching example.

The scale is encoded as an E8M0 byte (biased power-of-two exponent).  The
existing add_pipeline kernel still adds float32 values; this example exposes
the quantization boundary without claiming native FP8 arithmetic in that kernel.
"""

import torch

GROUP_SIZE = 32
FP8_MAX = 448.0  # E4M3FN


def quantize_mxfp8(x: torch.Tensor):
    """Return E4M3FN values and one E8M0 scale byte per group of 32."""
    if x.ndim != 1 or x.dtype != torch.float32 or not x.is_contiguous():
        raise ValueError("expected a contiguous 1-D float32 tensor")
    if not torch.isfinite(x).all().item():
        raise ValueError("this teaching example accepts finite values only")
    n = x.numel()
    if n == 0:
        return x.to(torch.float8_e4m3fn), torch.empty(0, device=x.device, dtype=torch.uint8)

    # Padding affects neither the maximum nor the returned data.
    padded = torch.nn.functional.pad(x, (0, (-n) % GROUP_SIZE))
    amax = padded.abs().reshape(-1, GROUP_SIZE).amax(dim=1)
    # OCP-style scale used in ops-transformer's grouped_matmul_swiglu_quant:
    # exponent = floor(log2(amax)) - 8. Clamp to E8M0's finite exponent range.
    exponent = torch.floor(torch.log2(torch.where(amax > 0, amax, 1.0))) - 8
    exponent = exponent.clamp(-127, 127).to(torch.int32)
    encoded = torch.where(amax > 0, exponent + 127, 0).to(torch.uint8)
    scale = torch.pow(2.0, encoded.to(torch.float32) - 127.0)
    repeated = scale.repeat_interleave(GROUP_SIZE)[:n]
    # E4M3FN conversion rounds the scaled values; clamp handles saturation.
    quantized = (x / repeated).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return quantized, encoded


def dequantize_mxfp8(values: torch.Tensor, encoded_scale: torch.Tensor):
    if values.dtype != torch.float8_e4m3fn or encoded_scale.dtype != torch.uint8:
        raise ValueError("expected E4M3FN values and uint8 E8M0 scale bytes")
    if values.ndim != 1 or encoded_scale.numel() != (values.numel() + 31) // 32:
        raise ValueError("scale shape does not match 32-element groups")
    scale = torch.pow(2.0, encoded_scale.to(torch.float32) - 127.0)
    return values.float() * scale.repeat_interleave(GROUP_SIZE)[:values.numel()]


def mxfp8_add_pipeline(x: torch.Tensor, y: torch.Tensor, add=None):
    """Quantize both inputs, then call the original FP32 Add kernel.

    Pass ``torch.ops.ascend_add_pipeline.add_pipeline`` as add on an NPU.
    The default torch.add allows a CPU numerical demonstration.
    """
    if x.shape != y.shape or x.device != y.device:
        raise ValueError("x and y must have the same shape and device")
    x8, sx = quantize_mxfp8(x)
    y8, sy = quantize_mxfp8(y)
    dx = dequantize_mxfp8(x8, sx)
    dy = dequantize_mxfp8(y8, sy)
    result = (torch.add if add is None else add)(dx, dy)
    return result, (x8, sx), (y8, sy)


if __name__ == "__main__":
    x = torch.tensor([1.0, -2.0, 0.1] * 11, dtype=torch.float32)
    y = torch.tensor([0.25, 1.5, -0.3] * 11, dtype=torch.float32)
    result, (_, sx), (_, sy) = mxfp8_add_pipeline(x, y)
    print("groups:", sx.tolist(), sy.tolist())
    print("max absolute error:", (result - (x + y)).abs().max().item())

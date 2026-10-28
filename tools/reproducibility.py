"""Deterministic training for Mask2Former's bilinear deformable attention."""
import os

import torch
from torch.nn import functional as F
from transformers.models.mask2former import modeling_mask2former as mask2former


def _bilinear_values(value, grid):
    # Same zero padding and align_corners=False coordinates as grid_sample.
    batch, channels, height, width = value.shape
    x = ((grid[..., 0] + 1) * width - 1) / 2
    y = ((grid[..., 1] + 1) * height - 1) / 2
    left, top = x.floor(), y.floor()
    dx, dy = x - left, y - top
    # Select whole channel rows: the deterministic backward sorts one index per
    # sample instead of duplicating/sorting that index for every channel.
    flat = value.permute(0, 2, 3, 1).reshape(batch * height * width, channels)
    offsets = torch.arange(batch, device=value.device).reshape(batch, 1, 1) * height * width

    def corner(cx, cy):
        valid = (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height)
        index = (cy.clamp(0, height - 1).long() * width + cx.clamp(0, width - 1).long())
        index = index + offsets
        sampled = flat.index_select(0, index.flatten()).reshape(batch, *x.shape[1:], channels)
        return sampled.movedim(-1, 1) * valid[:, None]

    return (corner(left, top) * ((1 - dx) * (1 - dy))[:, None]
            + corner(left + 1, top) * (dx * (1 - dy))[:, None]
            + corner(left, top + 1) * ((1 - dx) * dy)[:, None]
            + corner(left + 1, top + 1) * (dx * dy)[:, None])


class _DeterministicGridSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, grid):
        ctx.save_for_backward(value, grid)
        # Retain the original CUDA forward, including its rounding and autocast.
        return F.grid_sample(value, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    @staticmethod
    def backward(ctx, output_gradient):
        value, grid = ctx.saved_tensors
        # CUDA grid_sample backward uses atomic addition. Row selection has a
        # deterministic implementation when deterministic algorithms are enabled.
        with torch.enable_grad(), torch.autocast(value.device.type, enabled=False):
            dtype = torch.promote_types(value.dtype, grid.dtype)
            dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
            v = value.detach().to(dtype).requires_grad_(True)
            g = grid.detach().to(dtype).requires_grad_(True)
            sampled = _bilinear_values(v, g)
            dv, dg = torch.autograd.grad(sampled, (v, g), output_gradient.to(dtype))
        return dv.to(value.dtype), dg.to(grid.dtype)


_original_attention = mask2former.multi_scale_deformable_attention


def _deterministic_attention(value, value_spatial_shapes, sampling_locations, attention_weights):
    if not torch.is_grad_enabled() or not torch.are_deterministic_algorithms_enabled():
        return _original_attention(value, value_spatial_shapes, sampling_locations, attention_weights)
    batch, _, heads, channels = value.shape
    _, queries, _, levels, points, _ = sampling_locations.shape
    values = value.split([height * width for height, width in value_spatial_shapes], dim=1)
    grids = 2 * sampling_locations - 1
    sampled = []
    for level, (height, width) in enumerate(value_spatial_shapes):
        image = values[level].flatten(2).transpose(1, 2).reshape(batch * heads, channels, height, width)
        grid = grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled.append(_DeterministicGridSample.apply(image, grid))
    weights = attention_weights.transpose(1, 2).reshape(batch * heads, 1, queries, levels * points)
    result = (torch.stack(sampled, dim=-2).flatten(-2) * weights).sum(-1)
    return result.view(batch, heads * channels, queries).transpose(1, 2).contiguous()


def configure_determinism(enabled):
    if enabled:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.benchmark = not enabled
    torch.backends.cudnn.deterministic = enabled
    mask2former.multi_scale_deformable_attention = _deterministic_attention if enabled else _original_attention


def runtime_record(device):
    return dict(torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(device), device=str(device),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                threads=torch.get_num_threads(), autocast_dtype='bfloat16')

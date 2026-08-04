#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate CUDA or Ascend Gated Delta Rule data and accuracy metrics.

The positional CLI and output filename follow generate_vllm_gdn_ref.py from
the gdn_dev workspace. The device result comes from the corresponding vLLM
FLA implementation; an independent CPU float32 recurrent implementation is
used as the accuracy golden reference.
"""

from __future__ import annotations

import argparse
import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F


GDN_CHUNK_SIZE = 64
DEFAULT_OUTPUT_DIR = Path("/tmp/gdn_vllm_ref")


def parse_dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]


def parse_cu_seqlens(value: str, batch: int, seqlen: int) -> list[int] | None:
    if value.strip().lower() in {"", "none"}:
        return None
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (list, tuple)) or len(parsed) < 2:
        raise ValueError("cu_seqlens must be None or a list with at least two entries")
    cu_seqlens = [int(item) for item in parsed]
    if cu_seqlens[0] != 0 or any(end <= start for start, end in zip(cu_seqlens, cu_seqlens[1:])):
        raise ValueError("cu_seqlens must start at 0 and be strictly increasing")
    if batch != 1:
        raise ValueError("B must be 1 when cu_seqlens is provided")
    if cu_seqlens[-1] != seqlen:
        raise ValueError(f"cu_seqlens must end at T ({seqlen}), got {cu_seqlens[-1]}")
    return cu_seqlens


def generate_inputs(
    batch: int,
    seqlen: int,
    key_heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    q = torch.normal(0.0109, 0.0979, (batch, seqlen, key_heads, key_dim), device=device, dtype=dtype)
    k = torch.normal(0.0109, 0.0979, (batch, seqlen, key_heads, key_dim), device=device, dtype=dtype)
    v = torch.normal(0.0168, 0.1328, (batch, seqlen, value_heads, value_dim), device=device, dtype=dtype)
    g = torch.empty(batch, seqlen, value_heads, device=device, dtype=torch.float32).uniform_(-16.0, -0.1)
    beta = torch.empty(batch, seqlen, value_heads, device=device, dtype=dtype).uniform_(0.12, 0.88)
    return {"q": q, "k": k, "v": v, "g": g, "beta": beta}


def recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the recurrence and chunk_fwd_o inputs in float32 on CPU."""
    q, k, v, g, beta = (tensor.detach().cpu() for tensor in (q, k, v, g, beta))
    batch, seqlen, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    groups = value_heads // key_heads
    q = q.float().repeat_interleave(groups, dim=2) * scale
    k = k.float().repeat_interleave(groups, dim=2)
    v, g, beta = v.float(), g.float(), beta.float()

    if cu_seqlens is None:
        spans = [(batch_idx, 0, seqlen) for batch_idx in range(batch)]
    else:
        spans = [(0, start, end) for start, end in zip(cu_seqlens, cu_seqlens[1:])]

    output = torch.empty(batch, seqlen, value_heads, value_dim, device=q.device, dtype=torch.float32)
    v_new = torch.empty_like(v)
    g_cumsum = torch.empty_like(g)
    chunk_states: list[torch.Tensor] = []
    final_states = []
    for batch_idx, start, end in spans:
        state = torch.zeros(value_heads, key_dim, value_dim, device=q.device, dtype=torch.float32)
        for token_idx in range(start, end):
            if (token_idx - start) % GDN_CHUNK_SIZE == 0:
                chunk_states.append(state.clone())
                chunk_g = torch.zeros(value_heads, dtype=torch.float32)
            q_t = q[batch_idx, token_idx]
            k_t = k[batch_idx, token_idx]
            v_t = v[batch_idx, token_idx]
            state.mul_(g[batch_idx, token_idx].exp()[:, None, None])
            prediction = torch.einsum("hkv,hk->hv", state, k_t)
            delta = (v_t - prediction) * beta[batch_idx, token_idx, :, None]
            v_new[batch_idx, token_idx] = delta
            chunk_g.add_(g[batch_idx, token_idx])
            g_cumsum[batch_idx, token_idx] = chunk_g
            state.add_(torch.einsum("hk,hv->hkv", k_t, delta))
            output[batch_idx, token_idx] = torch.einsum("hk,hkv->hv", q_t, state)
        final_states.append(state.clone())
    if cu_seqlens is None:
        num_chunks = (seqlen + GDN_CHUNK_SIZE - 1) // GDN_CHUNK_SIZE
        h = torch.stack(chunk_states).reshape(batch, num_chunks, value_heads, key_dim, value_dim)
    else:
        h = torch.stack(chunk_states).unsqueeze(0)
    return output, torch.stack(final_states), v_new, h, g_cumsum


def resolve_device(requested: str) -> torch.device:
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if requested in {"auto", "npu"}:
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            if requested == "npu":
                raise RuntimeError("--device npu requires torch_npu") from None
        else:
            if torch.npu.is_available():
                return torch.device("npu")
    raise RuntimeError(f"Requested device '{requested}' is not available")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    else:
        torch.npu.synchronize()


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(torch.cuda.current_device())
    get_device_name = getattr(torch.npu, "get_device_name", None)
    return get_device_name(torch.npu.current_device()) if get_device_name else "Ascend NPU"


def install_minimal_vllm_shim() -> None:
    """Provide the small vLLM API surface used by the copied FLA kernels."""
    try:
        import vllm  # noqa: F401
        return
    except ImportError:
        pass

    import triton
    import triton.language as tl

    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    triton_utils = types.ModuleType("vllm.triton_utils")
    triton_utils.HAS_TRITON = True
    triton_utils.tl = tl
    triton_utils.triton = triton
    forward_context = types.ModuleType("vllm.forward_context")
    forward_context.get_forward_context = lambda: types.SimpleNamespace(attn_metadata=None)
    distributed = types.ModuleType("vllm.distributed")
    distributed.get_pcp_group = lambda: types.SimpleNamespace(world_size=1, rank_in_group=0)
    utils = types.ModuleType("vllm.model_executor.layers.fla.ops.utils")
    utils.SUPPRESS_LEVEL = 3

    modules = {
        "vllm": vllm,
        "vllm.triton_utils": triton_utils,
        "vllm.forward_context": forward_context,
        "vllm.distributed": distributed,
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": types.ModuleType("vllm.model_executor.layers"),
        "vllm.model_executor.layers.fla": types.ModuleType("vllm.model_executor.layers.fla"),
        "vllm.model_executor.layers.fla.ops": types.ModuleType("vllm.model_executor.layers.fla.ops"),
        "vllm.model_executor.layers.fla.ops.utils": utils,
    }
    sys.modules.update(modules)
    vllm.triton_utils = triton_utils
    vllm.forward_context = forward_context
    vllm.distributed = distributed


def install_lightweight_ascend_packages() -> None:
    """Bypass vllm_ascend.ops.__init__, which imports the full vLLM stack."""
    repo_root = Path(__file__).resolve().parents[1]
    package_paths = {
        "vllm_ascend.ops": repo_root / "vllm_ascend" / "ops",
        "vllm_ascend.ops.triton": repo_root / "vllm_ascend" / "ops" / "triton",
        "vllm_ascend.ops.triton.fla": repo_root / "vllm_ascend" / "ops" / "triton" / "fla",
    }
    for name, path in package_paths.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module

    from vllm.triton_utils import tl, triton

    @triton.jit
    def insert_slice(ful, sub, offsets, sizes, strides):
        if len(ful.shape) == 2:
            rows = tl.arange(0, ful.shape[0])[:, None]
            cols = tl.arange(0, ful.shape[1])[None, :]
            src_rows = (rows - offsets[0]) // strides[0]
            src_cols = (cols - offsets[1]) // strides[1]
            mask = (
                (rows >= offsets[0])
                & (rows < offsets[0] + sizes[0] * strides[0])
                & (cols >= offsets[1])
                & (cols < offsets[1] + sizes[1] * strides[1])
            )
            return tl.where(mask, sub[src_rows, src_cols], ful)
        dim0 = tl.arange(0, ful.shape[0])[:, None, None]
        dim1 = tl.arange(0, ful.shape[1])[None, :, None]
        dim2 = tl.arange(0, ful.shape[2])[None, None, :]
        src0 = (dim0 - offsets[0]) // strides[0]
        src1 = (dim1 - offsets[1]) // strides[1]
        src2 = (dim2 - offsets[2]) // strides[2]
        mask = (
            (dim0 >= offsets[0])
            & (dim0 < offsets[0] + sizes[0] * strides[0])
            & (dim1 >= offsets[1])
            & (dim1 < offsets[1] + sizes[1] * strides[1])
            & (dim2 >= offsets[2])
            & (dim2 < offsets[2] + sizes[2] * strides[2])
        )
        return tl.where(mask, sub[src0, src1, src2], ful)

    @triton.jit
    def extract_slice(ful, offsets, sizes, strides):
        if len(ful.shape) == 2:
            rows = offsets[0] + tl.arange(0, sizes[0])[:, None] * strides[0]
            cols = offsets[1] + tl.arange(0, sizes[1])[None, :] * strides[1]
            return ful[rows, cols]
        dim0 = offsets[0] + tl.arange(0, sizes[0])[:, None, None] * strides[0]
        dim1 = offsets[1] + tl.arange(0, sizes[1])[None, :, None] * strides[1]
        dim2 = offsets[2] + tl.arange(0, sizes[2])[None, None, :] * strides[2]
        return ful[dim0, dim1, dim2]

    ascend_utils = types.ModuleType("vllm_ascend.ops.triton.triton_utils")
    ascend_utils.insert_slice = insert_slice
    ascend_utils.extract_slice = extract_slice
    ascend_utils.get_element = lambda tensor, *indices: tensor[indices]
    ascend_utils.get_vectorcore_num = lambda: 1
    ascend_utils.get_aicore_num = lambda: 1
    ascend_utils.init_device_properties_triton = lambda: None
    sys.modules["vllm_ascend.ops.triton.triton_utils"] = ascend_utils


def accuracy_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual, expected = actual.detach().cpu().float(), expected.detach().cpu().float()
    error = (actual - expected).abs()
    denominator = expected.abs().clamp_min(1e-6)
    cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    return {
        "max_abs_error": error.max().item(),
        "mean_abs_error": error.mean().item(),
        "max_rel_error": (error / denominator).max().item(),
        "mean_rel_error": (error / denominator).mean().item(),
        "cosine_similarity": cosine.item(),
    }


def to_cpu(value: Any) -> Any:
    return value.detach().cpu() if isinstance(value, torch.Tensor) else value


def generate_reference(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    if args.chunk_size != GDN_CHUNK_SIZE:
        raise ValueError(f"vLLM Ascend FLA uses a fixed chunk_size of {GDN_CHUNK_SIZE}")
    if args.vH < args.kH or args.vH % args.kH:
        raise ValueError("vH must be greater than or equal to kH and divisible by kH")

    if device.type == "cuda":
        from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule
    else:
        install_minimal_vllm_shim()
        install_lightweight_ascend_packages()
        import vllm_ascend.ops.triton.fla.chunk as ascend_chunk

        # The kernel only needs PCP collectives when world_size > 1. Keep this
        # standalone single-device generator independent of distributed setup.
        ascend_chunk.get_pcp_group = lambda: SimpleNamespace(world_size=1, rank_in_group=0)
        chunk_gated_delta_rule = ascend_chunk.chunk_gated_delta_rule

    dtype = parse_dtype(args.dtype)
    cu_seqlens = parse_cu_seqlens(args.cu_seqlens, args.B, args.T)
    tensors = generate_inputs(args.B, args.T, args.kH, args.vH, args.D, args.VDim, dtype, device)
    cu_tensor = None if cu_seqlens is None else torch.tensor(cu_seqlens, device=device, dtype=torch.long)
    scale = args.scale if args.scale is not None else args.D**-0.5

    fp32_o, fp32_final_state, fwd_o_v, fwd_o_h, fwd_o_g = recurrent_reference(
        **tensors, scale=scale, cu_seqlens=cu_seqlens
    )
    kernel_status = "executed"
    kernel_error = None
    try:
        synchronize(device)
        ref_o, final_state = chunk_gated_delta_rule(
            **tensors,
            scale=scale,
            initial_state=None,
            output_final_state=True,
            cu_seqlens=cu_tensor,
            head_first=False,
        )
        synchronize(device)
    except Exception as exc:  # Triton/CANN compiler errors are backend-specific.
        if not args.allow_fallback:
            raise
        kernel_status = "fallback_recurrent"
        kernel_error = f"{type(exc).__name__}: {exc}"
        ref_o = fp32_o.to(dtype)
        final_state = fp32_final_state.to(dtype)
        print("[Warning] FLA kernel failed; saving CPU FP32 recurrent fallback.")
        print(f"[Warning] {kernel_error.splitlines()[0]}")

    metrics = {
        "output": accuracy_metrics(ref_o, fp32_o),
        "final_state": accuracy_metrics(final_state, fp32_final_state),
    }
    config = {
        "batch": args.B,
        "seqlen": args.T,
        "key_heads": args.kH,
        "value_heads": args.vH,
        "key_dim": args.D,
        "value_dim": args.VDim,
        "chunk_size": args.chunk_size,
        "dtype": args.dtype,
        "scale": scale,
        "seed": args.seed,
        "device_type": device.type,
        "device": device_name(device),
        "kernel_status": kernel_status,
        "kernel_error": kernel_error,
    }
    suffix = f"_var_{len(cu_seqlens) - 1}" if cu_seqlens is not None else ""
    path = args.output_dir / (
        f"{args.B}_{args.kH}_{args.vH}_{args.T}_{args.D}_{args.VDim}_"
        f"{args.chunk_size}_{args.dtype}{suffix}.pt"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "q": to_cpu(tensors["q"]),
            "k": to_cpu(tensors["k"]),
            "v": fwd_o_v.to(dtype),
            "h": fwd_o_h.to(dtype),
            "g": fwd_o_g,
            "raw_v": to_cpu(tensors["v"]),
            "raw_g": to_cpu(tensors["g"]),
            "beta": to_cpu(tensors["beta"]),
            "scale": scale,
            "ref_o": to_cpu(ref_o),
            "final_state": to_cpu(final_state),
            "fp32_ref_o": to_cpu(fp32_o),
            "fp32_final_state": to_cpu(fp32_final_state),
            "cu_seqlens": to_cpu(cu_tensor),
            "accuracy": metrics,
            "config": config,
        },
        path,
    )
    print(f"[Info] Tensor data saved successfully to: {path}")
    result_label = f"{device.type.upper()} FLA" if kernel_status == "executed" else "recurrent fallback"
    print(f"[Accuracy] {result_label} vs CPU FP32 recurrent reference")
    for name, values in metrics.items():
        formatted = ", ".join(f"{key}={value:.6e}" for key, value in values.items())
        print(f"  {name}: {formatted}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a CUDA or Ascend FLA GDN accuracy reference file.")
    parser.add_argument("B", type=int)
    parser.add_argument("T", type=int)
    parser.add_argument("kH", type=int)
    parser.add_argument("vH", type=int)
    parser.add_argument("D", type=int)
    parser.add_argument("VDim", type=int)
    parser.add_argument("chunk_size", type=int)
    parser.add_argument("dtype", choices=["fp16", "bf16"])
    parser.add_argument("cu_seqlens", help="Cumulative sequence lengths as a Python list, or None")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--device", choices=["auto", "cuda", "npu"], default="auto")
    parser.add_argument(
        "--no-fallback",
        dest="allow_fallback",
        action="store_false",
        help="Propagate FLA/Triton compilation errors instead of saving recurrent fallback data",
    )
    parser.set_defaults(allow_fallback=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.B, args.T, args.kH, args.vH, args.D, args.VDim) <= 0:
        raise ValueError("All tensor dimensions must be positive")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    generate_reference(args)


if __name__ == "__main__":
    main()

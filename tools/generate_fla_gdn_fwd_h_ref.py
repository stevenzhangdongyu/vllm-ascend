#!/usr/bin/env python3
"""Generate fwd_h inputs and an independent FP32 reference on CUDA/NPU hosts."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_fla_gdn_ref import (  # noqa: E402
    accuracy_metrics,
    install_lightweight_ascend_packages,
    install_minimal_vllm_shim,
)

CHUNK_SIZE = 64


def dtype_of(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]


def cu_of(value: str, batch: int, seqlen: int) -> list[int] | None:
    if value.lower() in {"none", ""}:
        return None
    cu = [int(x) for x in ast.literal_eval(value)]
    if batch != 1 or cu[0] != 0 or cu[-1] != seqlen or any(b <= a for a, b in zip(cu, cu[1:])):
        raise ValueError("varlen cu_seqlens requires B=1, starts at 0, ends at T, and is increasing")
    return cu


def fwd_h_reference(k, w, u, g, initial_state, cu_seqlens, output_final_state):
    k, w, u, g = [x.detach().cpu().float() for x in (k, w, u, g)]
    bsz, seqlen, kh, dim = k.shape
    vh, vdim = u.shape[2:]
    spans = (
        [(b, 0, seqlen) for b in range(bsz)]
        if cu_seqlens is None
        else [(0, a, b) for a, b in zip(cu_seqlens, cu_seqlens[1:])]
    )
    state0 = None if initial_state is None else initial_state.detach().cpu().float()
    h_chunks, v_new = [], torch.empty_like(u)
    final = []
    for seq, start, end in spans:
        state = torch.zeros(vh, dim, vdim) if state0 is None else state0[len(final)].clone()
        ratio = vh // kh
        for chunk_start in range(start, end, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, end)
            h_chunks.append(state.clone())
            k_chunk = k[seq, chunk_start:chunk_end].repeat_interleave(ratio, dim=1)
            w_chunk = w[seq, chunk_start:chunk_end]
            u_chunk = u[seq, chunk_start:chunk_end]
            g_chunk = g[seq, chunk_start:chunk_end]
            delta = u_chunk - torch.einsum("thk,hkv->thv", w_chunk, state)
            v_new[seq, chunk_start:chunk_end] = delta
            g_last = g_chunk[-1]
            weighted_delta = delta * (g_last[None, :] - g_chunk).exp()[..., None]
            state = state * g_last.exp()[:, None, None]
            state = state + torch.einsum("thk,thv->hkv", k_chunk, weighted_delta)
        final.append(state)
    chunks_per_batch = (seqlen + CHUNK_SIZE - 1) // CHUNK_SIZE if cu_seqlens is None else None
    h = torch.stack(h_chunks).unsqueeze(0) if cu_seqlens is not None else torch.stack(h_chunks).reshape(bsz, chunks_per_batch, vh, dim, vdim)
    final_state = torch.stack(final) if output_final_state else None
    return h, v_new, final_state


def main():
    p = argparse.ArgumentParser(description="Generate GDN fwd_h reference data")
    p.add_argument("B", type=int); p.add_argument("T", type=int); p.add_argument("kH", type=int); p.add_argument("vH", type=int)
    p.add_argument("D", type=int); p.add_argument("VDim", type=int); p.add_argument("chunk_size", type=int)
    p.add_argument("dtype", choices=["fp16", "bf16"]); p.add_argument("cu_seqlens")
    p.add_argument("use_initial_state", type=int, choices=[0, 1]); p.add_argument("use_final_state", type=int, choices=[0, 1])
    p.add_argument("--output_dir", type=Path, default=Path("/tmp/gdn_fwd_h_ref")); p.add_argument("--seed", type=int, default=24)
    args = p.parse_args()
    if args.chunk_size != CHUNK_SIZE or args.vH < args.kH or args.vH % args.kH:
        raise ValueError("chunk_size must be 64 and vH must be divisible by kH")
    torch.manual_seed(args.seed)
    dtype, cu = dtype_of(args.dtype), cu_of(args.cu_seqlens, args.B, args.T)
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        raise RuntimeError("This generator requires an Ascend torch_npu environment") from None
    if not torch.npu.is_available():
        raise RuntimeError("No Ascend NPU is available")
    device = "npu"
    k = torch.randn(args.B, args.T, args.kH, args.D, device=device, dtype=dtype)
    w = torch.randn(args.B, args.T, args.vH, args.D, device=device, dtype=dtype)
    u = torch.randn(args.B, args.T, args.vH, args.VDim, device=device, dtype=dtype)
    g = torch.empty(args.B, args.T, args.vH, device=device, dtype=torch.float32).uniform_(-16, -0.1)
    g_cpu = g.cpu()
    spans = [(b, 0, args.T) for b in range(args.B)] if cu is None else [(0, a, b) for a, b in zip(cu, cu[1:])]
    for seq, start, end in spans:
        for chunk_start in range(start, end, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, end)
            g_cpu[seq, chunk_start:chunk_end] = g_cpu[seq, chunk_start:chunk_end].cumsum(0)
    g = g_cpu.to(device)
    nseq = args.B if cu is None else len(cu) - 1
    initial = torch.randn(nseq, args.vH, args.D, args.VDim, device=device, dtype=dtype) if args.use_initial_state else None
    fp32_h, fp32_v_new, fp32_final = fwd_h_reference(k, w, u, g, initial, cu, bool(args.use_final_state))
    install_minimal_vllm_shim()
    install_lightweight_ascend_packages()
    from vllm_ascend.ops.triton.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h

    cu_tensor = None if cu is None else torch.tensor(cu, device=device, dtype=torch.long)
    h, v_new, final = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial,
        output_final_state=bool(args.use_final_state),
        chunk_size=args.chunk_size,
        save_new_value=True,
        cu_seqlens=cu_tensor,
    )
    torch.npu.synchronize()
    accuracy = {
        "h": accuracy_metrics(h, fp32_h),
        "v_new": accuracy_metrics(v_new, fp32_v_new),
    }
    if final is not None:
        accuracy["final_state"] = accuracy_metrics(final, fp32_final)
    suffix = f"_var_{len(cu)-1}" if cu else ""
    path = args.output_dir / f"{args.B}_{args.kH}_{args.vH}_{args.T}_{args.D}_{args.VDim}_{args.chunk_size}_{args.dtype}{suffix}.pt"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output_dir"] = str(args.output_dir)
    torch.save(
        {
            "k": k.cpu(),
            "w": w.cpu(),
            "u": u.cpu(),
            "g": g.cpu(),
            "initial_state": None if initial is None else initial.cpu(),
            "h": h.cpu(),
            "v_new": v_new.cpu(),
            "final_state": None if final is None else final.cpu(),
            "fp32_h": fp32_h,
            "fp32_v_new": fp32_v_new,
            "fp32_final_state": fp32_final,
            "cu_seqlens": None if cu is None else torch.tensor(cu),
            "accuracy": accuracy,
            "config": config,
        },
        path,
    )
    print(f"[Info] Tensor data saved successfully to: {path}")
    for name, metrics in accuracy.items():
        print(f"[Accuracy] {name}: " + ", ".join(f"{key}={value:.6e}" for key, value in metrics.items()))


if __name__ == "__main__":
    main()

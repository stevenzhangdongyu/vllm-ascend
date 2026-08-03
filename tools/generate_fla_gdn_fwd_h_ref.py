#!/usr/bin/env python3
"""Generate fwd_h inputs and an independent FP32 reference on CUDA/NPU hosts."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import torch

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
    spans = [(b, 0, seqlen) for b in range(bsz)] if cu_seqlens is None else [(0, a, b) for a, b in zip(cu_seqlens, cu_seqlens[1:])]
    state0 = None if initial_state is None else initial_state.detach().cpu().float()
    h_chunks, v_new = [], torch.empty_like(u)
    final = []
    for seq, start, end in spans:
        state = torch.zeros(vh, dim, vdim) if state0 is None else state0[len(final)].clone()
        ratio = vh // kh
        for token in range(start, end):
            local = token - start
            if local % CHUNK_SIZE == 0:
                h_chunks.append(state.clone())
            prediction = torch.einsum("hk,hkv->hv", w[seq, token], state)
            delta = u[seq, token] - prediction
            v_new[seq, token] = delta
            chunk_end = min(start + ((local // CHUNK_SIZE) + 1) * CHUNK_SIZE, end) - 1
            delta = delta * (g[seq, chunk_end] - g[seq, token]).exp()[:, None]
            state.mul_(g[seq, chunk_end].exp()[:, None, None])
            state.add_(torch.einsum("hk,hv->hkv", k[seq, token].repeat_interleave(ratio, 0), delta))
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
    device = "cuda" if torch.cuda.is_available() else "npu"
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
    h, v_new, final = fwd_h_reference(k, w, u, g, initial, cu, bool(args.use_final_state))
    suffix = f"_var_{len(cu)-1}" if cu else ""
    path = args.output_dir / f"{args.B}_{args.kH}_{args.vH}_{args.T}_{args.D}_{args.VDim}_{args.chunk_size}_{args.dtype}{suffix}.pt"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"k": k.cpu(), "w": w.cpu(), "u": u.cpu(), "g": g.cpu(), "initial_state": None if initial is None else initial.cpu(), "h": h.to(dtype), "v_new": v_new.to(dtype), "final_state": None if final is None else final.to(dtype), "cu_seqlens": None if cu is None else torch.tensor(cu), "config": vars(args)}, path)
    print(f"[Info] Tensor data saved successfully to: {path}")


if __name__ == "__main__":
    main()

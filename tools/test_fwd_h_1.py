#!/usr/bin/env python3
"""Run one GDN fwd_h Triton case and optionally check a saved reference."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_fla_gdn_ref import (  # noqa: E402
    install_lightweight_ascend_packages,
    install_minimal_vllm_shim,
)


def parse_dtype(name: str) -> torch.dtype:
    aliases = {
        "half": torch.float16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "float": torch.float32,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError:
        raise ValueError(f"unsupported dtype: {name}") from None


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    entries = []
    for sequence_id, length in enumerate(lengths.tolist()):
        entries.extend((sequence_id, chunk_id) for chunk_id in range(math.ceil(length / chunk_size)))
    return torch.tensor(entries, dtype=torch.long, device=cu_seqlens.device)


def validate_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} shape must be {expected}, got {tuple(tensor.shape)}")


def get_reference_output(data: dict, name: str) -> torch.Tensor:
    for key in (name, f"ref_{name}"):
        if key in data:
            return data[key]
    raise KeyError(f"reference file is missing output field {name!r} (or ref_{name!s})")


def load_reference(args: argparse.Namespace, device: torch.device):
    if not args.data_path.is_file():
        raise FileNotFoundError(f"reference file not found: {args.data_path}")
    data = torch.load(args.data_path, map_location="cpu", weights_only=True)
    required = ("k", "w", "u", "g")
    missing = [name for name in required if name not in data]
    if missing:
        raise KeyError(f"reference file is missing fields: {missing}")

    k = data["k"].to(device=device, dtype=args.dtype).contiguous()
    w = data["w"].to(device=device, dtype=args.dtype).contiguous()
    u = data["u"].to(device=device, dtype=args.dtype).contiguous()
    g = data["g"].to(device=device, dtype=args.g_dtype).contiguous()
    initial_state = data.get("initial_state") if args.use_initial_state else None
    if initial_state is not None:
        initial_state = initial_state.to(device=device, dtype=args.state_dtype).contiguous()
    cu_seqlens = data.get("cu_seqlens") if args.is_varied_len else None
    stored_cu_seqlens = data.get("cu_seqlens")
    if args.is_varied_len == 0 and stored_cu_seqlens is not None:
        raise ValueError(
            "case is marked fixed-length (is_varied_len=0), but the reference contains cu_seqlens; "
            "rerun with is_varied_len=1"
        )
    if args.is_varied_len == 1 and stored_cu_seqlens is None:
        raise ValueError(
            "case is marked variable-length (is_varied_len=1), but the reference has no cu_seqlens"
        )
    if args.is_varied_len and stored_cu_seqlens is not None:
        stored_token_batch = int(stored_cu_seqlens.numel() - 1)
        if stored_token_batch != args.token_batch:
            raise ValueError(
                f"token_batch mismatch: command token_batch={args.token_batch}, "
                f"reference contains {stored_token_batch} sequences from cu_seqlens; "
                "pass the logical sequence count as token_batch"
            )
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(device=device, dtype=torch.long).unique().contiguous()
    return data, k, w, u, g, initial_state, cu_seqlens


def generate_inputs(args: argparse.Namespace, device: torch.device):
    shape_batch = args.batch
    k = torch.normal(
        0.0109,
        0.0979,
        (shape_batch, args.seqlen, args.k_heads, args.k_dim),
        device=device,
        dtype=args.dtype,
    )
    w = torch.normal(
        0.0109,
        0.0979,
        (shape_batch, args.seqlen, args.v_heads, args.k_dim),
        device=device,
        dtype=args.dtype,
    )
    u = torch.normal(
        0.0168,
        0.1328,
        (shape_batch, args.seqlen, args.v_heads, args.v_dim),
        device=device,
        dtype=args.dtype,
    )
    raw_g = torch.empty(
        shape_batch,
        args.seqlen,
        args.v_heads,
        device="cpu",
        dtype=torch.float32,
    ).uniform_(-0.08, -0.002)
    cu_seqlens = None
    if args.is_varied_len:
        lengths = [args.seqlen // args.token_batch] * args.token_batch
        lengths[-1] += args.seqlen - sum(lengths)
        cu_values = [0]
        for length in lengths:
            cu_values.append(cu_values[-1] + length)
        cu_seqlens = torch.tensor(cu_values, device=device, dtype=torch.long)
        spans = [(0, start, end) for start, end in zip(cu_values, cu_values[1:])]
    else:
        spans = [(batch_id, 0, args.seqlen) for batch_id in range(args.batch)]
    for batch_id, start, end in spans:
        for chunk_start in range(start, end, args.chunk_size):
            chunk_end = min(chunk_start + args.chunk_size, end)
            raw_g[batch_id, chunk_start:chunk_end] = raw_g[batch_id, chunk_start:chunk_end].cumsum(0)
    g = raw_g.to(device=device, dtype=args.g_dtype)
    sequence_count = args.batch * args.token_batch
    initial_state = None
    if args.use_initial_state:
        initial_state = torch.normal(
            0.0,
            0.02,
            (sequence_count, args.v_heads, args.k_dim, args.v_dim),
            device=device,
            dtype=args.state_dtype,
        )
    return None, k, w, u, g, initial_state, cu_seqlens


def compare_tensor(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, object]:
    expected = expected.detach().cpu().float()
    actual = actual.detach().cpu().float()
    if actual.shape != expected.shape:
        raise ValueError(
            f"{name} shape mismatch: actual={tuple(actual.shape)}, "
            f"reference={tuple(expected.shape)}; check is_varied_len, batch, "
            "seqlen, and chunk_size"
        )
    absolute_error = (actual - expected).abs()
    denominator_floor = (1.0 / (1 << 14)) / args.diff_threshold
    denominator = torch.maximum(actual.abs(), expected.abs()).clamp_min(denominator_floor) + 1e-9
    metric_error = torch.where(
        absolute_error < args.diff_threshold,
        absolute_error,
        absolute_error / denominator,
    )
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    passed = (metric_error <= args.diff_threshold) & finite
    total = actual.numel()
    passed_count = int(passed.sum())
    pass_percent = passed_count / total * 100.0
    max_metric_error = float(metric_error.nan_to_num(nan=math.inf, posinf=math.inf).max())
    failure_ratio = 1.0 - passed_count / total
    success = failure_ratio <= args.failure_ratio and max_metric_error < args.max_error
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    print(
        f"[Accuracy] {name}: {'PASS' if success else 'FAIL'}, total={total}, "
        f"passed={pass_percent:.6f}%, max_error={max_metric_error:.6e}, "
        f"max_abs={absolute_error.max().item():.6e}, "
        f"mean_abs={absolute_error.mean().item():.6e}, cosine={cosine.item():.8f}"
    )
    failed_indices = (~passed).nonzero(as_tuple=False)[: args.max_failures]
    for index in failed_indices:
        coordinates = tuple(int(value) for value in index)
        print(
            f"[Mismatch] {name}{coordinates}: expected={expected[coordinates].item():.8e}, "
            f"actual={actual[coordinates].item():.8e}, "
            f"abs={absolute_error[coordinates].item():.8e}, "
            f"error={metric_error[coordinates].item():.8e}"
        )
    return {
        "name": name,
        "passed": success,
        "shape": list(actual.shape),
        "total": total,
        "passed_count": passed_count,
        "pass_percent": pass_percent,
        "max_error": max_metric_error,
        "max_abs_error": float(absolute_error.max()),
        "mean_abs_error": float(absolute_error.mean()),
        "cosine_similarity": float(cosine),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=int)
    parser.add_argument("seqlen", type=int)
    parser.add_argument("k_heads", type=int)
    parser.add_argument("v_heads", type=int)
    parser.add_argument("k_dim", type=int)
    parser.add_argument("v_dim", type=int)
    parser.add_argument("is_varied_len", type=int, choices=[0, 1])
    parser.add_argument("token_batch", type=int)
    parser.add_argument("chunk_size", type=int)
    parser.add_argument("use_initial_state", type=int, choices=[0, 1])
    parser.add_argument("store_final_state", type=int, choices=[0, 1])
    parser.add_argument("dtype", type=parse_dtype)
    parser.add_argument("use_actual_input", type=int, choices=[0, 1])
    parser.add_argument("use_actual_output", type=int, choices=[0, 1])
    parser.add_argument("data_path", type=Path)
    parser.add_argument("device", type=int)
    parser.add_argument("g_dtype", type=parse_dtype)
    parser.add_argument("state_dtype", type=parse_dtype)
    parser.add_argument("--diff-threshold", type=float, default=0.001)
    parser.add_argument("--failure-ratio", type=float, default=0.0001)
    parser.add_argument("--max-error", type=float, default=0.1)
    parser.add_argument("--max-failures", type=int, default=20)
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.batch, args.token_batch, args.seqlen, args.k_heads, args.v_heads, args.k_dim, args.v_dim,
           args.chunk_size) <= 0:
        raise ValueError("all dimensions and chunk_size must be positive")
    if not args.is_varied_len and args.token_batch != 1:
        raise ValueError("fixed-length mode requires token_batch=1")
    if args.is_varied_len and args.batch != 1:
        raise ValueError("variable-length flattened input requires batch=1")
    if not 0 <= args.failure_ratio < 1 or args.diff_threshold <= 0 or args.max_error <= 0:
        raise ValueError("comparison thresholds must be positive and failure_ratio must be less than 1")
    if args.max_failures < 0:
        raise ValueError("max_failures must be non-negative")
    if args.v_heads < args.k_heads or args.v_heads % args.k_heads:
        raise ValueError("v_heads must be divisible by k_heads")
    if args.use_actual_output and not args.use_actual_input:
        raise ValueError("use_actual_output=1 requires use_actual_input=1")

    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    torch.manual_seed(1)

    if args.use_actual_input:
        data, k, w, u, g, initial_state, cu_seqlens = load_reference(args, device)
    else:
        data, k, w, u, g, initial_state, cu_seqlens = generate_inputs(args, device)

    shape_batch = args.batch
    validate_shape("k", k, (shape_batch, args.seqlen, args.k_heads, args.k_dim))
    validate_shape("w", w, (shape_batch, args.seqlen, args.v_heads, args.k_dim))
    validate_shape("u", u, (shape_batch, args.seqlen, args.v_heads, args.v_dim))
    validate_shape("g", g, (shape_batch, args.seqlen, args.v_heads))
    if args.is_varied_len and cu_seqlens is None:
        raise ValueError("varlen case requires cu_seqlens in the reference file")
    if not args.is_varied_len and cu_seqlens is not None:
        raise ValueError("fixed-length case must not provide cu_seqlens")

    install_minimal_vllm_shim()
    install_lightweight_ascend_packages()
    from vllm_ascend.ops.triton.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h

    chunk_indices = None
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, args.chunk_size)

    torch.npu.synchronize()
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=bool(args.store_final_state),
        chunk_size=args.chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    torch.npu.synchronize()

    print(f"[Result] h={tuple(h.shape)}, v_new={tuple(v_new.shape)}")
    if final_state is not None:
        print(f"[Result] final_state={tuple(final_state.shape)}")

    if args.use_actual_output:
        if data is None:
            raise RuntimeError("reference output is unavailable")
        reports = [
            compare_tensor("h", h, get_reference_output(data, "h"), args),
            compare_tensor("v_new", v_new, get_reference_output(data, "v_new"), args),
        ]
        expected_final = data.get("final_state")
        if args.store_final_state:
            if final_state is None or expected_final is None:
                raise ValueError("final_state is required by store_final_state=1")
            reports.append(compare_tensor("final_state", final_state, expected_final, args))
        overall_passed = all(bool(report["passed"]) for report in reports)
        report = {
            "passed": overall_passed,
            "thresholds": {
                "diff_threshold": args.diff_threshold,
                "failure_ratio": args.failure_ratio,
                "max_error": args.max_error,
            },
            "tensors": reports,
        }
        if args.report_json:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"[Report] {args.report_json}")
        if not overall_passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

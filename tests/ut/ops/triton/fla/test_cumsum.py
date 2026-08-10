# SPDX-License-Identifier: Apache-2.0

import sys
import types

import pytest


@pytest.fixture
def cumsum_module(monkeypatch):
    triton_utils = types.ModuleType("vllm.triton_utils")
    triton_utils.tl = pytest.importorskip("triton.language")
    triton_utils.triton = pytest.importorskip("triton")
    monkeypatch.setitem(sys.modules, "vllm.triton_utils", triton_utils)

    from vllm_ascend.ops.triton.fla import cumsum

    return cumsum


@pytest.mark.parametrize(
    ("head_count", "chunk_size", "expected"),
    [(32, 128, 128), (48, 128, 128), (4, 128, 512), (32, 64, 128)],
)
def test_cumsum_block_size(cumsum_module, head_count, chunk_size, expected):
    block_size = cumsum_module.get_cumsum_block_size(head_count, chunk_size)

    assert block_size == expected
    assert block_size >= chunk_size
    assert block_size % chunk_size == 0

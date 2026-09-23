"""Experiment A must size the host pool with ``--hicache-ratio``, never with a
fractional ``--hicache-size``.

Regression guard for a bug that cost a full pod run. SGLang declares
``hicache_size`` as ``int`` (``arg_groups/fields/memory.py``:
``hicache_size: A[int, ...]``, "in gigabytes"), so ``--hicache-size 8.000078``
is rejected by argparse and the server exits with **code 2** before it loads the
model. Equal *token* capacity needs sub-GB precision (BF16 needs 8.000078 GB for
54,254 tokens), so the experiment was unrunnable until it moved to the ratio
knob, which is a float *token* ratio.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_experiment as rx  # noqa: E402

DEVICE_CAPACITY = 16384
TARGET = 54254
PAGE_SIZE = 1


def _configs(**overrides):
    kwargs = dict(
        model="Qwen/Qwen3-8B",
        host_size_gb=8.0,
        tp=1,
        page_size=PAGE_SIZE,
        max_total_tokens=DEVICE_CAPACITY,
        sizing="equal-tokens",
        target_l2_tokens=TARGET,
    )
    kwargs.update(overrides)
    return rx.build_configs(**kwargs)


def _ratio(configs, name: str) -> float:
    args = configs[name].server_args
    return float(args[args.index("--hicache-ratio") + 1])


@pytest.mark.parametrize("config", ["bf16", "int8"])
def test_equal_tokens_never_passes_a_fractional_hicache_size(config):
    """The exact value that made the server exit 2: 8.000078 is not an int."""
    args = _configs()[config].server_args
    assert "--hicache-size" not in args, (
        "--hicache-size is an int in GB; a fractional value is rejected by "
        "argparse and kills the server with exit code 2"
    )


@pytest.mark.parametrize("config", ["bf16", "int8"])
def test_equal_tokens_lands_exactly_on_the_target_token_count(config):
    """HostKVCache: size = int(device_capacity * ratio), then rounded up a page
    (page_num = size // page_size + 1). With page_size 1 that is +1 token."""
    ratio = _ratio(_configs(), config)
    assert int(DEVICE_CAPACITY * ratio) + 1 == TARGET


def test_equal_tokens_gives_both_tiers_the_same_token_capacity():
    configs = _configs()
    assert _ratio(configs, "bf16") == _ratio(configs, "int8"), (
        "equal LOGICAL capacity means one token ratio shared by both configs; "
        "INT8 then needs ~56% of the bytes for the same token count"
    )


def test_equal_tokens_refuses_an_unpinned_device_pool():
    """The ratio is relative to the device pool, so that pool must be known."""
    with pytest.raises(ValueError, match="max-total-tokens"):
        _configs(max_total_tokens=None)


def test_fixed_size_still_uses_the_integer_size_knob():
    """Experiment B is unchanged: 8.0 GB formats as the integer "8"."""
    args = _configs(sizing="fixed-size")["bf16"].server_args
    assert args[args.index("--hicache-size") + 1] == "8"
    assert "--hicache-ratio" not in args

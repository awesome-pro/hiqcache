"""Regression tests for the quality comparison metrics.

The subtlety these pin: once two runs produce different tokens, position k no
longer refers to the same token in both, so a logprob difference there measures
the divergence rather than the codec. An earlier version compared positionally
across the whole sequence and reported a max delta of 1.8 that was entirely an
artefact of comparing token A against token B.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from quality_compare import (  # noqa: E402
    build_prompt,
    compare,
    extract_chosen_logprobs,
    extract_output_ids,
)


def _gen(tokens, logprobs, text="x"):
    return {"token_ids": list(tokens), "logprobs": list(logprobs), "text": text}


def test_identical_runs_agree_perfectly():
    same = [_gen([1, 2, 3], [-0.1, -0.2, -0.3])] * 3
    r = compare(same, same)
    assert r["first_token_agreement"] == 1.0
    assert r["full_sequence_agreement"] == 1.0
    assert r["token_level_agreement"] == 1.0
    assert r["mean_abs_delta"] == 0.0
    assert r["max_abs_delta"] == 0.0


def test_logprob_delta_excludes_the_divergent_suffix():
    """The core property: only the agreed prefix contributes to the delta."""
    a = [_gen([1, 2, 3], [-0.1, -0.2, -0.3])]
    b = [_gen([1, 2, 9], [-0.1, -0.2, -5.0])]
    r = compare(a, b)
    assert r["max_abs_delta"] == 0.0, (
        "the differing position must not contribute; it compares different tokens"
    )
    assert r["full_sequence_agreement"] == 0.0
    assert r["first_token_agreement"] == 1.0
    assert r["token_level_agreement"] == pytest.approx(2 / 3)
    assert r["mean_agreed_prefix_tokens"] == pytest.approx(2.0)


def test_logprob_delta_is_measured_when_tokens_match():
    c = [_gen([1, 2, 3], [-1.0, -1.0, -1.0])]
    d = [_gen([1, 2, 3], [-1.5, -1.0, -0.5])]
    r = compare(c, d)
    assert r["max_abs_delta"] == pytest.approx(0.5)
    assert r["mean_abs_delta"] == pytest.approx(1.0 / 3.0)
    assert r["full_sequence_agreement"] == 1.0


def test_length_difference_is_reported_not_crashed():
    a = [_gen([1, 2, 3], [-1.0, -1.0, -1.0])]
    b = [_gen([1, 2, 3, 4, 5], [-1.0, -1.0, -1.0, -1.0, -1.0])]
    r = compare(a, b)
    assert r["mean_length_delta"] == pytest.approx(2.0)
    assert r["max_abs_delta"] == 0.0
    assert r["token_level_agreement"] == pytest.approx(3 / 5)


def test_missing_logprobs_are_tolerated():
    """Servers without logprob support must not break the comparison."""
    a = [{"token_ids": [1, 2], "logprobs": [], "text": "a"}]
    b = [{"token_ids": [1, 2], "logprobs": [], "text": "a"}]
    r = compare(a, b)
    assert "mean_abs_delta" not in r
    assert r["full_sequence_agreement"] == 1.0


def test_no_prompts_does_not_divide_by_zero():
    r = compare([], [])
    assert r["prompts"] == 0
    assert r["first_token_agreement"] == 0.0


def test_extract_chosen_logprobs_reads_sglang_shape():
    payload = {
        "meta_info": {
            "input_token_logprobs": [
                [-0.5, 10, "a"],
                [-1.5, 11, "b"],
                [-2.5, 12, "c"],
            ]
        }
    }
    assert extract_chosen_logprobs(payload) == [-0.5, -1.5, -2.5]
    assert extract_chosen_logprobs({}) == []
    assert extract_chosen_logprobs({"meta_info": {"input_token_logprobs": None}}) == []


def test_prompts_are_deterministic_and_identical_across_configs():
    """Both configs must receive byte-identical prompts."""
    a = build_prompt("general knowledge", "What is 2+2?")
    b = build_prompt("general knowledge", "What is 2+2?")
    assert a == b
    assert "What is 2+2?" in a


# ------------------------------------------------------------- capture guards
# The first real run of this harness captured no tokens at all and still printed
# "100% full-sequence agreement": [] == [] is a match, and there is no first
# token to disagree on. These pin the two defences against that reading as a
# pass.


def test_extract_output_ids_prefers_the_explicit_fields():
    assert extract_output_ids({"output_ids": [1, 2, 3]}) == [1, 2, 3]
    assert extract_output_ids({"meta_info": {"output_ids": [4, 5]}}) == [4, 5]


def test_extract_output_ids_falls_back_to_chosen_logprobs():
    """This build may omit output_ids; the ids ride along as the middle element
    of the [logprob, token_id, text] triples."""
    payload = {
        "meta_info": {"output_token_logprobs": [[-0.5, 11, "a"], [-0.25, 22, "b"]]}
    }
    assert extract_output_ids(payload) == [11, 22]


def test_extract_output_ids_is_empty_when_the_build_reports_neither():
    assert extract_output_ids({}) == []
    assert extract_output_ids({"meta_info": {"output_ids": None}}) == []


def test_empty_capture_is_visibly_undefined_not_a_perfect_match():
    """Reproduces the exact numbers that made a vacuous run look like a pass."""
    a = [_gen([], []) for _ in range(5)]
    b = [_gen([], []) for _ in range(5)]
    r = compare(a, b)
    assert r["tokens_compared"] == 0, "the validity guard keys off this"
    # The two misleading figures the guard exists to suppress.
    assert r["full_sequence_agreement"] == 1.0
    assert r["first_token_agreement"] == 0.0


def test_real_capture_gives_the_guard_a_nonzero_denominator():
    a = [_gen([1, 2, 3], [])]
    b = [_gen([1, 2, 3], [])]
    assert compare(a, b)["tokens_compared"] == 3

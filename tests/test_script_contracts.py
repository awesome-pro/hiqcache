"""Static checks on the pod-facing scripts.

These exist because three separate script defects reached the pod and each cost a
round trip, while every one of them was detectable on the Mac in milliseconds:

  * a relative path that did not resolve from the clone directory
  * ``grep -oP``, a GNU extension BSD grep rejects
  * a duplicated ``add_argument`` that made argparse exit before doing anything

The common thread is that the scripts are the part of this repo that cannot be
exercised here, so they get less scrutiny than the codec -- which is backwards,
because a broken script wastes GPU time and a broken codec gets caught by tests.

These checks are deliberately cheap and static: parse each script's AST and assert
the properties that have actually been violated. They are not a substitute for
running the scripts, but they catch the recurring mechanical mistakes.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((REPO_ROOT / "scripts").glob("*.py"))


def _script(name: str) -> Path:
    path = REPO_ROOT / "scripts" / name
    if not path.is_file():
        pytest.skip(f"{name} not present")
    return path


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_no_duplicate_argparse_options(path: Path):
    """Two add_argument calls with the same option string make argparse exit.

    argparse raises ArgumentError at parser construction time, so the script dies
    with a traceback before doing anything at all -- which is what a duplicated
    --max-total-tokens did on the pod.
    """
    tree = ast.parse(path.read_text())
    seen: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.startswith("-"):
                    seen[arg.value] = seen.get(arg.value, 0) + 1
    duplicates = {opt: n for opt, n in seen.items() if n > 1}
    assert not duplicates, (
        f"{path.name} defines these options more than once: {duplicates}. "
        f"argparse will raise ArgumentError before the script runs."
    )


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_scripts_parse_and_expose_main(path: Path):
    tree = ast.parse(path.read_text())
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "main" in names, f"{path.name} has no main()"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_no_gnu_only_grep_flags(path: Path):
    """``grep -oP`` is a GNU extension; BSD grep (macOS) rejects it.

    Scripts in this repo run on both, so a GNU-only flag is a portability bug.
    """
    source = path.read_text()
    offenders = re.findall(r"grep\s+-[a-zA-Z]*P", source)
    assert not offenders, (
        f"{path.name} uses a GNU-only grep flag ({offenders}); use sed or a "
        f"Python helper instead"
    )


def test_bootstrap_resolves_the_probe_by_absolute_path():
    """A relative path breaks after the script cds into the clone directory.

    That is how GATE 0 failed with "can't open file '/workspace/scripts/
    env_probe.py'": pod_bootstrap.sh cds to $WORKSPACE, but the probe lives one
    level down inside the hiqcache checkout.
    """
    source = _script("pod_bootstrap.sh").read_text()
    assert 'PROBE="$WORKSPACE/hiqcache/scripts/env_probe.py"' in source, (
        "the probe must be resolved against $WORKSPACE, not the cwd"
    )
    assert 'python "$PROBE"' in source, "the probe must be invoked via $PROBE"


def test_bootstrap_uses_https_clone_urls():
    """SSH clone URLs need a key that a fresh pod does not have.

    Public repositories are irrelevant here: ``git@github.com:`` always requires
    a private key, and pod_bootstrap.sh runs before any key could exist.

    Only *clone commands* are inspected. The script deliberately mentions the SSH
    form in a comment explaining why it is not used, and a plain substring search
    would flag its own explanation -- the same trap as the RNG guard in
    test_conformance_harness.
    """
    source = _script("pod_bootstrap.sh").read_text()
    clone_lines = [
        line.strip()
        for line in source.splitlines()
        if re.search(r"\bgit\s+clone\b", line) and not line.strip().startswith("#")
    ]
    ssh_clones = [line for line in clone_lines if "git@" in line]
    assert not ssh_clones, (
        f"pod_bootstrap.sh clones over SSH: {ssh_clones}. Use the HTTPS defaults; "
        f"a fresh pod has no key."
    )
    assert any("$HICACHE_URL" in line for line in clone_lines), (
        "the hiqcache clone must use the configurable HTTPS URL"
    )
    assert any("$SGLANG_URL" in line for line in clone_lines), (
        "the sglang clone must use the configurable HTTPS URL"
    )
    # And the defaults must actually be HTTPS.
    assert 'HICACHE_URL="${HICACHE_URL:-https://github.com/awesome-pro/hiqcache.git}"' in source
    assert 'SGLANG_URL="${SGLANG_URL:-https://github.com/awesome-pro/sglang.git}"' in source


def test_bootstrap_does_not_require_write_back_jit():
    """The pool rejects the page_first path, so that kernel is never called."""
    source = _script("pod_bootstrap.sh").read_text()
    assert "SGLANG_BUILD_RUST_EXTS=none" in source, (
        "the Rust extensions are optional and need cargo, which the image lacks"
    )


@pytest.mark.parametrize("name", ["smoke_test.py", "run_experiment.py"])
def test_wait_for_server_reports_progress(name: str):
    """A silent 16 GB model download is indistinguishable from a hang."""
    source = _script(name).read_text()
    assert "_hiqcache_log_path" in source, (
        f"{name} must record the server log path so the startup wait can report "
        f"progress instead of appearing to hang"
    )


def test_hicache_configs_use_write_back():
    """write_back is required for L2 to be reachable at all.

    Under write_through / write_through_selective, SGLang sets
    ``is_write_back = (hicache_write_policy == "write_back")`` to False, and
    ``UnifiedTreeCore.evict_device_leaf`` then DELETES an unbacked device leaf
    from the tree instead of demoting it to a host-only node. L2 fills with KV
    that no tree node references, so host_hit_length stays 0, load-back never
    fires, and every revisit re-prefills.

    Measured on the pod: 21.8 GB backed up and 0 tokens ever restored, with the
    smoke test reporting "the prefix was gone from L1 AND was not restored from
    L2". Nothing errors, so this is a silent misconfiguration -- worth a test.
    """
    script = _script("run_experiment.py")
    tree = ast.parse(script.read_text())
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.List):
            continue
        values = [e.value for e in node.elts if isinstance(e, ast.Constant)]
        if "--hicache-write-policy" in values:
            found = True
            idx = values.index("--hicache-write-policy")
            policy = values[idx + 1] if idx + 1 < len(values) else None
            assert policy == "write_back", (
                f"hicache write policy is {policy!r}; L2 entries written under "
                f"write_through are unreachable because the tree node is deleted "
                f"on eviction rather than demoted"
            )
    assert found, "no --hicache-write-policy found; did the config change shape?"


def test_hicache_configs_pin_the_pool_our_analysis_assumes():
    """The codec only supports layer_first + kernel + page_size 1 at TP=1.

    pod_bootstrap and the pool both reject anything else at startup, but a wrong
    flag here would surface as a failed server launch rather than a failed test.
    """
    tree = ast.parse(_script("run_experiment.py").read_text())
    flags: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.List):
            values = [e.value for e in node.elts if isinstance(e, ast.Constant)]
            for i, value in enumerate(values):
                if isinstance(value, str) and value.startswith("--") and i + 1 < len(values):
                    nxt = values[i + 1]
                    if isinstance(nxt, str) and not nxt.startswith("--"):
                        flags[value] = nxt
    assert flags.get("--hicache-mem-layout") == "layer_first"
    assert flags.get("--hicache-io-backend") == "kernel"

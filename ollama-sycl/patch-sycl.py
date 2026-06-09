#!/usr/bin/env python3
"""
Patch upstream llama.cpp ggml-sycl to match Ollama's modified ggml backend ABI.

Ollama vendors its own copy of ggml under ml/backend/ggml/ggml/ but its
.rsync-filter excludes ggml-sycl entirely, so this Dockerfile injects a
freshly-compiled libggml-sycl.so built from the upstream tree.

The two ABIs (Ollama's vendored ggml-backend-impl.h vs upstream's) drift
in a few well-known places that we must patch in the upstream source
before compiling, otherwise either the build fails (struct field name
mismatches) or the resulting .so is silently misaligned (vtable offsets).

Known drift points handled by this script:

1. graph_compute signature
   Ollama:   ggml_status (*graph_compute)(ggml_backend_t, ggml_cgraph *, int batch_size)
   Upstream: ggml_status (*graph_compute)(ggml_backend_t, ggml_cgraph *)

2. Backend (stream) vtable: set_tensor_2d_async / get_tensor_2d_async
   Upstream adds these between get_tensor_async and cpy_tensor_async.
   Ollama's ggml_backend_i has no such fields; if left in the designated
   initializer the C++ build fails with "no member named 'set_tensor_2d_async'".

3. Buffer vtable: set_tensor_2d / get_tensor_2d
   Same story for ggml_backend_buffer_i. Upstream defines these between
   get_tensor and cpy_tensor; Ollama doesn't.

The script reads Ollama's actual ggml-backend-impl.h to detect which patches
are needed at runtime (so a future Ollama version that converges on upstream
will simply skip the patches), and fails loudly if it can't apply a patch
that the ABI clearly requires.

Usage:
    patch-sycl.py <ggml-sycl.cpp>
    patch-sycl.py <ggml-sycl.cpp> --abi-header <ggml-backend-impl.h>

If --abi-header is omitted, defaults to:
    ml/backend/ggml/ggml/src/ggml-backend-impl.h (relative to cwd)

Exit codes:
    0  success (patches applied, or no-op because APIs already match)
    2  abi/source mismatch — patcher refuses to write a broken file
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


DEFAULT_ABI_HEADER = Path("ml/backend/ggml/ggml/src/ggml-backend-impl.h")


class PatchError(RuntimeError):
    """Raised when a required patch cannot be applied safely."""


@dataclass
class BackendABI:
    """Subset of the Ollama ggml backend ABI we care about."""

    graph_compute_has_batch_size: bool
    has_set_tensor_2d_async: bool
    has_get_tensor_2d_async: bool
    has_buffer_set_tensor_2d: bool
    has_buffer_get_tensor_2d: bool


@dataclass
class PatchResult:
    changed: bool
    applied: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ABI parsing
# ---------------------------------------------------------------------------


def parse_backend_abi(header_path: Path) -> BackendABI:
    """Inspect Ollama's ggml-backend-impl.h to learn the current vtable shapes."""
    if not header_path.exists():
        raise FileNotFoundError(f"ABI header not found: {header_path}")

    src = header_path.read_text()

    backend_i = _extract_struct(src, "ggml_backend_i")
    buffer_i = _extract_struct(src, "ggml_backend_buffer_i")

    graph_compute_re = re.compile(
        r"\(\s*\*\s*graph_compute\s*\)\s*\([^)]*?\bint\s+batch_size\b[^)]*\)",
    )

    return BackendABI(
        graph_compute_has_batch_size=bool(graph_compute_re.search(backend_i)),
        has_set_tensor_2d_async="set_tensor_2d_async" in backend_i,
        has_get_tensor_2d_async="get_tensor_2d_async" in backend_i,
        has_buffer_set_tensor_2d=_buffer_has_2d_field(buffer_i, "set_tensor_2d"),
        has_buffer_get_tensor_2d=_buffer_has_2d_field(buffer_i, "get_tensor_2d"),
    )


def _extract_struct(src: str, name: str) -> str:
    """Return the body of `struct <name> { ... };` (greedy-safely matched).

    If the struct cannot be located, returns "" so callers see "field absent"
    rather than crashing — better to fail later in patch_file with a clear
    "ABI requires X but source lacks Y" message.
    """
    pattern = re.compile(rf"struct\s+{re.escape(name)}\s*\{{(.*?)\n\s*\}};", re.DOTALL)
    m = pattern.search(src)
    return m.group(1) if m else ""


def _buffer_has_2d_field(buffer_struct_body: str, fname: str) -> bool:
    """Distinguish set_tensor_2d (the field) from set_tensor_2d_async (different)."""
    return bool(re.search(rf"\(\s*\*\s*{re.escape(fname)}\s*\)", buffer_struct_body))


# ---------------------------------------------------------------------------
# Patch rules
# ---------------------------------------------------------------------------


@dataclass
class Rule:
    name: str
    needed: Callable[[str, BackendABI], bool]
    apply: Callable[[str], tuple[str, bool]]
    """apply returns (new_src, applied). applied=False means the rule was a no-op
    even though `needed` returned True (e.g. already in target state)."""
    fail_if_not_applied: bool = True
    """If True and `needed(...)` returned True but the apply step couldn't find
    its target pattern, raise PatchError. Set False for rules whose absence is
    not fatal (e.g. cleanup of optional struct fields when absent in source)."""


# Match the *last* parameter as (struct)? ggml_cgraph * <id> — do NOT use a
# naive "[^)]*?cgraph" tail: that matches the substring "cgraph" inside the
# type name "ggml_cgraph" and breaks the regex (seen with upstream llama.cpp).
_GRAPH_COMPUTE_DEF_RE = re.compile(
    r"(static\s+(?:enum\s+)?ggml_status\s+ggml_backend_sycl_graph_compute\s*"
    r"\(\s*ggml_backend_t\s+\w+\s*,\s*(?:struct\s+)?ggml_cgraph\s*\*\s*\w+\s*)\s*\)",
    re.MULTILINE,
)
_GRAPH_COMPUTE_PATCHED_RE = re.compile(
    r"static\s+(?:enum\s+)?ggml_status\s+ggml_backend_sycl_graph_compute\s*"
    r"\(\s*ggml_backend_t\s+\w+\s*,\s*(?:struct\s+)?ggml_cgraph\s*\*\s*\w+\s*,\s*"
    r"int\s+batch_size\s*\)",
    re.MULTILINE,
)


def _graph_compute_already_patched(src: str) -> bool:
    return bool(_GRAPH_COMPUTE_PATCHED_RE.search(src))


def _rule_graph_compute_batch_size(src: str) -> tuple[str, bool]:
    """Add 'int batch_size' to ggml_backend_sycl_graph_compute and silence the unused warning."""
    new_src, count = _GRAPH_COMPUTE_DEF_RE.subn(r"\1, int batch_size)", src)
    if count == 0:
        return src, False
    new_src = re.sub(
        r"(ggml_backend_sycl_graph_compute\s*\(\s*ggml_backend_t\s+\w+\s*,\s*"
        r"(?:struct\s+)?ggml_cgraph\s*\*\s*\w+\s*,\s*int\s+batch_size\s*\)\s*\{)",
        r"\1\n    GGML_UNUSED(batch_size);",
        new_src,
        count=1,
    )
    return new_src, True


def _rule_strip_designated_field(field_name: str):
    """Build a rule that removes `/* .field_name = */ ...,` lines from designated initializers.

    The trailing `,` (and any trailing comment continuation) is consumed with
    the line. Multiple occurrences (e.g. main + split buffer interfaces) are
    all removed in one pass.
    """

    line_re = re.compile(
        rf"^[ \t]*/\*\s*\.{re.escape(field_name)}\s*=\s*\*/[^\n]*\n",
        re.MULTILINE,
    )

    def apply(src: str) -> tuple[str, bool]:
        new_src, count = line_re.subn("", src)
        return new_src, count > 0

    return apply


RULES: list[Rule] = [
    Rule(
        name="graph_compute_batch_size",
        needed=lambda src, abi: abi.graph_compute_has_batch_size
        and not _graph_compute_already_patched(src),
        apply=_rule_graph_compute_batch_size,
        fail_if_not_applied=True,
    ),
    Rule(
        name="strip_set_tensor_2d_async",
        needed=lambda src, abi: (not abi.has_set_tensor_2d_async)
        and "set_tensor_2d_async" in src,
        apply=_rule_strip_designated_field("set_tensor_2d_async"),
        fail_if_not_applied=False,
    ),
    Rule(
        name="strip_get_tensor_2d_async",
        needed=lambda src, abi: (not abi.has_get_tensor_2d_async)
        and "get_tensor_2d_async" in src,
        apply=_rule_strip_designated_field("get_tensor_2d_async"),
        fail_if_not_applied=False,
    ),
    Rule(
        name="strip_buffer_set_tensor_2d",
        needed=lambda src, abi: (not abi.has_buffer_set_tensor_2d)
        and re.search(r"/\*\s*\.set_tensor_2d\s*=", src) is not None,
        apply=_rule_strip_designated_field("set_tensor_2d"),
        fail_if_not_applied=False,
    ),
    Rule(
        name="strip_buffer_get_tensor_2d",
        needed=lambda src, abi: (not abi.has_buffer_get_tensor_2d)
        and re.search(r"/\*\s*\.get_tensor_2d\s*=", src) is not None,
        apply=_rule_strip_designated_field("get_tensor_2d"),
        fail_if_not_applied=False,
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def patch_file(src_path: Path, header_path: Path) -> PatchResult:
    abi = parse_backend_abi(header_path)
    src = src_path.read_text()
    original = src
    applied: list[str] = []

    for rule in RULES:
        if not rule.needed(src, abi):
            continue
        new_src, ok = rule.apply(src)
        if not ok:
            if rule.fail_if_not_applied:
                raise PatchError(
                    f"Rule '{rule.name}' is required by the Ollama ABI but its "
                    f"target pattern could not be located in {src_path}. "
                    "The upstream ggml-sycl source has likely drifted in a way "
                    "this patcher does not yet understand. Inspect the file "
                    "manually and update the patcher rules."
                )
            continue
        src = new_src
        applied.append(rule.name)

    changed = src != original
    if changed:
        src_path.write_text(src)
    return PatchResult(changed=changed, applied=applied)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("source", help="Path to upstream ggml-sycl.cpp")
    parser.add_argument(
        "--abi-header",
        default=str(DEFAULT_ABI_HEADER),
        help=f"Path to Ollama's ggml-backend-impl.h (default: {DEFAULT_ABI_HEADER})",
    )
    args = parser.parse_args(argv)

    src_path = Path(args.source)
    hdr_path = Path(args.abi_header)

    print(f"[patch-sycl] source: {src_path}")
    print(f"[patch-sycl] abi:    {hdr_path}")

    try:
        result = patch_file(src_path, hdr_path)
    except (FileNotFoundError, PatchError) as e:
        print(f"[patch-sycl] ERROR: {e}", file=sys.stderr)
        return 2

    if not result.changed:
        print("[patch-sycl] no patches needed — APIs match")
        return 0

    for name in result.applied:
        print(f"[patch-sycl]   applied: {name}")
    print(f"[patch-sycl] patched {src_path} successfully ({len(result.applied)} rules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

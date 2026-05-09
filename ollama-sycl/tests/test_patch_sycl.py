#!/usr/bin/env python3
"""
Tests for patch-sycl.py.

Each test writes a synthetic snippet representative of the upstream
llama.cpp ggml-sycl source and the matching Ollama ggml-backend-impl.h
header, runs the patcher, and asserts on the result.

Run with: python3 -m pytest ollama-sycl/tests/ -v
or just:  python3 ollama-sycl/tests/test_patch_sycl.py
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PATCH_SCRIPT = REPO_ROOT / "ollama-sycl" / "patch-sycl.py"


def _load_patcher():
    spec = importlib.util.spec_from_file_location("patch_sycl", PATCH_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["patch_sycl"] = module
    spec.loader.exec_module(module)
    return module


patcher = _load_patcher()


# -----------------------------------------------------------------------------
# Fixtures: minimal synthetic versions of the two real files
# -----------------------------------------------------------------------------

OLLAMA_HEADER_NEEDS_PATCH = textwrap.dedent(
    """\
    struct ggml_backend_i {
        const char * (*get_name)(ggml_backend_t backend);
        void (*free)(ggml_backend_t backend);
        void (*set_tensor_async)(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*get_tensor_async)(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
        bool (*cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);
        void (*synchronize)(ggml_backend_t backend);
        ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t backend, const struct ggml_cgraph * cgraph);
        void                      (*graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
        void                      (*graph_plan_update) (ggml_backend_t backend, ggml_backend_graph_plan_t plan, const struct ggml_cgraph * cgraph);
        enum ggml_status          (*graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);
        // compute graph (always async if supported by the backend). batch_size may be -1 if unknown
        enum ggml_status          (*graph_compute)     (ggml_backend_t backend, struct ggml_cgraph * cgraph, int batch_size);
        void (*event_record)(ggml_backend_t backend, ggml_backend_event_t event);
        void (*event_wait)  (ggml_backend_t backend, ggml_backend_event_t event);
        void                      (*graph_optimize)    (ggml_backend_t backend, struct ggml_cgraph * cgraph);
        enum ggml_status          (*graph_reserve)     (ggml_backend_t backend, struct ggml_cgraph * cgraph, bool alloc);
        size_t                    (*buffer_size)       (ggml_backend_t backend);
        void                      (*reset)             (ggml_backend_t backend);
    };

    struct ggml_backend_buffer_i {
        void         (*free_buffer)  (ggml_backend_buffer_t buffer);
        void *       (*get_base)     (ggml_backend_buffer_t buffer);
        enum ggml_status (*init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
        void         (*memset_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);
        void         (*set_tensor)   (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void         (*get_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
        bool         (*cpy_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst);
        void         (*clear)        (ggml_backend_buffer_t buffer, uint8_t value);
        void         (*reset)        (ggml_backend_buffer_t buffer);
    };
    """
)


OLLAMA_HEADER_CONVERGED = textwrap.dedent(
    """\
    struct ggml_backend_i {
        const char * (*get_name)(ggml_backend_t backend);
        void (*free)(ggml_backend_t backend);
        // already converged: no batch_size, has 2d_async
        void (*set_tensor_async)(ggml_backend_t, struct ggml_tensor *, const void *, size_t, size_t);
        void (*get_tensor_async)(ggml_backend_t, const struct ggml_tensor *, void *, size_t, size_t);
        void (*set_tensor_2d_async)(ggml_backend_t, struct ggml_tensor *, const void *, size_t, size_t, size_t, size_t, size_t);
        void (*get_tensor_2d_async)(ggml_backend_t, const struct ggml_tensor *, void *, size_t, size_t, size_t, size_t, size_t);
        bool (*cpy_tensor_async)(ggml_backend_t, ggml_backend_t, const struct ggml_tensor *, struct ggml_tensor *);
        void (*synchronize)(ggml_backend_t backend);
        ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t, const struct ggml_cgraph *);
        void                      (*graph_plan_free)   (ggml_backend_t, ggml_backend_graph_plan_t);
        void                      (*graph_plan_update) (ggml_backend_t, ggml_backend_graph_plan_t, const struct ggml_cgraph *);
        enum ggml_status          (*graph_plan_compute)(ggml_backend_t, ggml_backend_graph_plan_t);
        enum ggml_status          (*graph_compute)     (ggml_backend_t backend, struct ggml_cgraph * cgraph);
        void (*event_record)(ggml_backend_t backend, ggml_backend_event_t event);
        void (*event_wait)  (ggml_backend_t backend, ggml_backend_event_t event);
        void                      (*graph_optimize)    (ggml_backend_t, struct ggml_cgraph *);
    };
    """
)


UPSTREAM_SYCL_NEEDS_PATCH = textwrap.dedent(
    """\
    static const ggml_backend_buffer_i ggml_backend_sycl_buffer_interface = {
        /* .free_buffer     = */ ggml_backend_sycl_buffer_free_buffer,
        /* .get_base        = */ ggml_backend_sycl_buffer_get_base,
        /* .init_tensor     = */ ggml_backend_sycl_buffer_init_tensor,
        /* .memset_tensor   = */ ggml_backend_sycl_buffer_memset_tensor,
        /* .set_tensor      = */ ggml_backend_sycl_buffer_set_tensor,
        /* .get_tensor      = */ ggml_backend_sycl_buffer_get_tensor,
        /* .set_tensor_2d   = */ NULL,
        /* .get_tensor_2d   = */ NULL,
        /* .cpy_tensor      = */ ggml_backend_sycl_buffer_cpy_tensor,
        /* .clear           = */ ggml_backend_sycl_buffer_clear,
        /* .reset           = */ ggml_backend_sycl_buffer_reset,
    };

    static struct ggml_backend_buffer_i ggml_backend_sycl_split_buffer_interface = {
        /* .free_buffer     = */ ggml_backend_sycl_split_buffer_free_buffer,
        /* .get_base        = */ ggml_backend_sycl_split_buffer_get_base,
        /* .init_tensor     = */ ggml_backend_sycl_split_buffer_init_tensor,
        /* .memset_tensor   = */ NULL,
        /* .set_tensor      = */ ggml_backend_sycl_split_buffer_set_tensor,
        /* .get_tensor      = */ ggml_backend_sycl_split_buffer_get_tensor,
        /* .set_tensor_2d   = */ NULL,
        /* .get_tensor_2d   = */ NULL,
        /* .cpy_tensor      = */ NULL,
        /* .clear           = */ ggml_backend_sycl_split_buffer_clear,
        /* .reset           = */ NULL,
    };

    static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
        // body
        return GGML_STATUS_SUCCESS;
    }

    static ggml_backend_i ggml_backend_sycl_interface = {
        /* .get_name                = */ ggml_backend_sycl_get_name,
        /* .free                    = */ ggml_backend_sycl_free,
        /* .set_tensor_async        = */ ggml_backend_sycl_set_tensor_async,
        /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
        /* .set_tensor_2d_async     = */ NULL,
        /* .get_tensor_2d_async     = */ NULL,
        /* .cpy_tensor_async        = */ NULL,
        /* .synchronize             = */ ggml_backend_sycl_synchronize,
        /* .graph_plan_create       = */ NULL,
        /* .graph_plan_free         = */ NULL,
        /* .graph_plan_update       = */ NULL,
        /* .graph_plan_compute      = */ NULL,
        /* .graph_compute           = */ ggml_backend_sycl_graph_compute,
        /* .event_record            = */ ggml_backend_sycl_event_record,
        /* .event_wait              = */ ggml_backend_sycl_event_wait,
        /* .graph_optimize          = */ NULL,
    };
    """
)


UPSTREAM_SYCL_OLD_NO_2D = textwrap.dedent(
    """\
    static const ggml_backend_buffer_i ggml_backend_sycl_buffer_interface = {
        /* .free_buffer     = */ ggml_backend_sycl_buffer_free_buffer,
        /* .get_base        = */ ggml_backend_sycl_buffer_get_base,
        /* .init_tensor     = */ ggml_backend_sycl_buffer_init_tensor,
        /* .memset_tensor   = */ ggml_backend_sycl_buffer_memset_tensor,
        /* .set_tensor      = */ ggml_backend_sycl_buffer_set_tensor,
        /* .get_tensor      = */ ggml_backend_sycl_buffer_get_tensor,
        /* .cpy_tensor      = */ ggml_backend_sycl_buffer_cpy_tensor,
        /* .clear           = */ ggml_backend_sycl_buffer_clear,
        /* .reset           = */ ggml_backend_sycl_buffer_reset,
    };

    static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
        return GGML_STATUS_SUCCESS;
    }

    static ggml_backend_i ggml_backend_sycl_interface = {
        /* .get_name                = */ ggml_backend_sycl_get_name,
        /* .free                    = */ ggml_backend_sycl_free,
        /* .set_tensor_async        = */ ggml_backend_sycl_set_tensor_async,
        /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
        /* .cpy_tensor_async        = */ NULL,
        /* .synchronize             = */ ggml_backend_sycl_synchronize,
        /* .graph_plan_create       = */ NULL,
        /* .graph_plan_free         = */ NULL,
        /* .graph_plan_update       = */ NULL,
        /* .graph_plan_compute      = */ NULL,
        /* .graph_compute           = */ ggml_backend_sycl_graph_compute,
        /* .event_record            = */ ggml_backend_sycl_event_record,
        /* .event_wait              = */ ggml_backend_sycl_event_wait,
        /* .graph_optimize          = */ NULL,
    };
    """
)


@pytest.fixture
def tmp_files(tmp_path: Path):
    """Create a temporary src + header pair and return (src_path, header_path)."""

    def _make(src: str, header: str) -> tuple[Path, Path]:
        s = tmp_path / "ggml-sycl.cpp"
        h = tmp_path / "ggml-backend-impl.h"
        s.write_text(src)
        h.write_text(header)
        return s, h

    return _make


# -----------------------------------------------------------------------------
# ABI parser tests
# -----------------------------------------------------------------------------


def test_parse_abi_detects_batch_size(tmp_path: Path) -> None:
    h = tmp_path / "h.h"
    h.write_text(OLLAMA_HEADER_NEEDS_PATCH)
    abi = patcher.parse_backend_abi(h)
    assert abi.graph_compute_has_batch_size is True
    assert abi.has_set_tensor_2d_async is False
    assert abi.has_buffer_set_tensor_2d is False


def test_parse_abi_detects_converged(tmp_path: Path) -> None:
    h = tmp_path / "h.h"
    h.write_text(OLLAMA_HEADER_CONVERGED)
    abi = patcher.parse_backend_abi(h)
    assert abi.graph_compute_has_batch_size is False
    assert abi.has_set_tensor_2d_async is True


def test_parse_abi_missing_header_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        patcher.parse_backend_abi(tmp_path / "nope.h")


# -----------------------------------------------------------------------------
# Patching tests: applies all needed rules and reports correctly
# -----------------------------------------------------------------------------


def test_patch_applies_batch_size_and_strips_2d_fields(tmp_files) -> None:
    src_path, hdr_path = tmp_files(UPSTREAM_SYCL_NEEDS_PATCH, OLLAMA_HEADER_NEEDS_PATCH)
    result = patcher.patch_file(src_path, hdr_path)
    assert result.changed is True
    assert "graph_compute_batch_size" in result.applied
    assert "strip_set_tensor_2d_async" in result.applied
    assert "strip_get_tensor_2d_async" in result.applied
    assert "strip_buffer_set_tensor_2d" in result.applied
    assert "strip_buffer_get_tensor_2d" in result.applied

    patched = src_path.read_text()
    assert "int batch_size" in patched
    assert "GGML_UNUSED(batch_size);" in patched
    assert "set_tensor_2d_async" not in patched
    assert "get_tensor_2d_async" not in patched
    assert "set_tensor_2d   = " not in patched
    assert "get_tensor_2d   = " not in patched


def test_patch_no_op_when_apis_converged(tmp_files) -> None:
    converged_src = textwrap.dedent(
        """\
        static ggml_backend_i ggml_backend_sycl_interface = {
            /* .graph_compute = */ ggml_backend_sycl_graph_compute,
            /* .set_tensor_2d_async = */ NULL,
            /* .get_tensor_2d_async = */ NULL,
        };
        static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) { return GGML_STATUS_SUCCESS; }
        """
    )
    src_path, hdr_path = tmp_files(converged_src, OLLAMA_HEADER_CONVERGED)
    result = patcher.patch_file(src_path, hdr_path)
    assert result.changed is False
    assert result.applied == []


def test_patch_old_upstream_without_2d_only_adds_batch_size(tmp_files) -> None:
    src_path, hdr_path = tmp_files(UPSTREAM_SYCL_OLD_NO_2D, OLLAMA_HEADER_NEEDS_PATCH)
    result = patcher.patch_file(src_path, hdr_path)
    assert result.changed is True
    assert "graph_compute_batch_size" in result.applied
    assert "strip_set_tensor_2d_async" not in result.applied

    patched = src_path.read_text()
    assert "int batch_size" in patched


def test_patch_fails_loudly_when_required_signature_not_found(tmp_files) -> None:
    """If the header demands batch_size but graph_compute is unrecognisable in src,
    the patcher MUST raise — not exit silently with 'no changes'."""
    bad_src = "// no graph_compute here at all\nint other(void) { return 0; }\n"
    src_path, hdr_path = tmp_files(bad_src, OLLAMA_HEADER_NEEDS_PATCH)
    with pytest.raises(patcher.PatchError):
        patcher.patch_file(src_path, hdr_path)


def test_patch_is_idempotent(tmp_files) -> None:
    """Running the patcher twice produces the same result and reports 'no changes' on the second pass."""
    src_path, hdr_path = tmp_files(UPSTREAM_SYCL_NEEDS_PATCH, OLLAMA_HEADER_NEEDS_PATCH)
    first = patcher.patch_file(src_path, hdr_path)
    after_first = src_path.read_text()
    second = patcher.patch_file(src_path, hdr_path)
    after_second = src_path.read_text()
    assert first.changed is True
    assert second.changed is False
    assert after_first == after_second


# -----------------------------------------------------------------------------
# CLI smoke test
# -----------------------------------------------------------------------------


def test_cli_invocation(tmp_files, capsys) -> None:
    src_path, hdr_path = tmp_files(UPSTREAM_SYCL_NEEDS_PATCH, OLLAMA_HEADER_NEEDS_PATCH)
    rc = patcher.main([str(src_path), "--abi-header", str(hdr_path)])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "graph_compute_batch_size" in captured


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

"""Unit tests for the INT8 HiCache device staging buffers.

Runs against the real fork module ``int8_staging.py`` (imported as a flat module
by ``conftest.py``), so these exercise shipping code, not a copy. No CUDA needed:
the buffer geometry and growth policy are pure tensor logic.
"""

import unittest

import pytest
import torch

# The fork module is loaded by conftest only when the fork is present; skip at
# collection rather than erroring when it is missing (e.g. GitHub CI).
pytest.importorskip(
    "int8_staging", reason="requires the SGLang fork checked out at ../sglang"
)

from int8_staging import (  # noqa: E402
    DEFAULT_STAGING_TOKENS,
    STAGING_GROWTH_QUANTUM,
    StagingBuffers,
    allocate_staging,
    next_staging_capacity,
    pointer_table,
)

ROW_BYTES = 1152
LAYER_NUM = 36


class TestGrowthPolicy(unittest.TestCase):
    def test_fits_returns_current_capacity_unchanged(self):
        self.assertEqual(next_staging_capacity(0, 2048), 2048)
        self.assertEqual(next_staging_capacity(1, 2048), 2048)
        self.assertEqual(next_staging_capacity(2048, 2048), 2048)

    def test_grows_to_next_quantum(self):
        self.assertEqual(next_staging_capacity(2049, 2048), 3072)
        self.assertEqual(next_staging_capacity(3000, 2048), 3072)
        self.assertEqual(next_staging_capacity(3073, 2048), 4096)

    def test_growth_is_monotonic_and_grow_only(self):
        capacity = DEFAULT_STAGING_TOKENS
        for required in (1, 100, 2048, 2049, 5000, 4096, 10):
            new = next_staging_capacity(required, capacity)
            self.assertGreaterEqual(new, capacity)
            capacity = new

    def test_never_returns_less_than_required(self):
        for required in range(1, 5000, 137):
            capacity = next_staging_capacity(required, 512)
            self.assertGreaterEqual(capacity, required)

    def test_quantum_alignment(self):
        capacity = next_staging_capacity(1, 0)
        self.assertEqual(capacity % STAGING_GROWTH_QUANTUM, 0)


class TestAllocation(unittest.TestCase):
    def test_shape_is_layer_major(self):
        buffers = allocate_staging(LAYER_NUM, ROW_BYTES, device="cpu", capacity=64)
        self.assertEqual(buffers.k.shape, (LAYER_NUM, 64, ROW_BYTES))
        self.assertEqual(buffers.v.shape, (LAYER_NUM, 64, ROW_BYTES))
        self.assertEqual(buffers.k.dtype, torch.uint8)
        self.assertEqual(buffers.layer_num, LAYER_NUM)

    def test_zeroed_so_padding_is_deterministic(self):
        buffers = allocate_staging(2, ROW_BYTES, device="cpu", capacity=8)
        self.assertTrue(bool((buffers.k == 0).all()))
        self.assertTrue(bool((buffers.v == 0).all()))

    def test_capacity_property(self):
        buffers = allocate_staging(4, ROW_BYTES, device="cpu", capacity=128)
        self.assertEqual(buffers.capacity, 128)
        self.assertTrue(buffers.fits(128))
        self.assertFalse(buffers.fits(129))

    def test_rejects_non_positive_capacity(self):
        for capacity in (0, -1):
            with self.assertRaises(ValueError):
                allocate_staging(2, ROW_BYTES, device="cpu", capacity=capacity)

    def test_layer_slices_are_contiguous_rows(self):
        """Layer-major is what gives the mover a single per-token stride."""
        buffers = allocate_staging(4, ROW_BYTES, device="cpu", capacity=16)
        layer = buffers.layer_k(2, 16)
        self.assertEqual(layer.shape, (16, ROW_BYTES))
        self.assertTrue(layer.is_contiguous())
        # Row i of layer 2 starts exactly i * ROW_BYTES after row 0.
        offset = layer[1].data_ptr() - layer[0].data_ptr()
        self.assertEqual(offset, ROW_BYTES)

    def test_layer_views_share_storage(self):
        buffers = allocate_staging(4, ROW_BYTES, device="cpu", capacity=16)
        views = buffers.k_layer_views(16)
        self.assertEqual(len(views), 4)
        marker = torch.full((ROW_BYTES,), 7, dtype=torch.uint8)
        views[3][0] = marker
        self.assertTrue(torch.equal(buffers.k[3, 0], marker))
        # And a different layer is untouched.
        self.assertTrue(bool((buffers.k[2] == 0).all()))

    def test_partial_view_does_not_shrink_the_buffer(self):
        buffers = allocate_staging(2, ROW_BYTES, device="cpu", capacity=64)
        self.assertEqual(buffers.layer_k(0, 5).shape, (5, ROW_BYTES))
        self.assertEqual(buffers.capacity, 64)

    def test_k_and_v_are_independent_allocations(self):
        buffers = allocate_staging(2, ROW_BYTES, device="cpu", capacity=4)
        self.assertNotEqual(buffers.k.data_ptr(), buffers.v.data_ptr())


class TestPointerTables(unittest.TestCase):
    def test_table_holds_one_entry_per_tensor(self):
        buffers = allocate_staging(4, ROW_BYTES, device="cpu", capacity=8)
        views = buffers.k_layer_views(8)
        table = pointer_table(views, device="cpu")
        self.assertEqual(table.dtype, torch.uint64)
        self.assertEqual(table.shape, (4,))
        self.assertEqual([int(v) for v in table], [v.data_ptr() for v in views])

    def test_table_entries_are_row_bytes_apart(self):
        buffers = allocate_staging(4, ROW_BYTES, device="cpu", capacity=32)
        table = pointer_table(buffers.k_layer_views(32), device="cpu")
        entries = [int(v) for v in table]
        for a, b in zip(entries, entries[1:]):
            self.assertEqual(b - a, 32 * ROW_BYTES)

    def test_table_tracks_a_full_transfer(self):
        """Token i of layer l must sit at ptr[l] + i * ROW_BYTES."""
        buffers = allocate_staging(3, ROW_BYTES, device="cpu", capacity=16)
        table = pointer_table(buffers.k_layer_views(16), device="cpu")
        for layer in range(3):
            base = int(table[layer])
            for token in range(16):
                expected = buffers.k[layer, token].data_ptr()
                self.assertEqual(base + token * ROW_BYTES, expected)

    def test_empty_table(self):
        table = pointer_table([], device="cpu")
        self.assertEqual(table.shape, (0,))


class TestSeparateDirections(unittest.TestCase):
    def test_d2h_and_h2d_do_not_alias(self):
        """The two transfer streams can be in flight together, so the buffers
        must not overlap."""
        d2h = allocate_staging(2, ROW_BYTES, device="cpu", capacity=8)
        h2d = allocate_staging(2, ROW_BYTES, device="cpu", capacity=8)
        self.assertNotEqual(d2h.k.data_ptr(), h2d.k.data_ptr())
        # Writing one must not disturb the other.
        d2h.k.fill_(1)
        self.assertTrue(bool((h2d.k == 0).all()))


class TestDataclassShape(unittest.TestCase):
    def test_constructible_directly(self):
        k = torch.zeros((2, 4, ROW_BYTES), dtype=torch.uint8)
        v = torch.zeros((2, 4, ROW_BYTES), dtype=torch.uint8)
        buffers = StagingBuffers(k=k, v=v, layer_num=2)
        self.assertEqual(buffers.capacity, 4)
        self.assertTrue(buffers.fits(4))


if __name__ == "__main__":
    unittest.main()

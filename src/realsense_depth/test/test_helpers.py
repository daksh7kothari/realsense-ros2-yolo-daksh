"""Tests for the vectorised lattice helpers behind stages 2 and 4."""

import numpy as np
import pytest

from realsense_depth.cloud_pipeline import (connected_components,
                                            neighbour_offsets, voxel_packer)


def test_gap_one_is_the_26_neighbourhood():
    offsets = neighbour_offsets(1)
    assert offsets.shape == (26, 3)
    assert not (offsets == 0).all(axis=1).any()      # origin excluded


def test_gap_two_widens_the_neighbourhood():
    assert neighbour_offsets(2).shape == (124, 3)    # 5^3 - 1


def test_packing_is_injective():
    rng = np.random.default_rng(0)
    idx = np.unique(rng.integers(-50, 50, (5000, 3)), axis=0)
    keys = voxel_packer(idx)(idx)
    assert np.unique(keys).size == idx.shape[0]


def test_padding_keeps_out_of_range_probes_from_colliding():
    """A neighbour probe stepping outside the occupied set must not alias."""
    idx = np.array([[0, 0, 0], [0, 0, 5]], dtype=np.int64)
    pack = voxel_packer(idx, pad=1)
    real = pack(idx)
    outside = pack(idx + np.array([1, 0, 0]))
    assert not set(real.tolist()) & set(outside.tolist())


def test_packing_survives_negative_coordinates():
    idx = np.array([[-7, -3, -1], [0, 0, 0], [4, 2, 9]], dtype=np.int64)
    keys = voxel_packer(idx)(idx)
    assert np.unique(keys).size == 3
    assert (keys >= 0).all()


def test_isolated_nodes_each_form_their_own_component():
    labels = connected_components((np.empty(0, np.int64),
                                   np.empty(0, np.int64)), 4)
    assert np.unique(labels).size == 4


def test_a_chain_collapses_to_one_component():
    src = np.array([0, 1, 2, 3], dtype=np.int64)
    dst = np.array([1, 2, 3, 4], dtype=np.int64)
    labels = connected_components((src, dst), 5)
    assert np.unique(labels).size == 1


def test_two_disjoint_chains_stay_separate():
    src = np.array([0, 1, 5, 6], dtype=np.int64)
    dst = np.array([1, 2, 6, 7], dtype=np.int64)
    labels = connected_components((src, dst), 8)
    assert labels[0] == labels[1] == labels[2]
    assert labels[5] == labels[6] == labels[7]
    assert labels[0] != labels[5]


def test_labels_are_the_smallest_index_in_each_component():
    src = np.array([3, 4], dtype=np.int64)
    dst = np.array([4, 5], dtype=np.int64)
    labels = connected_components((src, dst), 6)
    assert labels[3] == labels[4] == labels[5] == 3


def test_long_chain_converges_within_the_iteration_cap():
    """Pointer jumping must flatten a 500-long chain, not walk it one hop a round."""
    n = 500
    src = np.arange(n - 1, dtype=np.int64)
    dst = np.arange(1, n, dtype=np.int64)
    labels = connected_components((src, dst), n)
    assert np.unique(labels).size == 1


def test_component_labelling_is_order_independent():
    rng = np.random.default_rng(1)
    src = np.array([0, 1, 2, 7, 8], dtype=np.int64)
    dst = np.array([1, 2, 3, 8, 9], dtype=np.int64)
    baseline = connected_components((src, dst), 10)

    shuffle = rng.permutation(src.size)
    shuffled = connected_components((src[shuffle], dst[shuffle]), 10)
    np.testing.assert_array_equal(baseline, shuffled)


@pytest.mark.parametrize('gap', [1, 2])
def test_neighbour_offsets_are_symmetric(gap):
    offsets = neighbour_offsets(gap)
    as_set = {tuple(o) for o in offsets}
    assert all(tuple(-np.array(o)) in as_set for o in as_set)

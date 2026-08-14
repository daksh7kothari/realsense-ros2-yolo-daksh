"""Stage-by-stage tests for the point cloud pipeline.

The scene is synthetic so the ground truth is exact: a horizontal plane at
z = 1.5 m, two boxes of known size in front of it, and far-field noise that the
passthrough stage is supposed to delete.
"""

import numpy as np
import pytest
import rclpy

from realsense_depth.cloud_pipeline import CloudPipeline

PLANE_Z = 1.5
BOX_A = ((-0.30, -0.10, 1.30), (-0.20, 0.05, 1.45))   # 0.10 x 0.15 x 0.15 m
BOX_B = ((0.15, -0.10, 1.20), (0.30, 0.05, 1.40))     # 0.15 x 0.15 x 0.20 m


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = CloudPipeline()
    yield n
    n.destroy_node()


@pytest.fixture
def scene():
    rng = np.random.default_rng(1)
    plane = np.column_stack([rng.uniform(-0.5, 0.5, 8000),
                             rng.uniform(-0.4, 0.4, 8000),
                             np.full(8000, PLANE_Z) + rng.normal(0, 0.002, 8000)])
    box_a = rng.uniform(BOX_A[0], BOX_A[1], (1500, 3))
    box_b = rng.uniform(BOX_B[0], BOX_B[1], (1500, 3))
    far = rng.uniform([-1, -1, 4.0], [1, 1, 6.0], (3000, 3))
    return np.vstack([plane, box_a, box_b, far]).astype(np.float32)


def set_param(node, name, value):
    node.set_parameters([rclpy.parameter.Parameter(name, value=value)])


# ------------------------------------------------------------------ stage 1

def test_passthrough_drops_everything_outside_the_band(node, scene):
    kept = node.passthrough(scene)
    assert kept.shape[0] == 11000            # the 3000 far points are gone
    assert kept[:, 2].min() >= 0.3
    assert kept[:, 2].max() <= 3.0


def test_passthrough_band_is_tunable_at_runtime(node, scene):
    set_param(node, 'z_max', 1.35)
    kept = node.passthrough(scene)
    assert kept[:, 2].max() <= 1.35


# ------------------------------------------------------------------ stage 2

def test_voxel_downsample_reduces_count_but_keeps_the_shape(node, scene):
    kept = node.passthrough(scene)
    small = node.voxel_downsample(kept)
    assert small.shape[0] < kept.shape[0]
    np.testing.assert_allclose(small.min(axis=0), kept.min(axis=0), atol=0.011)
    np.testing.assert_allclose(small.max(axis=0), kept.max(axis=0), atol=0.011)


def test_voxel_output_is_one_point_per_occupied_cell(node):
    """Two points in one 1 cm cube collapse to one; a third cube stays separate."""
    points = np.array([[0.000, 0.0, 1.0],
                       [0.005, 0.0, 1.0],
                       [0.050, 0.0, 1.0]], dtype=np.float32)
    out = node.voxel_downsample(points)
    assert out.shape[0] == 2


def test_voxel_takes_the_centroid_so_it_denoises(node):
    points = np.array([[0.001, 0.0, 1.0],
                       [0.009, 0.0, 1.0]], dtype=np.float32)
    out = node.voxel_downsample(points)
    assert out[0][0] == pytest.approx(0.005)


def test_larger_leaf_yields_fewer_points(node, scene):
    kept = node.passthrough(scene)
    fine = node.voxel_downsample(kept)
    set_param(node, 'voxel_leaf', 0.03)
    coarse = node.voxel_downsample(kept)
    assert coarse.shape[0] < fine.shape[0]


# ------------------------------------------------------------------ stage 3

def test_ransac_finds_the_dominant_plane(node, scene):
    small = node.voxel_downsample(node.passthrough(scene))
    rest, plane = node.remove_dominant_plane(small)

    assert plane.shape[0] > rest.shape[0]
    # Every inlier should sit on the synthetic plane.
    assert plane[:, 2].min() > PLANE_Z - 0.02
    assert plane[:, 2].max() < PLANE_Z + 0.02


def test_plane_removal_leaves_the_boxes_intact(node, scene):
    small = node.voxel_downsample(node.passthrough(scene))
    rest, _ = node.remove_dominant_plane(small)
    assert rest[:, 2].max() < PLANE_Z - 0.02


def test_declines_when_no_dominant_plane_is_in_view(node):
    """A weak best hypothesis must not carve a slab out of real objects."""
    rng = np.random.default_rng(2)
    blob = rng.uniform([-0.1, -0.1, 1.0], [0.1, 0.1, 1.4], (2000, 3)).astype(np.float32)
    rest, plane = node.remove_dominant_plane(blob)

    assert plane.shape[0] == 0
    assert rest.shape[0] == blob.shape[0]


def test_handles_a_cloud_too_small_to_define_a_plane(node):
    tiny = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)
    rest, plane = node.remove_dominant_plane(tiny)
    assert rest.shape[0] == 2 and plane.shape[0] == 0


def test_collinear_samples_do_not_produce_nan(node):
    """Three points on a line give a zero-length normal; it must be skipped."""
    line = np.column_stack([np.linspace(-0.2, 0.2, 500),
                            np.zeros(500),
                            np.full(500, 1.0)]).astype(np.float32)
    rest, plane = node.remove_dominant_plane(line)
    assert np.isfinite(rest).all() and np.isfinite(plane).all()


# ------------------------------------------------------------------ stage 4

def test_clustering_separates_the_two_boxes(node, scene):
    small = node.voxel_downsample(node.passthrough(scene))
    rest, _ = node.remove_dominant_plane(small)
    clusters = node.cluster(rest)
    assert len(clusters) == 2


def test_recovered_extents_match_ground_truth(node, scene):
    small = node.voxel_downsample(node.passthrough(scene))
    rest, _ = node.remove_dominant_plane(small)
    clusters = node.cluster(rest)

    found = sorted(tuple(np.round(c.max(axis=0) - c.min(axis=0), 2))
                   for c in clusters)
    expected = sorted([tuple(np.round(np.subtract(*reversed(box)), 2))
                       for box in (BOX_A, BOX_B)])
    # One voxel of slack: the grid quantises the extents.
    for got, want in zip(found, expected):
        np.testing.assert_allclose(got, want, atol=0.011)


def test_clusters_come_back_largest_first(node, scene):
    small = node.voxel_downsample(node.passthrough(scene))
    rest, _ = node.remove_dominant_plane(small)
    sizes = [c.shape[0] for c in node.cluster(rest)]
    assert sizes == sorted(sizes, reverse=True)


def test_specks_below_the_threshold_are_dropped(node):
    rng = np.random.default_rng(3)
    blob = rng.uniform([0, 0, 1.0], [0.1, 0.1, 1.1], (500, 3)).astype(np.float32)
    speck = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
    clusters = node.cluster(np.vstack([blob, speck]))
    assert len(clusters) == 1


def test_empty_cloud_clusters_to_nothing(node):
    assert node.cluster(np.empty((0, 3), np.float32)) == []


def test_wider_gap_tolerance_merges_neighbouring_objects(node):
    """Two blobs one empty voxel apart: separate at gap 1, merged at gap 2."""
    a = np.array([[0.000, 0.0, 1.0]], dtype=np.float32)
    b = np.array([[0.025, 0.0, 1.0]], dtype=np.float32)
    both = np.vstack([a, b])

    set_param(node, 'min_cluster_points', 1)
    assert len(node.cluster(both)) == 2
    set_param(node, 'cluster_gap_voxels', 2)
    assert len(node.cluster(both)) == 1

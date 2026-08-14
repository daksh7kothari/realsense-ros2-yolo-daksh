"""Decoder tests. These cover the failure modes that hide on a live camera."""

import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from realsense_depth.msg_utils import (Intrinsics, image_to_numpy,
                                       pointcloud2_to_xyz, xyz_to_pointcloud2)


def make_depth_image(values, step=None):
    """Build a 16UC1 Image, optionally with padding after each row."""
    values = np.asarray(values, dtype=np.uint16)
    height, width = values.shape
    used = width * 2
    step = used if step is None else step

    buf = np.zeros((height, step), dtype=np.uint8)
    buf[:, :used] = values.view(np.uint8).reshape(height, used)

    msg = Image()
    msg.height, msg.width = height, width
    msg.encoding = '16UC1'
    msg.step = step
    msg.data = buf.tobytes()
    return msg


def test_decodes_tightly_packed_depth():
    values = (np.arange(12, dtype=np.uint16) * 100).reshape(3, 4)
    out = image_to_numpy(make_depth_image(values))
    assert out.shape == (3, 4)
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, values)


def test_decodes_row_padded_depth():
    """step may exceed width*itemsize; ignoring it shears the image."""
    values = (np.arange(12, dtype=np.uint16) * 100).reshape(3, 4)
    out = image_to_numpy(make_depth_image(values, step=12))  # 8 used, 4 padding
    np.testing.assert_array_equal(out, values)


def test_decodes_rgb8():
    msg = Image()
    msg.height, msg.width = 2, 3
    msg.encoding = 'rgb8'
    msg.step = 9
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    msg.data = pixels.tobytes()
    np.testing.assert_array_equal(image_to_numpy(msg), pixels)


def test_rejects_unknown_encoding():
    msg = Image()
    msg.height, msg.width, msg.step = 1, 1, 1
    msg.encoding = 'bayer_rggb8'
    with pytest.raises(ValueError, match='unsupported image encoding'):
        image_to_numpy(msg)


def make_camera_info(width=848, height=480, fx=645.0, fy=645.0,
                     cx=424.0, cy=240.0):
    info = CameraInfo()
    info.width, info.height = width, height
    info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    info.header.frame_id = 'camera_color_optical_frame'
    return info


def test_intrinsics_parsed_from_k_matrix():
    intr = Intrinsics(make_camera_info())
    assert (intr.fx, intr.fy, intr.cx, intr.cy) == (645.0, 645.0, 424.0, 240.0)
    assert intr.frame_id == 'camera_color_optical_frame'


def test_principal_point_deprojects_onto_optical_axis():
    intr = Intrinsics(make_camera_info())
    x, y, z = intr.deproject(424, 240, 2.0)
    assert (x, y, z) == pytest.approx((0.0, 0.0, 2.0))


def test_deproject_scales_offset_with_range():
    """Same pixel offset subtends a larger metric offset further away."""
    intr = Intrinsics(make_camera_info())
    near = intr.deproject(524, 240, 1.0)
    far = intr.deproject(524, 240, 2.0)
    assert far[0] == pytest.approx(2 * near[0])
    assert near[0] == pytest.approx(100 * 1.0 / 645.0)


def test_deproject_y_is_positive_downward():
    """Optical frame convention: +y points down, not up."""
    intr = Intrinsics(make_camera_info())
    below = intr.deproject(424, 340, 1.0)
    assert below[1] > 0


def test_deproject_array_matches_scalar_path():
    intr = Intrinsics(make_camera_info())
    us = np.array([424.0, 524.0, 324.0])
    vs = np.array([240.0, 140.0, 340.0])
    zs = np.array([1.0, 2.0, 0.5])
    batch = intr.deproject_array(us, vs, zs)
    for i in range(3):
        assert batch[i] == pytest.approx(intr.deproject(us[i], vs[i], zs[i]))


def test_pointcloud_roundtrip():
    points = np.random.default_rng(0).random((64, 3)).astype(np.float32)
    back = pointcloud2_to_xyz(xyz_to_pointcloud2(points, Header()))
    np.testing.assert_allclose(points, back)


def test_pointcloud_roundtrip_preserves_header_and_layout():
    header = Header()
    header.frame_id = 'camera_depth_optical_frame'
    msg = xyz_to_pointcloud2(np.zeros((10, 3), np.float32), header)
    assert msg.header.frame_id == 'camera_depth_optical_frame'
    assert (msg.height, msg.width, msg.point_step, msg.row_step) == (1, 10, 12, 120)


def test_pointcloud_drops_non_finite_points():
    points = np.array([[0.0, 0.0, 1.0],
                       [np.nan, 0.0, 1.0],
                       [0.0, np.inf, 1.0],
                       [1.0, 1.0, 2.0]], dtype=np.float32)
    kept = pointcloud2_to_xyz(xyz_to_pointcloud2(points, Header()))
    assert kept.shape == (2, 3)


def test_pointcloud_reads_xyz_from_a_wider_stride():
    """Real RealSense clouds are XYZRGB at 32 bytes; xyz sits in the first 12."""
    msg = xyz_to_pointcloud2(np.array([[1.0, 2.0, 3.0]], np.float32), Header())
    msg.point_step = 32
    msg.row_step = 32
    msg.data = bytes(msg.data) + bytes(20)      # 12 bytes of xyz, then padding
    np.testing.assert_allclose(pointcloud2_to_xyz(msg), [[1.0, 2.0, 3.0]])

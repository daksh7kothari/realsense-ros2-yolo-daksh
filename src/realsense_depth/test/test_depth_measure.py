"""End-to-end tests for the depth probe, driven by synthetic messages."""

import math

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PointStamped

from realsense_depth.depth_measure import DepthMeasure

from test_msg_utils import make_camera_info, make_depth_image


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = DepthMeasure()
    n.set_parameters([rclpy.parameter.Parameter('roi_half', value=1)])
    n.roi_half = 1
    n.captured = []
    n.point_pub.publish = lambda msg: n.captured.append(msg)
    yield n
    n.destroy_node()


def uniform_depth(height, width, millimetres):
    return make_depth_image(np.full((height, width), millimetres, np.uint16))


def test_probe_defaults_to_image_centre(node):
    node.on_info(make_camera_info(width=848, height=480))
    assert node.probe == (424, 240)


def test_reports_depth_in_metres(node):
    node.on_info(make_camera_info(width=8, height=6, cx=4.0, cy=3.0))
    node.on_depth(uniform_depth(6, 8, 1500))

    assert len(node.captured) == 1
    assert isinstance(node.captured[0], PointStamped)
    assert node.captured[0].point.z == pytest.approx(1.5)


def test_ignores_depth_before_intrinsics_arrive(node):
    node.on_depth(uniform_depth(6, 8, 1500))
    assert node.captured == []


def test_zero_is_treated_as_no_measurement_not_zero_range(node):
    """0 is the RealSense invalid sentinel; averaging it in would bias low."""
    node.on_info(make_camera_info(width=8, height=6, cx=4.0, cy=3.0))
    values = np.full((6, 8), 2000, np.uint16)
    values[2:5, 3:6] = 0            # blank out the whole ROI around (4, 3)
    node.on_depth(make_depth_image(values))

    assert node.captured == []      # nothing published, rather than 0.0 m
    assert node.last_point is None


def test_median_rejects_flying_pixels(node):
    """A few wild samples at a depth edge must not move the reported range."""
    node.on_info(make_camera_info(width=8, height=6, cx=4.0, cy=3.0))
    values = np.full((6, 8), 1000, np.uint16)
    values[2, 3] = 9000             # flying pixel inside the 3x3 ROI
    values[4, 5] = 0                # dropout inside the 3x3 ROI
    node.on_depth(make_depth_image(values))

    assert node.captured[0].point.z == pytest.approx(1.0)


def test_deprojects_off_centre_probe_to_metric_offset(node):
    node.on_info(make_camera_info(width=848, height=480, fx=645.0, fy=645.0,
                                  cx=424.0, cy=240.0))
    node.probe = (524, 240)
    node.on_depth(uniform_depth(480, 848, 2000))

    point = node.captured[0].point
    assert point.x == pytest.approx(100 * 2.0 / 645.0)
    assert point.y == pytest.approx(0.0)
    assert point.z == pytest.approx(2.0)


def test_publishes_in_the_depth_message_frame(node):
    """Downstream TF2 consumers depend on this frame_id being passed through."""
    node.on_info(make_camera_info(width=8, height=6, cx=4.0, cy=3.0))
    msg = uniform_depth(6, 8, 1500)
    msg.header.frame_id = 'camera_color_optical_frame'
    node.on_depth(msg)

    assert node.captured[0].header.frame_id == 'camera_color_optical_frame'


def test_probe_outside_the_image_yields_nothing(node):
    node.on_info(make_camera_info(width=8, height=6, cx=4.0, cy=3.0))
    node.probe = (99, 99)
    node.on_depth(uniform_depth(6, 8, 1500))
    assert node.captured == []


def test_span_measures_true_3d_distance_not_pixel_scaling(node):
    """Two markers at different ranges: the span must include the depth gap."""
    intr_info = make_camera_info(width=848, height=480)
    node.on_info(intr_info)

    a = node.intr.deproject(424, 240, 1.0)      # on axis, 1.0 m
    b = node.intr.deproject(424, 240, 1.3)      # same pixel, 1.3 m
    node.markers = [(424, 240) + a, (424, 240) + b]

    assert node._span(node.markers) == pytest.approx(0.3)


def test_span_of_a_fronto_parallel_object(node):
    node.on_info(make_camera_info(width=848, height=480, fx=645.0, fy=645.0,
                                  cx=424.0, cy=240.0))
    left = node.intr.deproject(374, 240, 2.0)
    right = node.intr.deproject(474, 240, 2.0)
    node.markers = [(374, 240) + left, (474, 240) + right]

    # 100 px across at 2 m with fx=645 subtends 100*2/645 m.
    assert node._span(node.markers) == pytest.approx(100 * 2.0 / 645.0)


def test_error_model_grows_with_the_square_of_range(node):
    near = node.depth_rms(1.0)
    far = node.depth_rms(2.0)
    assert far == pytest.approx(4 * near)
    assert near * 1000 == pytest.approx(2.5, abs=0.3)   # ~2.5 mm at 1 m


def test_sample_uses_median_of_valid_pixels_only(node):
    node.on_info(make_camera_info(width=8, height=6, cx=4.0, cy=3.0))
    values = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1000, 1100, 1200, 0, 0],
                       [0, 0, 0, 1000, 0, 1200, 0, 0],
                       [0, 0, 0, 1000, 1100, 1200, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint16)
    # The 3x3 ROI holds 1000,1100,1200 / 1000,-,1200 / 1000,1100,1200.
    # Median of those 8 valid samples is 1100 mm; the 0 is excluded, not averaged.
    assert node.sample(values, 4, 3) == pytest.approx(1.1)


def test_span_is_symmetric(node):
    node.on_info(make_camera_info())
    a = (10, 10) + node.intr.deproject(10, 10, 1.0)
    b = (20, 20) + node.intr.deproject(20, 20, 1.5)
    assert node._span([a, b]) == pytest.approx(node._span([b, a]))
    assert node._span([a, b]) == pytest.approx(
        math.dist(a[2:], b[2:]))

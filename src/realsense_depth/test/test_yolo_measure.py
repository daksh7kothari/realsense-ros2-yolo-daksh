"""Tests for the YOLO box detector + distance/offset node.

ultralytics is not installed in the test environment (it's a heavy pip-only
dependency, see docs/12-yolo-detection.md), so every fixture points model_path
at a file that doesn't exist. YoloMeasure._load_model catches that and leaves
self.model = None, exactly as it would if ultralytics itself were missing.
That's enough to exercise everything downstream of a detection: the boxes
themselves are injected directly onto node.latest_box.
"""

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PointStamped

from realsense_depth.yolo_measure import (YoloMeasure, boxes_from_result,
                                          pick_largest_box)

from test_msg_utils import make_camera_info, make_depth_image


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = YoloMeasure(parameter_overrides=[
        rclpy.parameter.Parameter('model_path', value='/nonexistent/model.pt'),
    ])
    assert n.model is None      # confirms this didn't silently load real weights
    n.captured_points = []
    n.captured_dist = []
    n.captured_offset = []
    n.point_pub.publish = lambda msg: n.captured_points.append(msg)
    n.dist_pub.publish = lambda msg: n.captured_dist.append(msg)
    n.offset_pub.publish = lambda msg: n.captured_offset.append(msg)
    yield n
    n.destroy_node()


def uniform_depth(height, width, millimetres):
    return make_depth_image(np.full((height, width), millimetres, np.uint16))


# --------------------------------------------------------------- pure helpers

def test_pick_largest_box_returns_none_for_empty():
    assert pick_largest_box([]) is None


def test_pick_largest_box_picks_biggest_area():
    small = (0, 0, 10, 10, 0.9)
    big = (0, 0, 100, 50, 0.4)
    assert pick_largest_box([small, big]) == big


def test_pick_largest_box_single_box():
    only = (5, 5, 15, 25, 0.7)
    assert pick_largest_box([only]) == only


class FakeBoxes:
    def __init__(self, xyxy, conf):
        self.xyxy = np.array(xyxy)
        self.conf = np.array(conf)

    def __len__(self):
        return len(self.conf)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def test_boxes_from_result_empty_when_no_boxes():
    assert boxes_from_result(FakeResult(None)) == []
    assert boxes_from_result(FakeResult(FakeBoxes([], []))) == []


def test_boxes_from_result_pairs_xyxy_with_confidence():
    result = FakeResult(FakeBoxes([[0, 0, 10, 20], [5, 5, 15, 25]], [0.9, 0.3]))
    boxes = boxes_from_result(result)
    assert boxes == [(0.0, 0.0, 10.0, 20.0, 0.9), (5.0, 5.0, 15.0, 25.0, 0.3)]


# -------------------------------------------------------------------- offset

def test_offset_percent_zero_at_image_centre(node):
    node.on_info(make_camera_info(width=848, height=480))
    assert node.offset_percent(424) == pytest.approx(0.0)


def test_offset_percent_negative_on_the_left(node):
    node.on_info(make_camera_info(width=848, height=480))
    assert node.offset_percent(0) == pytest.approx(-100.0)


def test_offset_percent_positive_on_the_right(node):
    node.on_info(make_camera_info(width=848, height=480))
    assert node.offset_percent(848) == pytest.approx(100.0)


def test_offset_percent_scales_linearly(node):
    node.on_info(make_camera_info(width=848, height=480))
    # 10% of the way from centre (424) to the right edge (848) is 424 + 42.4
    assert node.offset_percent(424 + 42.4) == pytest.approx(10.0)


# ----------------------------------------------------------------- on_depth

def test_no_publish_without_a_detected_box(node):
    node.on_info(make_camera_info(width=8, height=6, cx=4.0, cy=3.0))
    node.on_depth(uniform_depth(6, 8, 1500))
    assert node.captured_points == []


def test_no_publish_before_intrinsics_arrive(node):
    node.latest_box = (0, 0, 8, 6, 0.9)
    node.on_depth(uniform_depth(6, 8, 1500))
    assert node.captured_points == []


def test_reports_distance_at_box_centre(node):
    node.on_info(make_camera_info(width=8, height=6, cx=4.0, cy=3.0))
    node.roi_half = 1
    node.latest_box = (2, 2, 6, 4, 0.9)     # centre = (4, 3)
    node.on_depth(uniform_depth(6, 8, 1500))

    assert len(node.captured_points) == 1
    assert isinstance(node.captured_points[0], PointStamped)
    assert node.captured_points[0].point.z == pytest.approx(1.5)
    assert node.captured_dist[0].data == pytest.approx(1.5)


def test_offset_published_alongside_distance(node):
    node.on_info(make_camera_info(width=8, height=6, cx=4.0, cy=3.0))
    node.roi_half = 1
    node.latest_box = (2, 2, 6, 4, 0.9)     # centre u=4 -> dead centre of an 8-wide frame
    node.on_depth(uniform_depth(6, 8, 1500))

    assert node.captured_offset[0].data == pytest.approx(0.0)


def test_box_left_of_centre_reports_negative_offset(node):
    node.on_info(make_camera_info(width=8, height=6, cx=4.0, cy=3.0))
    node.roi_half = 1
    node.latest_box = (0, 2, 2, 4, 0.9)     # centre u=1, left of the frame's centre (4)
    node.on_depth(uniform_depth(6, 8, 1500))

    assert node.captured_offset[0].data < 0


def test_box_right_of_centre_reports_positive_offset(node):
    node.on_info(make_camera_info(width=8, height=6, cx=4.0, cy=3.0))
    node.roi_half = 1
    node.latest_box = (6, 2, 8, 4, 0.9)     # centre u=7, right of the frame's centre (4)
    node.on_depth(uniform_depth(6, 8, 1500))

    assert node.captured_offset[0].data > 0


def test_no_valid_depth_at_box_centre_publishes_nothing(node):
    node.on_info(make_camera_info(width=8, height=6, cx=4.0, cy=3.0))
    node.roi_half = 1
    node.latest_box = (2, 2, 6, 4, 0.9)
    values = np.full((6, 8), 1500, np.uint16)
    values[2:5, 3:6] = 0        # blank the ROI around the box centre (4, 3)
    node.on_depth(make_depth_image(values))

    assert node.captured_points == []
    assert node.last_point is None

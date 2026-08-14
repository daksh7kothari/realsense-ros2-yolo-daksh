"""Detect a box with YOLO, then report its metric distance and framing offset.

Pipeline per colour frame: run the detector, keep the largest box (the model
is a single-class "item" detector trained to find the box to pick up), then on
the next aligned-depth frame sample depth at the box's centre pixel the same
way ``depth_measure`` samples its probe - median of a small ROI, since a
single pixel is noisy and sometimes a hole.

Framing offset is *pixel* position, not physical yaw: how far the box centre
sits from the image's horizontal centre, as a percentage of the half-width.
0% is dead centre, negative is left, positive is right. That is the signal a
robot needs to re-centre on the box; it says nothing about which way the box
itself is turned.
"""

import threading

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32

from realsense_depth.msg_utils import Intrinsics, image_to_numpy, sample_depth


def pick_largest_box(boxes):
    """boxes: iterable of (x1, y1, x2, y2, conf). Returns the biggest, or None."""
    if not boxes:
        return None

    def area(box):
        x1, y1, x2, y2, _ = box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    return max(boxes, key=area)


def boxes_from_result(result):
    """Pull (x1, y1, x2, y2, conf) tuples out of one ultralytics Results object."""
    boxes = getattr(result, 'boxes', None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.tolist()
    confs = boxes.conf.tolist()
    return [(x1, y1, x2, y2, c) for (x1, y1, x2, y2), c in zip(xyxy, confs)]


class YoloMeasure(Node):

    def __init__(self, **kwargs):
        super().__init__('yolo_measure', **kwargs)

        self.declare_parameter('model_path',
                               '/home/daksh/realsense/largestboxmodel(1).pt')
        self.declare_parameter('depth_topic',
                               '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('conf_thres', 0.5)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('device', 'cpu')
        # Half-width of the depth sampling window, so 5 gives an 11x11 ROI.
        self.declare_parameter('roi_half', 5)
        # D435 ships 16UC1 depth in millimetres.
        self.declare_parameter('depth_scale', 0.001)
        # Caps inference rate; a CPU-only nano model can't keep up with 30 Hz.
        self.declare_parameter('infer_period_sec', 0.2)
        self.declare_parameter('viewer', False)
        self.declare_parameter('log_period_sec', 0.5)

        self.conf_thres = float(self.get_parameter('conf_thres').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.device = self.get_parameter('device').value
        self.roi_half = int(self.get_parameter('roi_half').value)
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.infer_period = float(self.get_parameter('infer_period_sec').value)
        self.viewer = bool(self.get_parameter('viewer').value)

        self.intr = None
        self.latest_color = None
        self.latest_box = None         # (x1, y1, x2, y2, conf) in pixel coords
        self.last_point = None         # deprojected (x, y, z) of the box centre
        self.lock = threading.Lock()
        self.last_infer = 0.0
        self.last_log = 0.0

        self.model = self._load_model(self.get_parameter('model_path').value)

        self.point_pub = self.create_publisher(PointStamped, '~/point', 10)
        self.dist_pub = self.create_publisher(Float32, '~/distance', 10)
        self.offset_pub = self.create_publisher(Float32, '~/offset_percent', 10)
        self.detected_pub = self.create_publisher(Bool, '~/detected', 10)

        # RealSense image topics are BEST_EFFORT. Subscribing with the default
        # RELIABLE profile silently yields zero messages.
        self.create_subscription(CameraInfo,
                                 self.get_parameter('info_topic').value,
                                 self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image,
                                 self.get_parameter('depth_topic').value,
                                 self.on_depth, qos_profile_sensor_data)
        self.create_subscription(Image,
                                 self.get_parameter('color_topic').value,
                                 self.on_color, qos_profile_sensor_data)
        if self.viewer:
            self._init_viewer()

        self.get_logger().info('yolo_measure up, waiting for camera_info')

    # ------------------------------------------------------------- model I/O

    def _load_model(self, path):
        try:
            from ultralytics import YOLO
        except ImportError:
            self.get_logger().error(
                'ultralytics not installed - detection disabled. '
                'Run: pip3 install ultralytics')
            return None
        try:
            return YOLO(path)
        except Exception as exc:  # noqa: BLE001 - report and keep node alive
            self.get_logger().error(f'failed to load model at {path}: {exc}')
            return None

    # ---------------------------------------------------------------- viewer

    def _init_viewer(self):
        import cv2
        self.cv2 = cv2
        self.window = 'yolo_measure  [q: quit]'
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        self.create_timer(0.03, self._draw)

    def _draw(self):
        with self.lock:
            frame = None if self.latest_color is None else self.latest_color.copy()
            box, point = self.latest_box, self.last_point
        if frame is None:
            return
        cv2 = self.cv2
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if box is not None:
            x1, y1, x2, y2, conf = box
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
            u, v = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            cv2.drawMarker(frame, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
            if point is not None and self.intr is not None:
                pct = self.offset_percent(u)
                side = 'R' if pct >= 0 else 'L'
                label = f'{point[2]:.3f} m  {side}{abs(pct):.0f}%  conf={conf:.2f}'
                cv2.putText(frame, label, (p1[0], max(0, p1[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(self.window, frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            rclpy.shutdown()

    # ------------------------------------------------------------- callbacks

    def on_info(self, msg):
        if self.intr is None:
            self.intr = Intrinsics(msg)
            self.get_logger().info(
                f'intrinsics: fx={self.intr.fx:.2f} fy={self.intr.fy:.2f} '
                f'cx={self.intr.cx:.2f} cy={self.intr.cy:.2f} '
                f'frame={self.intr.frame_id}')

    def on_color(self, msg):
        frame = image_to_numpy(msg)
        with self.lock:
            self.latest_color = frame
        if self.model is None:
            return

        now = self._now()
        if now - self.last_infer < self.infer_period:
            return
        self.last_infer = now

        # rgb8 -> bgr, the array layout ultralytics/cv2 expect. Contiguous
        # because a reversed view trips up some cv2 resize/letterbox paths.
        bgr = np.ascontiguousarray(frame[:, :, ::-1])
        results = self.model.predict(bgr, conf=self.conf_thres,
                                     imgsz=self.imgsz, device=self.device,
                                     verbose=False)
        box = pick_largest_box(boxes_from_result(results[0]) if results else [])
        with self.lock:
            self.latest_box = box
        self.detected_pub.publish(Bool(data=box is not None))

    def on_depth(self, msg):
        if self.intr is None:
            return
        with self.lock:
            box = self.latest_box
        if box is None:
            return

        x1, y1, x2, y2, conf = box
        u = int(round((x1 + x2) / 2.0))
        v = int(round((y1 + y2) / 2.0))

        depth = image_to_numpy(msg)
        z = sample_depth(depth, u, v, self.roi_half, self.depth_scale)
        if z is None:
            with self.lock:
                self.last_point = None
            self.throttled_log(f'box centre ({u},{v}) conf={conf:.2f}: no valid depth')
            return

        point = self.intr.deproject(u, v, z)
        with self.lock:
            self.last_point = point

        x, y, z = point
        out = PointStamped()
        out.header = msg.header
        out.point.x, out.point.y, out.point.z = float(x), float(y), float(z)
        self.point_pub.publish(out)
        self.dist_pub.publish(Float32(data=float(z)))

        pct = self.offset_percent(u)
        self.offset_pub.publish(Float32(data=float(pct)))

        side = 'right' if pct >= 0 else 'left'
        self.throttled_log(
            f'box conf={conf:.2f} centre=({u},{v}) dist={z:.3f} m {side} {abs(pct):.1f}%')

    # ----------------------------------------------------------------- maths

    def offset_percent(self, u):
        """Signed horizontal offset of pixel column u from the frame centre.

        0% is dead centre, -100% is the left edge, +100% is the right edge.
        Uses the geometric image centre, not the lens principal point (cx),
        since this describes framing in the picture, not optical distortion.
        """
        half_w = self.intr.width / 2.0
        return (u - half_w) / half_w * 100.0

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def throttled_log(self, text):
        now = self._now()
        if now - self.last_log >= self.get_parameter('log_period_sec').value:
            self.last_log = now
            self.get_logger().info(text)


def main():
    rclpy.init()
    node = YoloMeasure()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

"""Probe the aligned depth image and report metric 3D positions.

Two things happen here:

1. A probe pixel is sampled every frame. The depth of a single pixel is noisy
   and is 0 wherever the stereo matcher found no correspondence, so the value
   reported is the median of the non-zero depths inside a small ROI.
2. The probe is deprojected to metres using the colour camera intrinsics, which
   is what turns "1832 raw units" into a real (x, y, z) in the optical frame.

With ``viewer:=true`` an OpenCV window opens. Left-click moves the probe.
Right-click drops a marker; with two markers down, the node prints the straight
line distance between them, which is how you measure an object's size.
"""

import math
import threading

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32

from realsense_depth.msg_utils import Intrinsics, image_to_numpy, sample_depth


class DepthMeasure(Node):

    def __init__(self):
        super().__init__('depth_measure')

        self.declare_parameter('depth_topic',
                               '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        # -1 means "centre of the image", resolved once intrinsics arrive.
        self.declare_parameter('probe_u', -1)
        self.declare_parameter('probe_v', -1)
        # Half-width of the sampling window, so 5 gives an 11x11 ROI.
        self.declare_parameter('roi_half', 5)
        # D435 ships 16UC1 depth in millimetres.
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('viewer', False)
        self.declare_parameter('log_period_sec', 0.5)

        self.depth_scale = self.get_parameter('depth_scale').value
        self.roi_half = int(self.get_parameter('roi_half').value)
        self.viewer = bool(self.get_parameter('viewer').value)

        self.intr = None
        self.probe = (int(self.get_parameter('probe_u').value),
                      int(self.get_parameter('probe_v').value))
        self.markers = []          # up to two (u, v, x, y, z) measurement anchors
        self.latest_color = None
        self.lock = threading.Lock()
        self.last_log = 0.0

        self.point_pub = self.create_publisher(PointStamped, '~/point', 10)
        self.dist_pub = self.create_publisher(Float32, '~/distance', 10)

        # RealSense image topics are BEST_EFFORT. Subscribing with the default
        # RELIABLE profile silently yields zero messages.
        self.create_subscription(CameraInfo,
                                 self.get_parameter('info_topic').value,
                                 self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image,
                                 self.get_parameter('depth_topic').value,
                                 self.on_depth, qos_profile_sensor_data)
        if self.viewer:
            self.create_subscription(Image,
                                     self.get_parameter('color_topic').value,
                                     self.on_color, qos_profile_sensor_data)
            self._init_viewer()

        self.get_logger().info('depth_measure up, waiting for camera_info')

    # ---------------------------------------------------------------- viewer

    def _init_viewer(self):
        import cv2
        self.cv2 = cv2
        self.window = 'depth_measure  [L-click: probe]  [R-click: marker]  [c: clear]  [q: quit]'
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._on_mouse)
        self.create_timer(0.03, self._draw)

    def _on_mouse(self, event, x, y, flags, _param):
        if event == self.cv2.EVENT_LBUTTONDOWN:
            with self.lock:
                self.probe = (x, y)
        elif event == self.cv2.EVENT_RBUTTONDOWN:
            with self.lock:
                point = self.last_point
                if point is not None:
                    self.markers.append((x, y) + point)
                    self.markers = self.markers[-2:]
                    self._report_span()

    def _draw(self):
        with self.lock:
            frame = None if self.latest_color is None else self.latest_color.copy()
            probe, markers, point = self.probe, list(self.markers), self.last_point
        if frame is None:
            return
        cv2 = self.cv2
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.drawMarker(frame, probe, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        label = 'no depth' if point is None else f'{point[2]:.3f} m'
        cv2.putText(frame, label, (probe[0] + 12, probe[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for u, v, *_ in markers:
            cv2.circle(frame, (u, v), 6, (0, 128, 255), -1)
        if len(markers) == 2:
            cv2.line(frame, markers[0][:2], markers[1][:2], (0, 128, 255), 2)
            span = self._span(markers)
            mid = ((markers[0][0] + markers[1][0]) // 2,
                   (markers[0][1] + markers[1][1]) // 2)
            cv2.putText(frame, f'{span * 100:.1f} cm', mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)

        cv2.imshow(self.window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            with self.lock:
                self.markers.clear()
        elif key == ord('q'):
            rclpy.shutdown()

    # ------------------------------------------------------------- callbacks

    def on_info(self, msg):
        if self.intr is None:
            self.intr = Intrinsics(msg)
            if self.probe[0] < 0 or self.probe[1] < 0:
                self.probe = (msg.width // 2, msg.height // 2)
            self.get_logger().info(
                f'intrinsics: fx={self.intr.fx:.2f} fy={self.intr.fy:.2f} '
                f'cx={self.intr.cx:.2f} cy={self.intr.cy:.2f} '
                f'frame={self.intr.frame_id}')

    def on_color(self, msg):
        with self.lock:
            self.latest_color = image_to_numpy(msg)

    def on_depth(self, msg):
        if self.intr is None:
            return
        depth = image_to_numpy(msg)
        if depth.shape != (self.intr.height, self.intr.width):
            self.get_logger().warn(
                f'depth {depth.shape} does not match intrinsics '
                f'{(self.intr.height, self.intr.width)} - are you subscribed to '
                'aligned_depth_to_color and using the colour camera_info?',
                once=True)

        with self.lock:
            u, v = self.probe
        z = self.sample(depth, u, v)
        with self.lock:
            self.last_point = None if z is None else self.intr.deproject(u, v, z)

        if z is None:
            self.throttled_log(f'pixel ({u},{v}): no valid depth in ROI')
            return

        x, y, z = self.last_point
        out = PointStamped()
        out.header = msg.header
        out.point.x, out.point.y, out.point.z = float(x), float(y), float(z)
        self.point_pub.publish(out)
        self.dist_pub.publish(Float32(data=float(z)))
        self.throttled_log(
            f'pixel ({u},{v}) -> x={x:+.3f} y={y:+.3f} z={z:.3f} m '
            f'| expected error ~{self.depth_rms(z) * 1000:.1f} mm')

    # ----------------------------------------------------------------- maths

    def sample(self, depth, u, v):
        """Median of the non-zero depths in the ROI around (u, v), in metres."""
        return sample_depth(depth, u, v, self.roi_half, self.depth_scale)

    @staticmethod
    def depth_rms(z, baseline=0.050, subpixel=0.08, focal_px=645.0):
        """D435 stereo error model: error grows with the square of range."""
        return (z * z * subpixel) / (focal_px * baseline)

    @staticmethod
    def _span(markers):
        (_, _, x0, y0, z0), (_, _, x1, y1, z1) = markers
        return math.dist((x0, y0, z0), (x1, y1, z1))

    def _report_span(self):
        if len(self.markers) == 2:
            span = self._span(self.markers)
            self.get_logger().info(f'measured span: {span:.4f} m ({span * 100:.1f} cm)')

    def throttled_log(self, text):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_log >= self.get_parameter('log_period_sec').value:
            self.last_log = now
            self.get_logger().info(text)

    last_point = None


def main():
    rclpy.init()
    node = DepthMeasure()
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

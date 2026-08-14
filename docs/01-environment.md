# 01 — Environment: what was there, and what was broken

Before writing any code I checked what the machine actually had. Two things
came out of that: one missing piece, and one broken piece that changed the
design of the whole package.

## What I checked, and why

```bash
ls /opt/ros/                                  # which ROS distro
ls /opt/ros/humble/share/ | grep -i realsense # is the ROS wrapper installed?
apt list --installed | grep -i realsense      # is the SDK installed?
lsusb | grep -i intel                         # is the camera plugged in?
rs-enumerate-devices -s                       # does the SDK see it?
```

Results:

| Check | Result |
|---|---|
| ROS distro | Humble |
| librealsense SDK | `2.58.3` installed, plus `librealsense2-dkms` |
| `realsense-viewer` | present at `/usr/bin/realsense-viewer` |
| Camera on USB | `8086:0b3a` on **Bus 002** (USB 3 controller) |
| SDK sees camera | `RealSense D435IF`, serial `327122074022`, FW `5.15.0.2` |
| **`realsense2_camera`** | **absent** |

So the SDK half of the setup was complete and the ROS half did not exist. That
is the gap `docs/02` closes.

The bus number matters. Bus 002 is the USB 3 controller on this box; if the
camera enumerates on the USB 2 bus you are capped at lower resolutions and
frame rates, and the driver will warn about it. `lsusb -t` shows the tree if
you ever need to confirm the link speed.

## The numpy problem

The normal way to turn a `sensor_msgs/Image` into an array in Python is
`cv_bridge`. It did not work:

```
$ python3 -c "from cv_bridge import CvBridge; ..."
AttributeError: _ARRAY_API not found
```

Then the same class of failure from scipy, which is the usual source of the
KD-tree used for point cloud clustering:

```
$ python3 -c "from scipy.spatial import cKDTree; ..."
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
            Expected 96 from C header, got 88 from PyObject
```

### Why

Both are C extensions compiled against **numpy 1.x**. This machine has numpy
`2.2.6`. numpy 2.0 changed the size of `PyArray_Descr` — the struct behind
`numpy.dtype` — from 96 bytes to 88. Any extension module built against the old
header reads that struct at the wrong offsets, so the import either aborts
(`_ARRAY_API not found`) or refuses outright (`dtype size changed`). ROS 2
Humble targets Ubuntu 22.04, whose system numpy is 1.21, so every Humble binary
Python package on this machine has the same defect.

Note that `import cv_bridge` *appears* to succeed. The failure only surfaces
when you call `imgmsg_to_cv2`, because that is the call that reaches the boost
extension. A smoke test that only imports the module will lie to you.

### The two ways out

**A — downgrade numpy.** `pip install "numpy<2"` restores `cv_bridge`, scipy,
and every other Humble Python binary at once. This is the right long-term fix
if you plan to use more of the Humble Python ecosystem, and it is what most
Humble users run. The cost is that anything else on the machine wanting numpy 2
breaks instead.

**B — depend on neither.** Decode messages with `numpy.frombuffer` and
implement the point cloud maths directly.

I chose **B** for this package, so it runs on the machine as it stands today,
with no system-wide change made on your behalf. `msg_utils.py` is the entire
cost of that choice — about 100 lines, and it is worth reading, because writing
the decoder is how you learn what is actually inside these messages.

You can still take option A later; nothing in this package conflicts with it.

## What did work

- `numpy 2.2.6`
- `opencv-python 4.13.0` (pure Python bindings, independent of the ROS build) —
  used only for the optional viewer window, never for message conversion
- `rclpy` and all message packages

Next: [02 — Driver and topics](02-driver-and-topics.md)

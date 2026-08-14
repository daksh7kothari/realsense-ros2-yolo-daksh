# RealSense D435i depth analysis

A standalone colcon workspace for depth probing and point cloud analysis on an
Intel RealSense D435i, using ROS 2 Humble.

**Two ways to read the docs.** `01`–`08` teach the subject: how depth works, what
the pipeline does, why it is fast. `09`–`10` teach the process: how the work was
done, what went wrong, and which habits transfer to the next project. If you want
to learn the method rather than the code, start at `09`.

```
realsense/
├── docs/                      ← read these, in order
│   ├── 01-environment.md      what was on the machine, and what was broken
│   ├── 02-driver-and-topics.md  installing the wrapper, and the topic map
│   ├── 03-depth-fundamentals.md how depth is encoded, and how wrong it is
│   ├── 04-pixel-to-3d.md      deprojection, and measuring object size
│   ├── 05-cloud-pipeline.md   the five-stage pipeline, stage by stage
│   ├── 06-design-decisions.md why the code looks like this
│   ├── 07-tests.md            what the 56 tests pin down, and how to run them
│   ├── 08-performance.md      finding and fixing the 0.4 Hz bottleneck
│   ├── 09-how-this-was-built.md  the process, in order, mistakes included
│   ├── 10-method.md           the transferable habits and commands
│   ├── 11-commands-and-videos.md  cheatsheet + video primer
│   └── 12-yolo-detection.md   node 3: YOLO box distance + framing offset
└── src/realsense_depth/
    ├── realsense_depth/
    │   ├── msg_utils.py       numpy-only Image / PointCloud2 decoders
    │   ├── depth_measure.py   node 1: probe a pixel, get metric XYZ
    │   ├── cloud_pipeline.py  node 2: filter → cluster → measure
    │   └── yolo_measure.py    node 3: detect box, get distance + offset%
    ├── launch/depth_analysis.launch.py
    └── config/params.yaml
```

## Hardware and software as found

| Item | Value |
|---|---|
| Camera | D435IF, serial `327122074022`, firmware `5.15.0.2`, USB 3 |
| SDK | librealsense `2.58.3` (apt) |
| ROS | Humble on Ubuntu 22.04 |
| ROS wrapper | `realsense2_camera` 4.58.3, with `ros-humble-librealsense2` 2.58.3 — same version as the standalone SDK, so no mismatch |

Verified running against the camera: driver publishes at 30 Hz, both nodes
subscribe, cluster detection and plane removal confirmed live.

## Quick start

```bash
# every shell
source /opt/ros/humble/setup.bash
source /home/daksh/realsense/install/setup.bash

# camera + both nodes
ros2 launch realsense_depth depth_analysis.launch.py

# with the click-to-measure window
ros2 launch realsense_depth depth_analysis.launch.py viewer:=true

# + YOLO box detection (needs `pip3 install ultralytics` first, see docs/12)
ros2 launch realsense_depth depth_analysis.launch.py yolo:=true

# nodes only, against a camera you already started
ros2 launch realsense_depth depth_analysis.launch.py launch_camera:=false
```

## What each node gives you

`depth_measure`

- `~/point` (`geometry_msgs/PointStamped`) — probe position in metres
- `~/distance` (`std_msgs/Float32`) — range to the probe
- with `viewer:=true`: left-click moves the probe, right-click drops a marker,
  two markers print the 3D distance between them, `c` clears, `q` quits

`cloud_pipeline`

- `~/filtered` (`PointCloud2`) — cropped, downsampled, plane removed
- `~/plane` (`PointCloud2`) — the points that were removed
- `~/clusters` (`MarkerArray`) — a labelled box per detected object

`yolo_measure` (opt-in, `yolo:=true` — see `docs/12-yolo-detection.md`)

- `~/point` / `~/distance` — same as `depth_measure`, but at the centre of the
  largest YOLO-detected box instead of a manual probe
- `~/offset_percent` (`std_msgs/Float32`) — how far the box sits from frame
  centre, signed: negative left, positive right, ±100 at the frame edge
- `~/detected` (`std_msgs/Bool`) — whether a box was found this frame

View in RViz2 with Fixed Frame `camera_link`.

## Tests

```bash
cd /home/daksh/realsense
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 colcon test --packages-select realsense_depth
colcon test-result --verbose        # 72 tests, 0 failures
```

No camera needed — everything runs on synthetic messages. See `docs/07-tests.md`.

## Known environment issues

Most from the same family: system Python packages built against older
dependencies than the ones installed.

1. **numpy 2.2.6 vs Humble binaries.** `cv_bridge` and `scipy` are compiled
   against numpy 1.x and raise on use. This package depends on neither — it
   decodes messages with `numpy.frombuffer` instead. `docs/01`.
2. **pytest 6.x vs a user-site anyio plugin.** Needs
   `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` when running tests. `docs/07`.
3. **The ROS apt signing key was truncated to 20 bytes**, which is why the
   wrapper would not install. Replaced with the official key, fingerprint
   `C1CF 6E31 E6BA DE88 68B1 72B4 F42E D6FB AB17 C654`. `docs/02`.
4. **System `matplotlib` (apt, numpy 1.x ABI) breaks on import**, same numpy2
   issue as #1 — `ultralytics` pulls it in transitively even though
   `yolo_measure` never calls it directly. Fixed with
   `pip3 install --user -U matplotlib` (pulls a numpy2-built wheel into user
   site, which shadows the broken apt one). `docs/12`.
5. **`colcon build` fails with `error: option --uninstall not recognized`**
   on `setuptools>=71` — that release dropped `setup.py develop --uninstall`,
   which colcon's ament_python build calls on every rebuild. Fixed by pinning
   `pip3 install --user "setuptools<71"` (tested against 70.3.0). Confirmed
   `yolo_measure` live against the D435i after both fixes — see `docs/12`.

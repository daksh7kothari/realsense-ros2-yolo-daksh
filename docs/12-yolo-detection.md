# YOLO box detection: distance + framing offset

A third node, `yolo_measure`, sits alongside `depth_measure` and
`cloud_pipeline`. It runs a YOLO detector on the colour stream, picks the
largest detected box, and reports two things about it every depth frame:
straight-line distance to its centre, and how far off-centre it sits in the
frame, as a signed percentage.

## What model this expects

`largestboxmodel(1).pt` (dropped at the repo root) is an Ultralytics
checkpoint — inspecting its pickle shows `ultralytics==8.4.69`, task
`detect`, base model `yolo26n`, one class (`item`), trained at `imgsz=640` for
150 epochs on a ~2000-image box dataset. Any Ultralytics detection `.pt` works
the same way; swap `model_path` if you train a new one.

## How it works

```
colour image ──► YOLO inference ──► largest box (x1,y1,x2,y2,conf)
                                          │
                                          ▼
                                    box centre (u, v)
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                          ▼
        aligned depth, same ROI-median            offset_percent(u):
        sampling as depth_measure.py               (u - width/2) / (width/2) * 100
                     │                                          │
                     ▼                                          ▼
        ~/point, ~/distance                          ~/offset_percent
```

Two things worth being precise about, because both were explicit judgment
calls rather than obvious defaults:

**Distance** samples an 11x11 pixel window centred on the box (median of the
non-zero depths in it), exactly like `depth_measure`'s mouse-probe. Not the
whole box — a wide median gets pulled toward the background at the box's
silhouette edges.

**Offset is pixel framing, not physical yaw.** `offset_percent` says how far
the box's centre sits from the image's horizontal centre — 0% is dead centre,
negative is left, positive is right, ±100% is the frame edge. It does **not**
mean "the box is turned 20°" — this model gives an axis-aligned box, not a
rotated one, so there's no pose/yaw information to extract from a single
frame. If you need the box's actual physical tilt (is it turned away from the
camera?), that requires comparing depth across the box's left and right edges,
which `yolo_measure` doesn't currently do — flag it if you want that added.

Inference runs on the colour callback, throttled to `infer_period_sec`
(default 5 Hz) — a CPU-only nano model can't keep up with the RealSense's
30 Hz, and there's no point trying. The most recent detected box is reused for
every depth frame that arrives between inferences.

## Install

`ultralytics` (and the `torch` it pulls in) are pip-only — there's no rosdep
key for them, so they're not in `package.xml`, and the launch file's `yolo`
argument defaults to `false` so nothing breaks if they're absent.

```bash
# system python3.10, the same interpreter rclpy runs under —
# NOT the .venv (that's py3.11, no ROS on its path)
pip3 install --user ultralytics
```

That alone isn't enough on this machine — two more numpy2/toolchain fixes
were needed before it actually ran, both from the same family as the
`cv_bridge`/`scipy` issue in the README:

```bash
# ultralytics pulls in matplotlib transitively; the apt one is built
# against numpy 1.x and raises ImportError on import. A pip wheel shadows it.
pip3 install --user -U matplotlib

# setuptools>=71 dropped `develop --uninstall`, which colcon's ament_python
# build calls on every rebuild -> "error: option --uninstall not recognized".
pip3 install --user "setuptools<71"
```

Confirmed working after both: `colcon build` succeeds, and `yolo_measure`
runs live against the D435i — steady detections around conf 0.9+, distance
agreeing with `depth_measure`'s own centre-pixel probe to the millimetre.

This machine tested as CPU-only (`nvidia-smi` not found), so `device: cpu` is
the default. If you do have a CUDA GPU, set the `device` param to `0`.

## Running it

```bash
source /opt/ros/humble/setup.bash
source /home/daksh/realsense/install/setup.bash

# camera + all three nodes
ros2 launch realsense_depth depth_analysis.launch.py yolo:=true

# + the OpenCV window (box, centre marker, distance/offset label)
ros2 launch realsense_depth depth_analysis.launch.py yolo:=true viewer:=true

# just this node, camera already running
ros2 run realsense_depth yolo_measure --ros-args --params-file src/realsense_depth/config/params.yaml
```

Watch it live:

```bash
ros2 topic echo /yolo_measure/distance
ros2 topic echo /yolo_measure/offset_percent
ros2 topic echo /yolo_measure/detected
```

## Topics and params

| Publishes | Type | Meaning |
|---|---|---|
| `~/point` | `geometry_msgs/PointStamped` | 3D position of the box centre, metres |
| `~/distance` | `std_msgs/Float32` | range to box centre, metres |
| `~/offset_percent` | `std_msgs/Float32` | signed horizontal offset, -100 (left edge) to +100 (right edge) |
| `~/detected` | `std_msgs/Bool` | whether a box was found in the latest inference |

Key params (`config/params.yaml`, under `yolo_measure:`): `model_path`,
`conf_thres` (default 0.5), `imgsz` (640), `device` (`cpu`), `infer_period_sec`
(0.2 — raise it if the CPU can't keep up, lower it if you have a GPU),
`roi_half`, `viewer`.

## Tests

`test/test_yolo_measure.py` covers the pure box-picking logic
(`pick_largest_box`, `boxes_from_result`), the offset math, and the
depth-sampling/publish path — all against synthetic messages, no camera or
model weights needed. `model_path` in the test fixture points at a file that
doesn't exist, so `YoloMeasure._load_model` fails closed to `self.model =
None` exactly as it would with `ultralytics` missing; the tests inject boxes
directly onto `node.latest_box` rather than running real inference.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  src/realsense_depth/test/test_yolo_measure.py -v
```

## Videos to build background

**YOLO / Ultralytics basics**
- [Train your first YOLO model — Ultralytics Academy](https://academy.ultralytics.com/courses/train-your-first-yolo) — train/validate/export a custom model from the CLI, the same library this node calls at inference time
- [How to Train Ultralytics YOLO26 Model on Custom Dataset (Google Colab)](https://www.youtube.com/watch?v=7lZa3Yi2kbo) — matches this model's base (`yolo26n`) exactly
- [Train the New YOLO26 on Your Custom Dataset with Ultralytics](https://www.youtube.com/watch?v=g6K9912-Acw)

**YOLO + RealSense specifically**
- [Yolov8 and ROS: Object detection and localization using Yolov8 and depth camera](https://www.youtube.com/watch?v=vchM3bcElYI) — detect with YOLO, then read position from the depth image, the same pattern `yolo_measure.py` follows
- [Identify and Measure precisely Objects distance — Deep Learning + Intel RealSense](https://www.youtube.com/watch?v=_gzcp8dURbU)

**Reference code** (not videos, but directly on-topic)
- [andreasHovaldt/yolov8_ros2](https://github.com/andreasHovaldt/yolov8_ros2) — ROS2 package, YOLOv8 segmentation + D435
- [bunyaminbingol/Yolo-Object-Detection-and-Distance-Measurement-With-Intel-Realsense-Camera](https://github.com/bunyaminbingol/Yolo-Object-Detection-and-Distance-Measurement-With-Intel-Realsense-Camera)

For the ROS2/RViz fundamentals this node reuses (publishers, params, launch
files), see the video list already in `docs/11-commands-and-videos.md` §4 —
no need to repeat it here.

## Open questions for you

1. Confirmed: offset is pixel framing (box position in the picture), not the
   box's physical yaw. Say the word if you actually want a depth-based tilt
   estimate later — it's a different computation (left-edge vs right-edge
   depth), not a small tweak to this one.
2. `model_path` defaults to the absolute path of the dropped file. Fine for
   this single-machine setup; flag it if this ever needs to run somewhere
   `largestboxmodel(1).pt` isn't at that exact path.
3. `infer_period_sec: 0.2` (5 Hz cap) turned out not to be the real limit —
   measured against the camera, this CPU actually completes a `yolo26n`
   inference roughly every 0.6-0.9s (~1.2-1.6 Hz), so the throttle never
   engages. Left the default as-is since it's harmless (a ceiling, not a
   floor) — lower it further only if you want to deliberately cut CPU load
   below what inference alone already uses.

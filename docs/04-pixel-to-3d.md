# 04 — From a pixel to a point in metres

Depth alone gives you Z: how far away the thing is along the optical axis. It
does not tell you where it is left/right or up/down in metres. Turning a pixel
plus a depth into a full 3D point is called **deprojection**, and it needs the
camera intrinsics.

## The intrinsics

`CameraInfo.k` is the 3×3 pinhole matrix, row-major:

```
K = [ fx   0  cx ]        fx = k[0]    cx = k[2]
    [  0  fy  cy ]        fy = k[4]    cy = k[5]
    [  0   0   1 ]
```

- `fx`, `fy` — focal length **in pixels**, not millimetres. Encodes the field of
  view together with the sensor size.
- `cx`, `cy` — the principal point, where the optical axis crosses the sensor.
  Close to the image centre but never exactly, because of assembly tolerance.

`Intrinsics` in `msg_utils.py` pulls these out once, on the first `CameraInfo`
message, and caches them. They are static per stream configuration, so
re-parsing them 30 times a second would be waste.

## The projection, and its inverse

A pinhole camera maps a 3D point to a pixel:

```
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

Solve for X and Y, given that depth handed you Z:

```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = raw_depth * 0.001
```

That is `Intrinsics.deproject()`, in full:

```python
def deproject(self, u, v, z):
    return ((u - self.cx) * z / self.fx,
            (v - self.cy) * z / self.fy,
            z)
```

Note that Z has to come *first*. You cannot recover X and Y without it — one
image pixel corresponds to an entire ray through space, and depth is what picks
the point on that ray. This is exactly the ambiguity a depth camera exists to
remove.

## The coordinate frame

The result is in the **optical frame** (`camera_color_optical_frame`), which
follows the computer-vision convention:

```
+X  right
+Y  down          ← not up
+Z  forward, along the optical axis
```

ROS's own body convention (`camera_link`) is different: +X forward, +Y left, +Z
up. The driver publishes the static transform between them, so if you want
robot-frame coordinates, do not hand-roll the rotation — let TF2 do it:

```python
from tf2_ros import Buffer, TransformListener
# ... then transform the PointStamped that depth_measure publishes
```

`depth_measure` publishes a `PointStamped` with the correct `frame_id` copied
from the depth message header, precisely so TF2 can consume it directly.

## Measuring an object's size

This is the part that is easy to get wrong.

**Wrong:** measure the object's width in pixels, then scale by depth. That is
only valid if the object is exactly fronto-parallel — both edges at the same Z.
Tilt it and the answer drifts, with no warning.

**Right:** deproject both endpoints independently, then take the Euclidean
distance between the two 3D points.

```python
p0 = intr.deproject(u0, v0, z0)
p1 = intr.deproject(u1, v1, z1)
size = math.dist(p0, p1)
```

Each endpoint carries its own depth, so an object tilted away from the camera
is measured correctly. This is `DepthMeasure._span()`, and it is what the
viewer's right-click markers drive.

### Using it

```bash
ros2 launch realsense_depth depth_analysis.launch.py viewer:=true
```

- **left-click** — move the probe; the live range is drawn at the crosshair
- **right-click** — drop a measurement marker at the probe's current 3D point
- **two markers** — the span is drawn on the line between them and logged in
  metres and centimetres
- **c** clears the markers, **q** quits

Headless, without the window, set the probe from parameters instead:

```bash
ros2 run realsense_depth depth_measure --ros-args \
  -p probe_u:=424 -p probe_v:=240 -p roi_half:=7
ros2 topic echo /depth_measure/point
```

### Reading the result honestly

A span between two points at 1.5 m carries roughly 6 mm of uncertainty per
endpoint (see `docs/03`), so about ±1 cm on the span. Quoting the measurement to
the millimetre is quoting noise. Increase `roi_half` to average more, move the
camera closer, or accept the bound.

Next: [05 — The point cloud pipeline](05-cloud-pipeline.md)

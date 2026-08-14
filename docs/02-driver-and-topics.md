# 02 — The driver and its topics

## Installing the wrapper

`librealsense` is the vendor SDK. `realsense2_camera` is the ROS 2 node that
wraps it and publishes topics. You have the first and not the second:

```bash
sudo apt install ros-humble-realsense2-camera ros-humble-realsense2-description
```

**Installed here: wrapper 4.58.3.** It pulled in `ros-humble-librealsense2` at
**2.58.3** — the same version as your standalone SDK, so the version-mismatch
worry did not materialise.

### The signing key had to be fixed first

The install failed with 404s on every package, and `apt-get update` explained
why:

```
NO_PUBKEY F42ED6FBAB17C654
```

`/usr/share/keyrings/ros-archive-keyring.gpg` was **20 bytes** — truncated, so
apt could not verify the repository, fell back to a stale package index, and
asked for `.deb` versions that no longer existed on the mirror. The fix:

```bash
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o ros.key
gpg --show-keys --with-fingerprint ros.key    # verify BEFORE installing it
sudo cp ros.key /usr/share/keyrings/ros-archive-keyring.gpg
sudo apt-get update
```

Check the fingerprint against the ID apt named — `C1CF 6E31 E6BA DE88 68B1
72B4 F42E D6FB AB17 C654` ends in `F42ED6FBAB17C654`, so it is the key apt was
asking for. A repository signing key is a trust anchor; installing one without
checking its fingerprint defeats the point of signing.

If the node ever crashes at startup with a version complaint, the fix is to
build `realsense-ros` from source against your SDK instead:

```bash
cd ~/realsense/src
git clone -b ros2-development https://github.com/IntelRealSense/realsense-ros.git
cd ~/realsense && colcon build --symlink-install
```

Your firmware is `5.15.0.2`; librealsense 2.58 prefers `5.16.x`. A mismatch
warning at startup is normal and not fatal. Update through `realsense-viewer`
if you want it silent.

## Launching

```bash
ros2 launch realsense2_camera rs_launch.py \
  depth_module.depth_profile:=848x480x30 \
  rgb_camera.color_profile:=848x480x30 \
  align_depth.enable:=true \
  pointcloud.enable:=true
```

`depth_analysis.launch.py` in this package passes exactly these four arguments,
so you do not have to remember them.

**Why 848x480.** That is the D435's native depth resolution — the resolution
the depth ASIC actually correlates at. Asking for 640x480 does not give you a
cleaner image; it gives you the same image rescaled, with the aspect ratio
cropped and some accuracy discarded. Ask for the native size and downsample
later if you need to.

**Why `align_depth.enable`.** The depth and colour sensors are physically
different lenses about 15 mm apart. Pixel (400, 240) in the colour image and
pixel (400, 240) in the raw depth image are *not* the same point in the world.
`align_depth` reprojects depth into the colour camera's pixel grid, which is
what makes "what is the depth at this colour pixel?" a well-posed question. It
costs CPU, and it is non-negotiable for the measurement task.

**Why `pointcloud.enable`.** It is what feeds `cloud_pipeline`. The driver
builds the cloud on the GPU/CPU internally, which is faster than you rebuilding
it from the depth image in Python.

## The topic map

Default namespace is `/camera/camera` — the wrapper nests `camera_namespace`
inside `camera_name`, and both default to `camera`. Surprising the first time.
Confirmed against the running driver: every name below matched.

| Topic | Type | Notes |
|---|---|---|
| `/camera/camera/depth/image_rect_raw` | `Image` | `16UC1`, in the **depth** optical frame |
| `/camera/camera/aligned_depth_to_color/image_raw` | `Image` | `16UC1`, in the **colour** optical frame ← use this |
| `/camera/camera/color/image_raw` | `Image` | `rgb8` |
| `/camera/camera/color/camera_info` | `CameraInfo` | intrinsics for the aligned depth |
| `/camera/camera/depth/color/points` | `PointCloud2` | XYZRGB, 32 bytes per point |
| `/camera/camera/imu` | `Imu` | D435i only, if `unite_imu_method` is set |

The pairing rule that trips people up: **aligned depth goes with the *colour*
`camera_info`, not the depth one.** Aligned depth lives on the colour pixel
grid, so the colour intrinsics are the ones that deproject it correctly.
`depth_measure` defaults to exactly this pair, and warns if the image size and
the intrinsics disagree.

## QoS: the silent-failure trap

RealSense image topics are published **BEST_EFFORT**. The rclpy default for a
subscription is **RELIABLE**. A RELIABLE subscriber and a BEST_EFFORT publisher
are incompatible under DDS, so the connection is never made — and you get *no
error*. `ros2 topic hz` shows the publisher is alive, your callback never runs,
and nothing anywhere says why.

Hence, in both nodes:

```python
from rclpy.qos import qos_profile_sensor_data
self.create_subscription(Image, topic, self.on_depth, qos_profile_sensor_data)
```

If you ever write a RealSense subscriber that "does nothing", check this first.

## Verifying before you write code

```bash
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw   # ~30
ros2 topic echo /camera/camera/color/camera_info --once         # k matrix
ros2 run rviz2 rviz2   # Fixed Frame camera_link, add PointCloud2
```

Next: [03 — Depth fundamentals](03-depth-fundamentals.md)

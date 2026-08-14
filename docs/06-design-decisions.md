# 06 — Design decisions

Why the code looks the way it does. Each of these was a fork in the road.

## No cv_bridge

**Forced by the environment.** `cv_bridge` on this machine raises
`AttributeError: _ARRAY_API not found` under numpy 2.2 (`docs/01`).

Replaced by ~20 lines in `msg_utils.image_to_numpy`. The one subtlety is `step`:

```python
rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
used = msg.width * channels * dtype.itemsize
arr = rows[:, :used].reshape(msg.height, -1).view(dtype)
```

`step` is the **row stride in bytes**, and it is allowed to be larger than
`width × channels × itemsize` when rows are padded for alignment. Reshaping
straight to `(height, width)` and ignoring `step` works on most RealSense
streams and shears the image on any padded one — a bug that appears only on
certain resolutions, which is the worst kind. So the decode reshapes by `step`
first and slices the used bytes out of each row. There is a unit check for
exactly this case in the verification I ran.

Cost of not using cv_bridge: no automatic colour conversion. The viewer does
its own single `cvtColor(RGB→BGR)`, which is honest about the fact that
`image_raw` is `rgb8` and OpenCV wants BGR.

## No PCL, no Open3D

PCL has no maintained Python bindings — `python-pcl` has been dead for years,
and the alternative is writing the node in C++. Open3D is not installed and is
a heavy dependency to add for four operations.

All five stages are 120 lines of numpy, and the numpy versions are *readable*
in a way that the PCL calls are not: `pcl::SACSegmentation` is a black box,
whereas the RANSAC loop in `remove_dominant_plane` shows you the entire
algorithm in ten lines. For learning, that trade is worth taking.

Where this stops being the right call: if you need to run at 30 Hz on the full
unfiltered cloud, or you want ICP registration, normal estimation, or surface
reconstruction. Then write the node in C++ against PCL — `pcl_conversions`
bridges `PointCloud2` ↔ `pcl::PointCloud` in one call.

## No scipy KD-tree

Same ABI break (`docs/01`). Clustering runs on the voxel lattice with a dict
instead — see `docs/05` stage 4. Since the cloud is voxelised one stage earlier,
this is close to what PCL's Euclidean clustering computes anyway, and it avoids
building a tree per frame.

## `qos_profile_sensor_data` on every subscription

RealSense publishes BEST_EFFORT; rclpy subscribes RELIABLE by default;
mismatched QoS means the subscription silently never connects. No error, no
warning, callback never fires. This is the single most common reason a
RealSense ROS 2 node "does nothing", and it costs one import to avoid.

## camera_info cached, not synchronised

Intrinsics are constant for a given stream configuration, so `message_filters`
time synchronisation between depth and `CameraInfo` would be machinery for no
benefit — and one more package to depend on. `on_info` stores the first message
and ignores the rest; `on_depth` returns early until it exists.

The one thing this cannot catch is a mid-run resolution change. So `on_depth`
compares the image shape against the cached intrinsics and warns once if they
disagree — which also catches the much more likely mistake of pairing raw
(unaligned) depth with the colour `camera_info`.

## `None` instead of `0.0` for invalid depth

`sample()` returns `None` when every pixel in the ROI is invalid. Returning
`0.0` would be a plausible-looking number that propagates silently into a
measurement. `None` makes the caller decide.

Same reasoning for `plane_min_ratio`: when there is no real plane, the stage
declines rather than returning its best guess.

## Parameters for everything, in one YAML

Every threshold is a ROS parameter with a documented default in
`config/params.yaml`, so tuning is `ros2 param set` at runtime rather than an
edit-and-rebuild loop. Point cloud work is mostly parameter tuning; making that
loop fast is the highest-leverage thing in the whole package.

## Two nodes, not one

`depth_measure` and `cloud_pipeline` answer different questions — "how far is
that pixel?" versus "what objects are in front of me?" — and have different
costs. Splitting them means you can run the cheap one alone, restart one
without disturbing the other, and put them on different machines. The launch
file starts both; `launch_camera:=false` lets you attach to a camera someone
else started.

## Verification before hand-off

The package was checked before you saw it, rather than assumed to work:

- `image_to_numpy` against a deliberately padded `step` — the failure mode most
  likely to hide
- `PointCloud2` encode/decode round-trip, asserting equality
- all five pipeline stages against a synthetic scene with known ground truth: a
  plane at z = 1.5 m, two boxes of known size, plus far-field noise
- `colcon build` clean

Recovered extents were `0.15 × 0.15 × 0.20 m` and `0.10 × 0.15 × 0.15 m` against
ground truth of exactly those, and the plane was removed with the boxes intact.

Then verified live, once the wrapper was installed: every assumed topic name
matched the running driver, the QoS pairing connected, both nodes produced
sensible output on real data, and plane removal and clustering worked on the
scene in front of the camera. The suite has since grown to 56 tests, covering
the vectorised lattice helpers added during the optimisation pass.

## Optimised only after measuring

The first live run was correct and ran at 6 Hz against a 30 Hz driver. Rather
than guess, each stage was timed in isolation: one stage was 93% of the frame.
Fixing it and two others gave a 7.7× speedup with identical output and no test
changes. The full account, including a detour where a stale `install/` copy made
the fix look ineffective, is in `docs/08-performance.md`.

The general shape of all three fixes was the same — replace a numpy call that
falls back to a slow path (`np.unique(..., axis=0)`, `np.add.at`, per-column
buffer slicing) with one that stays vectorised (`np.bincount`, flat-key unique,
a structured dtype). None of it changed the algorithms.

## What I would add next

- **Oriented bounding boxes** via PCA on each cluster, for dimensions that do
  not depend on camera orientation (`docs/05`, stage 5).
- **Temporal filtering.** The driver exposes hole-filling and temporal filters
  (`temporal_filter.enable`, `hole_filling_filter.enable`) that cut jitter
  substantially for a static scene.
- **TF2 output**, publishing cluster centroids in a robot frame rather than the
  optical frame.
- **Tracking**, associating clusters across frames so an object keeps its ID.

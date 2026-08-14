# Command cheatsheet + video primer

Practical companion to `docs/01`–`10`. Those explain *why*; this is *what to type*.
Two parts: (1) how to run every file in this repo, (2) a default `ros2` CLI
cookbook, (3) YouTube videos to build background if any of this is unfamiliar.

## 0. Every shell, first

```bash
source /opt/ros/humble/setup.bash
source /home/daksh/realsense/install/setup.bash
```

Nothing below works without this. If you edited code, rebuild first:

```bash
cd /home/daksh/realsense
colcon build --packages-select realsense_depth
source install/setup.bash   # re-source after every build
```

## 1. How to run every file

### `src/realsense_depth/launch/depth_analysis.launch.py`
The normal entry point — starts the camera driver + both nodes.

```bash
ros2 launch realsense_depth depth_analysis.launch.py                    # camera + both nodes
ros2 launch realsense_depth depth_analysis.launch.py viewer:=true       # + OpenCV click-to-measure window
ros2 launch realsense_depth depth_analysis.launch.py launch_camera:=false  # nodes only, camera already running elsewhere
```

### `realsense_depth/depth_measure.py`
Not run directly as a script (it imports `realsense_depth.msg_utils` as a
package, and needs a `rclpy.init()` context). Run it as an installed node:

```bash
ros2 run realsense_depth depth_measure
ros2 run realsense_depth depth_measure --ros-args -p viewer:=true
ros2 run realsense_depth depth_measure --ros-args --params-file src/realsense_depth/config/params.yaml
```

### `realsense_depth/cloud_pipeline.py`
Same deal:

```bash
ros2 run realsense_depth cloud_pipeline
ros2 run realsense_depth cloud_pipeline --ros-args -p z_max:=2.0
```

### `realsense_depth/yolo_measure.py`
Same deal, opt-in via the launch file's `yolo:=true` (needs `ultralytics`
installed — `docs/12-yolo-detection.md`):

```bash
ros2 run realsense_depth yolo_measure
ros2 run realsense_depth yolo_measure --ros-args -p viewer:=true
```

### `realsense_depth/msg_utils.py`
Not a node — a library (`Intrinsics`, `image_to_numpy`, `pointcloud2_to_xyz`,
`xyz_to_pointcloud2`) imported by both nodes and by the tests. Nothing to run;
exercised indirectly via `test_msg_utils.py`.

### `realsense_depth/__init__.py`
Empty package marker. Never run directly.

### `config/params.yaml`
Not a script — parameter file. Loaded automatically by the launch file, or
pass explicitly with `--ros-args --params-file <path>` as shown above.

### Test files (`test/test_*.py`)
Synthetic messages only, no camera needed.

```bash
# whole suite, via colcon (what CI/the README uses)
cd /home/daksh/realsense
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 colcon test --packages-select realsense_depth
colcon test-result --verbose

# one file directly with pytest (faster feedback while editing)
cd /home/daksh/realsense/src/realsense_depth
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_depth_measure.py -v
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_cloud_pipeline.py -v
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_msg_utils.py -v

# one test function
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_cloud_pipeline.py -k remove_plane -v
```

`test_helpers.py` is not a test file itself (no `test_` functions) — it's
shared fixtures/builders imported by the other three.

## 2. Default `ros2` CLI cookbook

Run these while `depth_analysis.launch.py` is up, in another sourced shell.

**Discovery**
```bash
ros2 node list                        # running nodes
ros2 node info /depth_measure         # its pubs/subs/params/services
ros2 pkg list | grep realsense        # confirm the package is visible
ros2 pkg prefix realsense_depth       # where it's installed
```

**Topics**
```bash
ros2 topic list                                   # all topics
ros2 topic list -t                                # with message types
ros2 topic echo /depth_measure/point              # watch messages live
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw   # publish rate
ros2 topic bw /camera/camera/depth/color/points   # bandwidth
ros2 topic info /depth_measure/distance           # type + pub/sub counts
ros2 topic pub /some/topic std_msgs/msg/Float32 "{data: 1.0}"   # manual publish
```

**Parameters**
```bash
ros2 param list                                   # per-node, run with node name
ros2 param list /cloud_pipeline
ros2 param get /cloud_pipeline z_max
ros2 param set /cloud_pipeline z_max 2.0           # live override
ros2 param dump /cloud_pipeline                    # dump current values to yaml
```

**Introspection / debugging**
```bash
ros2 doctor                     # sanity-check the ROS install/environment
ros2 doctor --report            # verbose version
rqt_graph                       # visual node/topic graph
rqt                             # generic Qt inspector, many plugins
```

**Recording / playback**
```bash
ros2 bag record -a                              # record every topic
ros2 bag record /depth_measure/point /depth_measure/distance
ros2 bag play <bag_folder>                      # replay later, no camera needed
ros2 bag info <bag_folder>
```

**Visualization**
```bash
rviz2                           # Fixed Frame: camera_link
                                 # Add > By topic > /cloud_pipeline/filtered (PointCloud2)
                                 # Add > By topic > /cloud_pipeline/clusters (MarkerArray)
```

**Build / workspace**
```bash
colcon build                                     # whole workspace
colcon build --packages-select realsense_depth   # just this package
colcon build --symlink-install                   # edits to .py take effect without rebuild
colcon list                                      # packages colcon sees
rosdep install --from-paths src --ignore-src -y  # install missing deps from package.xml
```

## 3. Project topic/parameter reference

From `config/params.yaml` and the node source — for when a command above needs a real name.

| Node | Publishes | Type |
|---|---|---|
| `depth_measure` | `~/point` | `geometry_msgs/PointStamped` |
| `depth_measure` | `~/distance` | `std_msgs/Float32` |
| `cloud_pipeline` | `~/filtered` | `sensor_msgs/PointCloud2` |
| `cloud_pipeline` | `~/plane` | `sensor_msgs/PointCloud2` |
| `cloud_pipeline` | `~/clusters` | `visualization_msgs/MarkerArray` |
| `yolo_measure` | `~/point` | `geometry_msgs/PointStamped` |
| `yolo_measure` | `~/distance` | `std_msgs/Float32` |
| `yolo_measure` | `~/offset_percent` | `std_msgs/Float32` |
| `yolo_measure` | `~/detected` | `std_msgs/Bool` |

`~/` resolves to the node name, e.g. `/depth_measure/point`.

Key tunable params: `depth_measure.viewer`, `depth_measure.roi_half`,
`cloud_pipeline.z_min`/`z_max`, `cloud_pipeline.voxel_leaf`,
`cloud_pipeline.remove_plane`. Full list in `config/params.yaml`.

## 4. Videos to build background

Picked for direct relevance to this repo's stack: ROS2 pub/sub + launch +
params (what both nodes are built from), RViz2 + markers (how to see the
output), and the RealSense ROS wrapper (what `depth_analysis.launch.py`
launches under the hood). I did not include a RANSAC-specific video — search
only turned up papers, not a solid walkthrough; `docs/05-cloud-pipeline.md`
covers the algorithm as implemented here instead.

**ROS2 fundamentals**
- [ROS2 Tutorial - ROS2 Humble 2H50 (Crash Course)](https://www.youtube.com/watch?v=Gg25GfA456o) — nodes, topics, params, launch files in one sitting
- [ROS2 Tutorials — ROS2 Humble For Beginners (playlist)](https://www.youtube.com/playlist?list=PLLSegLrePWgJudpPUof4-nVFHGkB62Izy)
- [ROS2 Tutorials #3: How to create a ROS2 Workspace](https://www.youtube.com/watch?v=dPn-KwrJ9eo) — maps directly to this repo's `src/`/`build/`/`install/` layout
- [ROS2 Basics #11 - Writing a Simple Publisher and Subscriber (Python)](https://www.youtube.com/watch?v=eqfoy2ctixE) — same shape as `depth_measure.py`'s publishers

**RealSense + ROS2**
- [RealSense D435i + Docker: Full ROS2 Setup](https://www.youtube.com/watch?v=Ou6S5ispJKI) — driver bring-up, comparable to what `depth_analysis.launch.py` wraps

**Visualization**
- [ROS2 RViz Part-01 Tutorial](https://www.youtube.com/watch?v=WA3ynlo30vw) — general RViz2 navigation
- [ROS Developers LIVE-Class #24: How to create basic markers in ROS Rviz](https://www.youtube.com/watch?v=5pGzW-M6iGQ) — same `Marker`/`MarkerArray` types `cloud_pipeline.py` publishes on `~/clusters`

## Open questions for you

1. Want a video on the numpy-only point cloud math itself (voxel grid, RANSAC,
   connected components) or is `docs/05`/`06` enough on that front?
2. Any topic above you already know solid — trim it, or keep as reference?
3. Should this file also cover `rqt_plot` / `plotjuggler` for graphing
   `~/distance` over time, or is that out of scope for now?

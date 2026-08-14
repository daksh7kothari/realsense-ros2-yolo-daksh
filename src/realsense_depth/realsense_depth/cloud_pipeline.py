"""Classic five-stage point cloud pipeline, written with numpy only.

The stage order is the point of this node. Each stage is cheap because the
stage before it threw work away:

    1. passthrough   crop the depth range, killing the far noise that dominates
                     a stereo camera's error budget
    2. voxel grid    collapse the cloud onto a 3D lattice, ~300k points -> ~20k
    3. RANSAC plane  find and remove the dominant plane (table, floor, wall)
    4. clustering    connected components over occupied voxels, one per object
    5. measurement   centroid and axis-aligned extents per cluster

PCL is not used: there are no maintained Python bindings for it, and the numpy
version is short enough to read. scipy's cKDTree would be the usual choice for
stage 4, but it is ABI-broken against this machine's numpy, so connectivity is
computed on the voxel lattice instead - which is what PCL's own
``EuclideanClusterExtraction`` approximates anyway once the cloud is voxelised.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

from realsense_depth.msg_utils import pointcloud2_to_xyz, xyz_to_pointcloud2

def neighbour_offsets(gap):
    """Lattice offsets within `gap` cells, origin excluded. gap=1 gives the 26."""
    span = range(-gap, gap + 1)
    return np.array([(dx, dy, dz)
                     for dx in span for dy in span for dz in span
                     if (dx, dy, dz) != (0, 0, 0)], dtype=np.int64)


def voxel_packer(idx, pad=0):
    """Return a function packing integer voxel coordinates into one int64 key.

    The origin and extents are fixed up front and padded, so a neighbour lookup
    that steps `pad` cells outside the occupied set still packs to a key that
    cannot collide with a real one.
    """
    origin = idx.min(axis=0) - pad
    dims = (idx.max(axis=0) + pad) - origin + 1

    def pack(coords):
        shifted = coords - origin
        return (shifted[:, 0] * dims[1] + shifted[:, 1]) * dims[2] + shifted[:, 2]

    return pack


def connected_components(edges, count, max_iterations=64):
    """Label connected components from an undirected edge list.

    Min-label propagation: every node repeatedly takes the smallest label among
    itself and its neighbours, so the smallest index in each component floods
    outward. Pointer jumping (`labels[labels]`) halves the chain length each
    round, which is what keeps the iteration count near log(diameter) instead of
    proportional to it.

    Written this way because the obvious alternatives are unavailable or slow:
    scipy.sparse.csgraph is ABI-broken here (see docs/01), and a per-node Python
    flood fill costs seconds on a full-resolution cloud.
    """
    labels = np.arange(count, dtype=np.int64)
    if edges[0].size == 0:
        return labels

    # Symmetrise once, then sort by source so each round is a single reduceat
    # rather than an unbuffered scatter-min.
    src = np.concatenate([edges[0], edges[1]])
    dst = np.concatenate([edges[1], edges[0]])
    order = np.argsort(src, kind='stable')
    src, dst = src[order], dst[order]
    owners, starts = np.unique(src, return_index=True)

    for _ in range(max_iterations):
        best = np.minimum.reduceat(labels[dst], starts)
        updated = labels.copy()
        updated[owners] = np.minimum(updated[owners], best)
        updated = updated[updated]          # pointer jumping
        if np.array_equal(updated, labels):
            break
        labels = updated
    return labels


class CloudPipeline(Node):

    def __init__(self):
        super().__init__('cloud_pipeline')

        self.declare_parameter('cloud_topic', '/camera/camera/depth/color/points')
        # Stage 1. Below 0.3 m the D435 stereo baseline cannot triangulate; past
        # ~3 m its error passes 2 cm and clustering starts merging objects.
        self.declare_parameter('z_min', 0.3)
        self.declare_parameter('z_max', 3.0)
        # Stage 2. Cube edge in metres.
        self.declare_parameter('voxel_leaf', 0.01)
        # Stage 3.
        self.declare_parameter('remove_plane', True)
        self.declare_parameter('plane_threshold', 0.015)
        self.declare_parameter('plane_iterations', 100)
        self.declare_parameter('plane_min_ratio', 0.25)
        # Stage 4. Gap tolerated inside one object, in voxels.
        self.declare_parameter('cluster_gap_voxels', 1)
        self.declare_parameter('min_cluster_points', 40)
        self.declare_parameter('max_clusters', 20)

        self.rng = np.random.default_rng(0)
        self.filtered_pub = self.create_publisher(PointCloud2, '~/filtered', 1)
        self.plane_pub = self.create_publisher(PointCloud2, '~/plane', 1)
        self.marker_pub = self.create_publisher(MarkerArray, '~/clusters', 1)

        self.create_subscription(PointCloud2,
                                 self.get_parameter('cloud_topic').value,
                                 self.on_cloud, qos_profile_sensor_data)
        self.get_logger().info('cloud_pipeline up')

    # ------------------------------------------------------------------ main

    def on_cloud(self, msg):
        xyz = pointcloud2_to_xyz(msg)
        raw_count = xyz.shape[0]
        if raw_count == 0:
            return

        xyz = self.passthrough(xyz)
        if xyz.shape[0] == 0:
            self.get_logger().warn('passthrough removed every point', once=True)
            return

        xyz = self.voxel_downsample(xyz)

        plane = np.empty((0, 3), dtype=np.float32)
        if self.get_parameter('remove_plane').value:
            xyz, plane = self.remove_dominant_plane(xyz)

        self.filtered_pub.publish(xyz_to_pointcloud2(xyz, msg.header))
        if plane.shape[0]:
            self.plane_pub.publish(xyz_to_pointcloud2(plane, msg.header))

        clusters = self.cluster(xyz)
        self.publish_markers(clusters, msg.header)

        self.get_logger().info(
            f'{raw_count} raw -> {xyz.shape[0]} kept '
            f'({plane.shape[0]} plane) -> {len(clusters)} clusters',
            throttle_duration_sec=1.0)
        for i, points in enumerate(clusters[:5]):
            cx, cy, cz = points.mean(axis=0)
            dx, dy, dz = points.max(axis=0) - points.min(axis=0)
            self.get_logger().info(
                f'  cluster {i}: n={points.shape[0]:5d} '
                f'centroid=({cx:+.3f},{cy:+.3f},{cz:.3f}) m  '
                f'extents={dx:.3f}x{dy:.3f}x{dz:.3f} m',
                throttle_duration_sec=1.0)

    # ---------------------------------------------------------------- stages

    def passthrough(self, xyz):
        """Stage 1: keep only points inside the trustworthy depth band."""
        z = xyz[:, 2]
        keep = (z >= self.get_parameter('z_min').value) & \
               (z <= self.get_parameter('z_max').value)
        return xyz[keep]

    def voxel_downsample(self, xyz):
        """Stage 2: one output point per occupied voxel, at the voxel centroid.

        Averaging inside the voxel is what makes this a denoiser and not just a
        decimator - the per-point stereo jitter partially cancels.
        """
        leaf = float(self.get_parameter('voxel_leaf').value)
        idx = np.floor(xyz / leaf).astype(np.int64)

        # Collapse the 3D voxel index to one integer so the unique-ing runs on a
        # flat array. np.unique(..., axis=0) lexsorts rows instead and is ~50x
        # slower on a full-resolution cloud.
        _, inverse = np.unique(voxel_packer(idx)(idx), return_inverse=True)

        # np.bincount is the buffered accumulator; np.add.at is unbuffered and
        # roughly an order of magnitude slower for the same reduction.
        counts = np.bincount(inverse)
        sums = np.stack([np.bincount(inverse, weights=xyz[:, i])
                         for i in range(3)], axis=1)
        return (sums / counts[:, None]).astype(np.float32)

    def remove_dominant_plane(self, xyz):
        """Stage 3: RANSAC a plane, return (points off it, points on it).

        Three points define a plane; its normal is the cross product of two
        edges. A point's distance to the plane is the dot of the offset with
        that unit normal. The hypothesis with the most inliers wins.
        """
        threshold = float(self.get_parameter('plane_threshold').value)
        iterations = int(self.get_parameter('plane_iterations').value)
        if xyz.shape[0] < 3:
            return xyz, np.empty((0, 3), dtype=np.float32)

        best_inliers = None
        best_count = 0
        for _ in range(iterations):
            a, b, c = xyz[self.rng.choice(xyz.shape[0], 3, replace=False)]
            normal = np.cross(b - a, c - a)
            norm = np.linalg.norm(normal)
            if norm < 1e-9:      # collinear sample, no plane defined
                continue
            normal = normal / norm
            inliers = np.abs((xyz - a) @ normal) < threshold
            count = int(inliers.sum())
            if count > best_count:
                best_count, best_inliers = count, inliers

        # A weak best hypothesis means there is no dominant plane in view;
        # removing it anyway would eat chunks of real objects.
        ratio = best_count / xyz.shape[0]
        if best_inliers is None or ratio < self.get_parameter('plane_min_ratio').value:
            return xyz, np.empty((0, 3), dtype=np.float32)
        return xyz[~best_inliers], xyz[best_inliers]

    def cluster(self, xyz):
        """Stage 4: connected components over the occupied voxel lattice."""
        min_points = int(self.get_parameter('min_cluster_points').value)
        if xyz.shape[0] == 0:
            return []

        leaf = float(self.get_parameter('voxel_leaf').value)
        gap = int(self.get_parameter('cluster_gap_voxels').value)
        idx = np.floor(xyz / leaf).astype(np.int64)

        pack = voxel_packer(idx, pad=gap)
        keys = pack(idx)
        order = np.argsort(keys, kind='stable')
        sorted_keys = keys[order]

        # For each lattice offset, ask where every occupied cell's neighbour
        # would sit in the sorted key array. A hit is an edge. This replaces a
        # per-point radius search with one searchsorted per offset.
        sources, targets = [], []
        for offset in neighbour_offsets(gap):
            probe = pack(idx + offset)
            position = np.searchsorted(sorted_keys, probe)
            np.clip(position, 0, sorted_keys.size - 1, out=position)
            hit = sorted_keys[position] == probe
            if hit.any():
                sources.append(np.flatnonzero(hit))
                targets.append(order[position[hit]])

        edges = ((np.concatenate(sources), np.concatenate(targets)) if sources
                 else (np.empty(0, np.int64), np.empty(0, np.int64)))
        labels = connected_components(edges, xyz.shape[0])

        _, inverse, counts = np.unique(labels, return_inverse=True,
                                       return_counts=True)
        clusters = [xyz[inverse == label]
                    for label in np.flatnonzero(counts >= min_points)]
        clusters.sort(key=lambda c: c.shape[0], reverse=True)
        return clusters[:int(self.get_parameter('max_clusters').value)]

    # --------------------------------------------------------------- markers

    def publish_markers(self, clusters, header):
        """Stage 5: one translucent box per cluster, sized by its extents."""
        markers = MarkerArray()
        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        for i, points in enumerate(clusters):
            low, high = points.min(axis=0), points.max(axis=0)
            centre = (low + high) / 2.0
            size = np.maximum(high - low, 1e-3)

            box = Marker()
            box.header = header
            box.ns = 'clusters'
            box.id = i
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose.position.x = float(centre[0])
            box.pose.position.y = float(centre[1])
            box.pose.position.z = float(centre[2])
            box.pose.orientation.w = 1.0
            box.scale.x, box.scale.y, box.scale.z = map(float, size)
            box.color.r, box.color.g, box.color.b, box.color.a = 0.1, 0.8, 1.0, 0.35
            markers.markers.append(box)

            text = Marker()
            text.header = header
            text.ns = 'labels'
            text.id = i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(centre[0])
            text.pose.position.y = float(low[1] - 0.03)
            text.pose.position.z = float(centre[2])
            text.pose.orientation.w = 1.0
            text.scale.z = 0.04
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = (f'#{i} {size[0]*100:.0f}x{size[1]*100:.0f}x'
                         f'{size[2]*100:.0f} cm @ {centre[2]:.2f} m')
            markers.markers.append(text)

        self.marker_pub.publish(markers)


def main():
    rclpy.init()
    node = CloudPipeline()
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

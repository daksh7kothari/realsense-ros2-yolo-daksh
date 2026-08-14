"""Decoding helpers for sensor_msgs, implemented with numpy only.

cv_bridge and scipy are deliberately not used here: the Humble binaries for both
are compiled against numpy 1.x and raise on this machine's numpy 2.2.
"""

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

# sensor_msgs/Image encoding -> (numpy dtype, channel count)
_IMAGE_ENCODINGS = {
    '16UC1': (np.uint16, 1),
    'mono16': (np.uint16, 1),
    '8UC1': (np.uint8, 1),
    'mono8': (np.uint8, 1),
    'rgb8': (np.uint8, 3),
    'bgr8': (np.uint8, 3),
    'rgba8': (np.uint8, 4),
    'bgra8': (np.uint8, 4),
    '32FC1': (np.float32, 1),
}

_PF_TO_NP = {
    PointField.INT8: np.int8,
    PointField.UINT8: np.uint8,
    PointField.INT16: np.int16,
    PointField.UINT16: np.uint16,
    PointField.INT32: np.int32,
    PointField.UINT32: np.uint32,
    PointField.FLOAT32: np.float32,
    PointField.FLOAT64: np.float64,
}


def image_to_numpy(msg):
    """Return a (height, width) or (height, width, channels) view of an Image."""
    try:
        dtype, channels = _IMAGE_ENCODINGS[msg.encoding]
    except KeyError:
        raise ValueError(f'unsupported image encoding: {msg.encoding}')

    dtype = np.dtype(dtype).newbyteorder('>' if msg.is_bigendian else '<')
    # step is the row stride in bytes and may be larger than the used width.
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    used = msg.width * channels * dtype.itemsize
    arr = rows[:, :used].reshape(msg.height, -1).view(dtype)
    return arr if channels == 1 else arr.reshape(msg.height, msg.width, channels)


def sample_depth(depth, u, v, roi_half, depth_scale):
    """Median of the non-zero depths in a (2*roi_half+1) square around (u, v).

    Returns metres, or None if the pixel is out of bounds or the whole ROI is
    the RealSense "no measurement" sentinel (0).
    """
    h, w = depth.shape
    if not (0 <= u < w and 0 <= v < h):
        return None
    patch = depth[max(0, v - roi_half):min(h, v + roi_half + 1),
                  max(0, u - roi_half):min(w, u + roi_half + 1)]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) * depth_scale


class Intrinsics:
    """Pinhole intrinsics pulled out of a CameraInfo message."""

    def __init__(self, camera_info):
        k = camera_info.k
        self.fx, self.fy = float(k[0]), float(k[4])
        self.cx, self.cy = float(k[2]), float(k[5])
        self.width, self.height = camera_info.width, camera_info.height
        self.frame_id = camera_info.header.frame_id

    def deproject(self, u, v, z):
        """Pixel (u, v) plus depth z in metres -> (x, y, z) in the optical frame.

        Optical frame convention: +x right, +y down, +z forward.
        """
        return ((u - self.cx) * z / self.fx,
                (v - self.cy) * z / self.fy,
                z)

    def deproject_array(self, us, vs, zs):
        """Vectorised deproject. Inputs are 1-D arrays, output is (N, 3)."""
        return np.stack([(us - self.cx) * zs / self.fx,
                         (vs - self.cy) * zs / self.fy,
                         zs], axis=1)


def pointcloud2_to_xyz(msg, skip_nan=True):
    """Extract an (N, 3) float32 array of XYZ from a PointCloud2."""
    offsets = {f.name: (f.offset, _PF_TO_NP[f.datatype]) for f in msg.fields
               if f.datatype in _PF_TO_NP}
    for axis in ('x', 'y', 'z'):
        if axis not in offsets:
            raise ValueError(f'point cloud has no "{axis}" field')

    # A structured dtype whose itemsize is the full point stride lets numpy pull
    # the three coordinates out in one pass, ignoring the interleaved rgb/padding
    # bytes. Slicing each column out of a (N, point_step) uint8 view instead
    # costs three full copies of the buffer, which dominates the frame at
    # 848x480 (~400k points).
    view = np.dtype({
        'names': ['x', 'y', 'z'],
        'formats': [np.dtype(offsets[a][1]).newbyteorder(
            '>' if msg.is_bigendian else '<') for a in ('x', 'y', 'z')],
        'offsets': [offsets[a][0] for a in ('x', 'y', 'z')],
        'itemsize': msg.point_step,
    })
    fields = np.frombuffer(msg.data, dtype=view)

    xyz = np.empty((fields.shape[0], 3), dtype=np.float32)
    for i, axis in enumerate(('x', 'y', 'z')):
        xyz[:, i] = fields[axis]

    if skip_nan:
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
    return xyz


def xyz_to_pointcloud2(xyz, header):
    """Build an XYZ-only PointCloud2 (12 bytes per point) from an (N, 3) array."""
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = int(xyz.shape[0])
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * msg.width
    msg.is_dense = True
    msg.data = xyz.tobytes()
    return msg

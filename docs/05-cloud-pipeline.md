# 05 — The point cloud pipeline

`cloud_pipeline.py` implements the standard five stages. The **order** is the
whole idea: each stage is cheap because the stage before it threw work away. Run
them in the wrong order and the same pipeline is ten times slower and worse.

```
PointCloud2 (~400k points)
  │
  1. passthrough      crop the depth band          → ~250k
  2. voxel grid       one point per 1 cm cube      → ~20k
  3. RANSAC plane     remove the table/floor       → ~5k
  4. clustering       connected components         → N objects
  5. measurement      centroid + extents per object
```

Measured on the synthetic test scene (14 000 points, a plane plus two boxes plus
far noise): `14000 → 11000 → 8688 → 2382 → 2 clusters`, and the recovered box
extents matched the ground truth to the voxel size.

## Stage 1 — Passthrough

```python
keep = (z >= z_min) & (z <= z_max)
```

First, because it is the cheapest possible operation and it deletes the worst
data. Far points carry error growing as Z² (`docs/03`), so they are simultaneously
the most numerous and the least trustworthy. Cropping to 0.3–3.0 m removes them
before any stage has to pay to process them.

Tune `z_max` to your actual working volume. Every stage downstream gets faster
and more accurate when you do.

## Stage 2 — Voxel grid downsample

Overlay a 3D lattice of 1 cm cubes; replace all points in a cube with their
centroid.

```python
idx = np.floor(xyz / leaf).astype(np.int64)
_, inverse = np.unique(voxel_packer(idx)(idx), return_inverse=True)
counts = np.bincount(inverse)
sums = np.stack([np.bincount(inverse, weights=xyz[:, i]) for i in range(3)], axis=1)
return (sums / counts[:, None]).astype(np.float32)
```

`voxel_packer` folds the three integer voxel coordinates into a single `int64`
key so the unique-ing runs on a flat array; `np.bincount` accumulates
coordinates per voxel; dividing by the counts gives centroids. No Python loop.

The obvious spelling — `np.unique(idx, axis=0)` with `np.add.at` — is 13×
slower and was the single biggest bottleneck in the first version. `docs/08`.

Taking the **centroid** rather than picking a representative point is what makes
this a denoiser rather than a decimator: independent per-point stereo jitter
partially cancels inside each cube.

`voxel_leaf` is the pipeline's master dial. It sets the cost of everything after
it, and it sets the smallest feature that can survive. 1 cm suits tabletop
objects; go to 2–3 cm for room-scale work.

## Stage 3 — RANSAC plane removal

Most useful scenes have one large flat surface — a table, the floor, a wall —
and it connects every object into a single blob. Removing it is what makes
clustering work at all.

RANSAC = RANdom SAmple Consensus. Three points define a plane, so:

1. sample 3 points at random
2. normal = normalised cross product of two edges
3. distance of every point to that plane = `|(p − a) · n̂|`
4. count how many fall within `plane_threshold`
5. keep the hypothesis with the most inliers, repeat `plane_iterations` times

```python
normal = np.cross(b - a, c - a)
normal = normal / np.linalg.norm(normal)
inliers = np.abs((xyz - a) @ normal) < threshold
```

Step 3 is a single matrix-vector product for the entire cloud, which is why 100
iterations is affordable at frame rate.

Two guards worth noting:

- **Collinear samples.** Three points on a line give a zero-length cross
  product. Normalising that yields NaN and poisons the comparison, so a sample
  with `‖normal‖ < 1e-9` is skipped.
- **`plane_min_ratio`.** If the best hypothesis claims less than 25% of the
  cloud, there is no dominant plane in view — the camera is pointed at open
  space. Removing the "best" plane anyway would carve a slab out of real
  objects. The stage declines and passes the cloud through untouched.

`plane_threshold` of 1.5 cm is chosen to sit above the sensor noise at ~2 m
(≈1 cm) so a real plane is captured in one hypothesis, but below the height of
anything you care about, so objects are not absorbed into it.

## Stage 4 — Clustering

Group the surviving points into objects. The textbook approach is Euclidean
cluster extraction with a KD-tree — which is what PCL does, and what scipy's
`cKDTree` would give you. scipy is ABI-broken here (`docs/01`), so the
connectivity is computed on the voxel lattice instead.

The cloud is already voxelised, so every point *is* a lattice cell. Two objects
are separate exactly when their occupied cells are not adjacent. This is not a
compromise so much as the same algorithm: once a cloud is voxelised, PCL's
Euclidean clustering with tolerance ≈ leaf size is doing this.

It runs in two vectorised halves. First the edge list — sort the packed voxel
keys once, then for each of the 26 lattice offsets ask where every cell's
neighbour *would* sit, with one `searchsorted` per offset:

```python
for offset in neighbour_offsets(gap):
    probe = pack(idx + offset)
    position = np.searchsorted(sorted_keys, probe)
    hit = sorted_keys[position] == probe
```

Then the labelling, by min-label propagation with pointer jumping:

```python
best = np.minimum.reduceat(labels[dst], starts)
updated[owners] = np.minimum(updated[owners], best)
updated = updated[updated]          # pointer jumping
```

The first draft of this stage was a Python flood fill with a `dict` keyed by
coordinate tuples. It was correct and 11× slower, and it did not finish at all
on a dense cloud. `docs/08` has the measurements and the reasoning.

`cluster_gap_voxels` is the tolerance. At 1, cells must touch (including
diagonally). Raise it to bridge gaps where an object was partially removed by
stage 3, at the cost of merging objects that stand close together.

`min_cluster_points` drops specks. Clusters come back sorted largest-first, so
`clusters[0]` is the most prominent object.

## Stage 5 — Measurement

Per cluster: centroid, and the axis-aligned extents.

```python
low, high = points.min(axis=0), points.max(axis=0)
centre = (low + high) / 2.0
size = high - low
```

Published as a translucent `CUBE` marker with a `TEXT_VIEW_FACING` label.

**Read the caveat.** These extents are axis-aligned **in the optical frame** —
X right, Y down, Z forward. They are the object's bounding box as the camera
happens to be oriented, not its intrinsic dimensions. Rotate the camera 45° and
the numbers change. For orientation-independent dimensions you need an oriented
bounding box: take the eigenvectors of the cluster's covariance (PCA), project
the points onto that basis, then take the extents there. Roughly ten more lines,
and worth adding when you need it.

The `DELETEALL` marker published first each frame is not optional. Without it,
markers from a previous frame with higher IDs linger in RViz forever, and you
end up looking at a scene of ghosts.

## Tuning, in the order that pays

| Symptom | Change |
|---|---|
| Too slow | raise `voxel_leaf` to 0.02, lower `z_max` |
| Objects merge into one | lower `cluster_gap_voxels`, lower `z_max`, lower `voxel_leaf` |
| One object splits in two | raise `cluster_gap_voxels`, raise `plane_threshold` |
| Table not removed | raise `plane_threshold` or `plane_iterations` |
| Objects eaten by plane removal | lower `plane_threshold`, raise `plane_min_ratio` |
| Nothing detected | check `z_min`/`z_max` against your real distances first |

Live, without relaunching:

```bash
ros2 param set /cloud_pipeline voxel_leaf 0.02
ros2 param set /cloud_pipeline z_max 2.0
```

## Throughput

Worst case, a fully dense 407k-point cloud: **289 ms**, dominated by voxel
(124 ms) and RANSAC (76 ms). Live on the camera, where much of the frame is
invalid: **10.5 Hz** against a 30 Hz driver.

If you need more, the order that pays is in `docs/08`. The cheapest win is
always tightening the passthrough band.

Next: [06 — Design decisions](06-design-decisions.md)

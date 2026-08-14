# 08 — Performance: finding and fixing the bottleneck

The first live run worked and was **too slow**. The driver published clouds at
30 Hz; `cloud_pipeline` republished at 6 Hz. This is the record of finding out
why, because the method generalises further than the fix does.

## Measure first

```bash
ros2 topic hz /camera/camera/depth/color/points   # 30.0
ros2 topic hz /cloud_pipeline/filtered            #  6.0
```

That establishes the gap but not the cause. The next step was to time each stage
in isolation against a synthetic full-resolution cloud (848×480 = 407 040
points), because guessing which stage is slow is exactly the mistake that leads
to optimising the wrong thing.

First result:

```
decode          53.8 ms
passthrough     16.8 ms
voxel         1800.7 ms     ← 93% of the frame
ransac          68.8 ms
cluster       (hung)
```

One stage was the entire problem, and a second was bad enough to hang a
benchmark on a pathological cloud.

## A detour worth recording

After editing the code the benchmark barely moved. The reason was not the code:

```python
print(cp.__file__)   # .../install/.../cloud_pipeline.py
print('voxel_packer' in open(cp.__file__).read())   # False
```

`colcon build --symlink-install` had left a **stale copy** in `install/`, so
every measurement after the edit was still running the old implementation. The
tests kept passing because pytest ran with `src/` on the path and imported the
new code, while the benchmark sourced `install/` and imported the old.

The lesson is cheap and generalises: when a change appears to have no effect,
verify the file you edited is the file being executed *before* you go looking
for a subtler explanation. `colcon build` again fixed it.

## Fix 1 — voxel grid: 1640 ms → 124 ms

Two numpy calls were responsible.

**`np.unique(idx, axis=0)`.** With `axis=0` numpy lexsorts the rows of a
(407040, 3) array, which is dramatically slower than sorting a flat array. The
fix is to pack the three integer voxel coordinates into a single `int64` key and
unique *that*:

```python
def voxel_packer(idx, pad=0):
    origin = idx.min(axis=0) - pad
    dims = (idx.max(axis=0) + pad) - origin + 1

    def pack(coords):
        shifted = coords - origin
        return (shifted[:, 0] * dims[1] + shifted[:, 1]) * dims[2] + shifted[:, 2]

    return pack
```

The `pad` argument matters for stage 4 — see below.

**`np.add.at(sums, inverse, xyz)`.** `ufunc.at` is the *unbuffered* path, chosen
for correctness under duplicate indices, and it is roughly an order of magnitude
slower than the buffered equivalent. Since the reduction here is a plain sum
grouped by an integer label, `np.bincount` does the same job:

```python
counts = np.bincount(inverse)
sums = np.stack([np.bincount(inverse, weights=xyz[:, i]) for i in range(3)], axis=1)
```

## Fix 2 — clustering: 434 ms → 38 ms

The original was a Python flood fill: for each unvisited voxel, walk its 26
neighbours, look each up in a `dict` keyed by a coordinate tuple. Every step
allocated a tuple and took the Python interpreter's slow path. On a cloud with
tens of thousands of occupied voxels it did not finish.

The replacement is in two vectorised halves.

**Building the edge list.** Sort the packed keys once. For each of the 26 lattice
offsets, compute where every cell's neighbour *would* sit with one
`np.searchsorted`, and keep the hits:

```python
for offset in neighbour_offsets(gap):
    probe = pack(idx + offset)
    position = np.searchsorted(sorted_keys, probe)
    np.clip(position, 0, sorted_keys.size - 1, out=position)
    hit = sorted_keys[position] == probe
```

26 vectorised passes replace N × 26 interpreted lookups. This is why
`voxel_packer` takes `pad`: probes step outside the occupied set, and padding
the lattice bounds guarantees an out-of-range probe cannot pack to the same key
as a real cell.

**Labelling.** Min-label propagation — each node repeatedly takes the smallest
label among itself and its neighbours, so the lowest index in each component
floods outward:

```python
best = np.minimum.reduceat(labels[dst], starts)
updated[owners] = np.minimum(updated[owners], best)
updated = updated[updated]          # pointer jumping
```

Sorting the symmetrised edge list once up front turns each round into a single
`reduceat` instead of a scatter-min (`np.minimum.at`, unbuffered and slow
again). `updated[updated]` is pointer jumping: it halves every label chain per
round, so the iteration count tracks log(diameter) rather than diameter.

scipy's `csgraph.connected_components` would be the normal answer here. It is
ABI-broken on this machine (`docs/01`).

## Fix 3 — decode: 54 ms → 34 ms

The decoder sliced three columns out of a `(N, point_step)` uint8 view, each
slice forcing a full copy. A structured dtype whose `itemsize` is the point
stride reads all three in one pass, skipping the interleaved rgb and padding
bytes:

```python
view = np.dtype({'names': ['x', 'y', 'z'],
                 'formats': [...],
                 'offsets': [...],
                 'itemsize': msg.point_step})
fields = np.frombuffer(msg.data, dtype=view)
```

## Result

| Stage | Before | After |
|---|---|---|
| decode | 53.8 ms | 33.8 ms |
| passthrough | 16.8 ms | 18.0 ms |
| voxel | 1800.7 ms | 124.0 ms |
| ransac | 68.8 ms | 75.8 ms |
| cluster | 433.8 ms | 37.8 ms |
| **total** | **2231.7 ms (0.4 Hz)** | **289.3 ms (3.5 Hz)** |

Worst case, on a fully dense 407k-point cloud. Live on the camera, where much
of the frame is invalid and the passthrough band trims further:

```
/cloud_pipeline/filtered : 6.0 Hz  →  10.5 Hz
```

Recovered box extents were identical before and after — `0.15 × 0.15 × 0.20 m`
and `0.10 × 0.15 × 0.15 m` — and all 43 tests passed unchanged, which is what
makes this a refactor rather than a rewrite.

## What is still on the table

`voxel` at 124 ms and `ransac` at 76 ms now lead. Ideas, roughly in order of
payoff per unit effort:

- **Tighten the passthrough band.** Still the cheapest win by far. Every stage
  downstream scales with what survives it.
- **Score RANSAC on a subsample.** The plane hypothesis does not need all 25k
  points to be ranked; a random 10% gives the same winner for a fraction of the
  cost, with a single full pass at the end to extract the inliers.
- **Drop `plane_iterations`.** 100 is conservative for a plane covering >25% of
  the cloud; 30 usually suffices.
- **Decimate at the driver.** `decimation_filter.enable:=true` halves the cloud
  before it ever reaches ROS, which beats any optimisation on this side.
- **Rewrite in C++** if you need a genuine 30 Hz. That is the real ceiling here —
  Python plus numpy is at roughly the right order of magnitude now, and closing
  the last 3× means leaving the interpreter.

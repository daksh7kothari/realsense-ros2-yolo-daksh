# 07 — The test suite

56 tests, all passing, no camera required. They exist because the interesting
failures in depth work are *quiet* — a sheared image, a zero averaged in as a
range, a plane hypothesis that eats an object — and none of them throw.

## Running

```bash
cd /home/daksh/realsense
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 colcon test --packages-select realsense_depth
colcon test-result --verbose
```

Or directly, which is faster while iterating:

```bash
source /opt/ros/humble/setup.bash
source /home/daksh/realsense/install/setup.bash
cd src/realsense_depth
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test -q
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is a third environment clash, in the same
family as the numpy one in `docs/01`. The system pytest is 6.x; a user-site
`anyio` plugin in `~/.local` imports `_pytest.scope`, which only exists in
pytest 7+:

```
ModuleNotFoundError: No module named '_pytest.scope'
```

The variable stops pytest loading third-party plugins it does not need. The
alternative fix is `pip install -U pytest`, which you may not want.

## What is covered

### `test_msg_utils.py` — the decoders

The one worth understanding is `test_decodes_row_padded_depth`. `Image.step` is
a row stride in bytes and may exceed `width × channels × itemsize`. Reshaping
straight to `(height, width)` works on most streams and *shears the image* on a
padded one. The test builds a deliberately padded message so that bug cannot
come back.

Also covered: rgb8 decode, rejection of unknown encodings, `CameraInfo` parsing,
deprojection identities (principal point maps to the optical axis, offsets scale
linearly with range, +y is down), vectorised and scalar deproject agreeing,
`PointCloud2` round-trip, non-finite point removal, and reading xyz out of a
32-byte XYZRGB stride — which is what the real camera publishes.

### `test_depth_measure.py` — the probe

Driven by synthetic `Image` and `CameraInfo` messages fed straight into the
callbacks, with the publisher stubbed to capture output. No DDS, no camera.

The two that encode real design decisions:

- `test_zero_is_treated_as_no_measurement_not_zero_range` — blank the whole ROI
  and assert *nothing is published*, rather than a confident 0.0 m.
- `test_median_rejects_flying_pixels` — plant a 9000 mm flying pixel and a 0
  dropout inside the ROI and assert the reported range does not move. This is
  the test that would fail if someone "simplified" the median to a mean.

Plus: probe defaults to image centre, depth ignored before intrinsics arrive,
off-centre deprojection, `frame_id` passthrough for TF2, out-of-bounds probe,
the Z² error model, and that a span between two markers measures true 3D
distance rather than scaled pixels.

### `test_cloud_pipeline.py` — the five stages

Synthetic scene with exact ground truth: a plane at z = 1.5 m, a 0.10 × 0.15 ×
0.15 m box, a 0.15 × 0.15 × 0.20 m box, and far-field noise.

Per stage:

| Stage | Asserted |
|---|---|
| passthrough | the 3000 far points are gone; band is runtime-tunable |
| voxel | count drops, bounds preserved, one point per cell, centroid not decimation, larger leaf → fewer points |
| RANSAC | the plane is found, boxes survive intact, **declines when no dominant plane is in view**, collinear samples produce no NaN, tiny clouds handled |
| clustering | two boxes come back as two clusters, **extents match ground truth to one voxel**, largest-first ordering, specks dropped, empty cloud handled, gap tolerance merges neighbours |

`test_recovered_extents_match_ground_truth` is the end-to-end claim: run a scene
whose object dimensions you know through all five stages and get those
dimensions back, within the 1 cm voxel quantisation.

`test_declines_when_no_dominant_plane_is_in_view` and
`test_collinear_samples_do_not_produce_nan` guard the two RANSAC guards
described in `docs/05`. Both would pass silently as wrong behaviour without a
test — the first by quietly deleting a slab of a real object, the second by
poisoning comparisons with NaN.

### `test_helpers.py` — the vectorised lattice helpers

Added with the optimisation pass (`docs/08`), because the rewritten clustering
replaced a simple flood fill with something much easier to get subtly wrong.

`test_padding_keeps_out_of_range_probes_from_colliding` guards the reason
`voxel_packer` takes a `pad` argument at all: neighbour probes step outside the
occupied set, and without padded lattice bounds an out-of-range probe can pack
to the same key as a real cell, silently fusing two objects.

`test_long_chain_converges_within_the_iteration_cap` builds a 500-node chain.
Min-label propagation without pointer jumping advances one hop per round and
would hit the 64-iteration cap and return a wrong answer; with it, the chain
flattens logarithmically. The test fails if someone removes `labels[labels]`.

Also: packing is injective, survives negative coordinates, offsets are symmetric,
isolated nodes stay separate, disjoint chains stay disjoint, labels are the
smallest index in each component, and labelling is independent of edge order.

## What is not covered

The OpenCV viewer window, and the launch file itself. Everything else has now
run against the real camera — topic names, QoS negotiation, and both nodes on
live data are confirmed working (`docs/02`, `docs/08`).

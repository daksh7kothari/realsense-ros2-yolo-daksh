# 03 — Depth fundamentals

## The encoding

The depth topics carry `16UC1`: unsigned 16-bit, one channel. The value is the
distance in **millimetres**, so

```
metres = raw_value * depth_scale        # depth_scale = 0.001 for the D435
```

Two consequences you have to design around:

**Zero is not zero distance. Zero means "no measurement".** Where the stereo
matcher found no correspondence — shiny surfaces, uniform white walls, shadowed
regions, anything closer than the minimum range — it writes 0. If you average a
patch of depth without masking zeros, every invalid pixel drags your result
toward the camera and you get a confidently wrong answer.

**The 16-bit ceiling is 65.535 m.** Not a practical limit for a D435, but it
means you cannot naively subtract two depth values in `uint16` without
underflow wrapping to ~65000. Cast to float first.

## Why the code takes a median, not a mean

`depth_measure.sample()` collects an 11×11 ROI, discards zeros, and takes the
**median**:

```python
valid = patch[patch > 0]
if valid.size == 0:
    return None
return float(np.median(valid)) * self.depth_scale
```

Three separate reasons, and each matters:

1. **Zeros are sentinels, so they must be dropped, not averaged.**
2. **Depth noise is not Gaussian.** It has a heavy tail of "flying pixels" at
   depth discontinuities — points that land halfway between the foreground
   object and the background wall because the correlation window straddled the
   edge. A mean is dragged by those; a median ignores them.
3. **A single pixel is far noisier than a patch.** The ROI is free averaging.

Returning `None` rather than `0.0` when the whole ROI is invalid is deliberate.
`0.0` is a plausible-looking number that will propagate into your measurements
silently. `None` forces the caller to handle the case.

## How wrong is the number?

The D435 is a stereo camera. It finds the same feature in the left and right
infrared images, and the horizontal shift between them — the *disparity* — gives
the range by triangulation. Standard stereo relation:

```
Z = focal_px * baseline / disparity
```

Differentiate with respect to disparity and you get the error model:

```
depth_error ≈ Z² * subpixel_error / (focal_px * baseline)
```

For the D435: baseline = 50 mm, focal ≈ 645 px at 848×480, subpixel ≈ 0.08 px.

| Range | Expected RMS error |
|---|---|
| 0.5 m | ~0.6 mm |
| 1 m | ~2.5 mm |
| 2 m | ~10 mm |
| 3 m | ~22 mm |
| 5 m | ~62 mm |
| 10 m | ~248 mm |

**The error grows with the square of the range.** This one fact should drive
every parameter choice you make:

- It is why `z_max` defaults to 3.0 m in `cloud_pipeline`. Past that, error
  exceeds 2 cm and clustering starts fusing distinct objects into one blob.
- It is why `z_min` is 0.3 m. Below the minimum range the stereo pair cannot
  triangulate at all and you get zeros.
- It is why measuring a small object at 4 m is not a resolution problem you can
  fix with a better algorithm. The information is not in the data.

`depth_measure` prints the expected error alongside every reading:

```
pixel (424,240) -> x=+0.000 y=+0.000 z=1.204 m | expected error ~4.5 mm
```

That number is `DepthMeasure.depth_rms()`, the formula above. It is there so
you never quote a measurement without knowing its uncertainty.

## Things that make depth worse

| Cause | Effect | Mitigation |
|---|---|---|
| Untextured surface (white wall) | zeros / dropout | the D435's IR projector supplies texture; keep it on |
| Direct sunlight | washes out the IR projector | shorter range, or indoors |
| Glass, mirrors, gloss | wrong depth or zeros | no fix; treat as invalid |
| Object edges | flying pixels | median filtering, and the voxel stage |
| Range | error grows as Z² | keep the working volume close |

Next: [04 — Pixel to 3D](04-pixel-to-3d.md)

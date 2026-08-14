# 10 — Method: the transferable parts

`docs/09` is the narrative. This is the reusable version — the habits, with the
commands, so you can apply them to the next thing rather than re-derive them.

---

## Before writing code, establish ground truth

The three questions, in order:

1. **What is installed?** `ls /opt/ros/`, `apt list --installed | grep X`,
   `ros2 pkg list | grep X`
2. **Is the hardware there?** `lsusb`, plus the vendor's own tool
   (`rs-enumerate-devices`) — this separates "hardware broken" from "ROS broken"
   in one step and is worth doing first every time
3. **Do the libraries I plan to use actually work?** A three-line script that
   performs the real operation

Question 3 caught the numpy break here. It is the one people skip.

```python
# not this
import cv_bridge
# this
CvBridge().imgmsg_to_cv2(real_message, '16UC1')
```

Import success proves the file exists. It does not prove the C extension inside
it is ABI-compatible with your numpy.

## Read errors backwards to the cause

```
404 Not Found  (symptom)
  ← stale package index
    ← apt could not verify the repo
      ← NO_PUBKEY F42ED6FBAB17C654
        ← keyring file is 20 bytes   (cause)
```

The visible error was three steps from the actual problem. `apt-get update` was
what exposed the chain. Most tools have an equivalent verbose or refresh mode;
run it before forming a theory.

The general failure to avoid: fixing the symptom. Manually downloading those
`.deb` files would have "worked" and left the broken keyring in place to fail
again on your next install.

## Test against known answers, not against plausibility

The single most useful thing in this package is a synthetic scene with exact
ground truth:

```python
BOX_A = ((-0.30, -0.10, 1.30), (-0.20, 0.05, 1.45))   # 0.10 x 0.15 x 0.15 m
```

Because the answer is known, "did the pipeline work?" is a real question with a
real answer. Point a camera at your desk and it is a matter of opinion.

This applies to almost any perception or numerical work: construct input whose
output you can derive by hand, and assert on it.

## Profile before optimising, and profile per stage

```python
def bench(f, n=5):
    f()                              # warm up, ignore first run
    t = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - t) / n * 1000
```

Time each stage separately and print the sizes flowing between them. One stage
was 93% here; the guess before measuring would have been the RANSAC loop, which
turned out to be 3%.

Also: make the benchmark input *realistic*. Uniform random noise made the
clustering stage hang and told us nothing about real performance.

## numpy: know which calls fall off the fast path

The three fixes in `docs/08` were all the same shape — a call that silently
takes a slow path, swapped for one that stays vectorised.

| Slow | Why | Fast |
|---|---|---|
| `np.unique(a, axis=0)` | lexsorts rows | pack to one int64 key, unique that |
| `np.add.at(out, idx, v)` | unbuffered scatter | `np.bincount(idx, weights=v)` |
| `arr[:, i:j].copy().view(t)` per column | one full copy each | structured dtype with `itemsize=stride` |
| Python loop with `dict[tuple(k)]` | interpreter per element | sort keys + `np.searchsorted` |

The pattern behind all four: **replace per-element Python work with one
array-wide operation.** When you find yourself writing `for` over points, ask
what the whole-array spelling is.

`ufunc.at` is the trap worth remembering — it exists specifically to handle
duplicate indices correctly, and pays roughly 10× for the privilege. If your
reduction is a sum grouped by integer label, `bincount` does it.

## Refactor only behind tests

The rewrite touched the two most intricate stages. The safety net was one
assertion:

```
before: extents 0.15 x 0.15 x 0.20  and  0.10 x 0.15 x 0.15
after:  extents 0.15 x 0.15 x 0.20  and  0.10 x 0.15 x 0.15
```

Identical output plus 7.7× faster is a refactor. Without that assertion it is a
rewrite, and you are hoping.

Then add tests for the new failure modes. The vectorised clustering can fail in
ways the flood fill could not — key collisions, and non-convergence on a long
chain — so `test_helpers.py` covers exactly those.

## When something has no effect, check you are running it

```python
print(module.__file__)
print('new_function_name' in open(module.__file__).read())
```

Two lines, and they would have saved a full benchmark cycle here. Stale build
artefacts, shadowed imports, an unsourced workspace, a second Python
environment — all present as "my change did nothing".

Check this *before* concluding the change was wrong.

## Distinguish verified from assumed, out loud

Until the wrapper was installed, the topic names in `docs/02` came from
documentation, not from a running system — and the docs said so. After the live
run they were confirmed, and the docs were updated.

Both states are fine. Confusing them is not: an assumption presented as a
verified fact is how you end up debugging the wrong layer for an hour.

Useful phrasings: "verified live", "from the docs, untested here", "passes on
synthetic data, no hardware yet".

## ROS 2 specifics worth carrying forward

- **QoS mismatch fails silently.** Sensor topics are BEST_EFFORT; the rclpy
  default is RELIABLE; incompatible pairs simply never connect, with no error.
  Use `qos_profile_sensor_data` for anything from a driver. First thing to check
  when a subscriber "does nothing".
- **`ros2 topic hz` on both ends** localises a bottleneck in seconds — is the
  driver slow, or your node?
- **Everything tunable should be a parameter**, so `ros2 param set` replaces an
  edit-rebuild-relaunch cycle. Perception work is mostly tuning.
- **`colcon build` after editing**, even with `--symlink-install`. It does not
  always symlink.
- **Nodes should be testable without a camera.** Both nodes here are driven by
  synthetic messages straight into the callbacks, with publishers stubbed. No
  DDS, no hardware, tests run in 3 seconds.

## The loop

```
inspect → probe deps → design around constraints
       → build → verify against known answers
       → integrate → verify live → measure → optimise under test
       → correct the record
```

The two most commonly skipped steps are *probe deps* and *measure*. They were
also the two that mattered most here.

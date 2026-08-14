# 09 — How this was built, in order

Docs 01–08 explain the finished thing. This one is the process: what was done
first, what each step revealed, and where it went wrong. The mistakes are left
in, because the recovery is the part worth learning.

---

## Step 1 — Find out what is actually there

Nothing was written until the machine had been inspected. Five commands:

```bash
ls /opt/ros/                                   # ROS distro
ls /opt/ros/humble/share/ | grep -i realsense  # is the wrapper installed?
apt list --installed | grep -i realsense       # is the SDK installed?
lsusb | grep -i intel                          # camera present?
rs-enumerate-devices -s                        # SDK can talk to it?
```

You had said "setup is done". That was true of the SDK half and false of the
ROS half — `realsense2_camera` was absent. Not a contradiction; "setup" is
ambiguous, and the cheapest way to resolve ambiguity is to look rather than ask.

**The principle:** verify the starting state before building on it. A five-second
check beats an hour of debugging code that was never going to run.

## Step 2 — Probe the tools before committing to them

The plan was to use `cv_bridge` (the standard way to convert ROS images to
arrays) and scipy's KD-tree (the standard way to cluster points). Both were
tested with three-line scripts first.

Both were broken — numpy 2.2 against binaries compiled for numpy 1.x (`docs/01`).

The detail that matters: `import cv_bridge` **succeeds**. Only the actual
conversion call fails.

```python
from cv_bridge import CvBridge      # fine
bridge.imgmsg_to_cv2(msg, '16UC1')  # AttributeError: _ARRAY_API not found
```

A smoke test that only imports would have passed, and the failure would have
surfaced later with the camera attached, where it is far harder to attribute.

**The principle:** test a dependency by doing the thing you need it for, not by
importing it.

## Step 3 — Design around the constraint, and say so

Two options: downgrade numpy system-wide, or depend on neither library.

Option B was chosen — not because it is better in the abstract, but because a
system-wide downgrade is a change to your machine that affects everything else
you run, and that is your call to make, not mine. The cost was ~100 lines in
`msg_utils.py`. Both options are written down in `docs/01` so the decision stays
reversible.

**The principle:** when a constraint forces a choice, record the alternative
too. A decision you cannot revisit is a decision you cannot correct.

## Step 4 — Build, then verify against known answers

The two nodes were written, then tested on a **synthetic scene with exact ground
truth**: a plane at z = 1.5 m and two boxes of known dimensions.

```
passthrough: 14000 -> 11000
voxel:                8688
plane:                2382   (6306 plane points)
clusters: 2
   0  extents [0.15 0.15 0.2 ]
   1  extents [0.1  0.15 0.149]
```

The boxes were built to be exactly 0.15 × 0.15 × 0.20 and 0.10 × 0.15 × 0.15.
Getting those numbers back is a real check; watching a point cloud in RViz and
thinking "looks about right" is not.

**The principle:** test against data whose answer you already know. With a real
camera you cannot tell a working pipeline from a broken one by looking.

## Step 5 — Fix the blocker that was not the task

The wrapper install failed with 404s on every package. The obvious read is a
mirror problem. `apt-get update` gave the real reason:

```
NO_PUBKEY F42ED6FBAB17C654
```

and the keyring was 20 bytes — truncated. apt could not verify the repo, so it
silently kept using a stale index and requested `.deb` versions that no longer
existed upstream. The 404s were a symptom two steps removed from the cause.

The fix downloaded the official key **and checked its fingerprint before
installing it**:

```bash
gpg --show-keys --with-fingerprint ros.key
# C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
```

The last 16 hex digits are exactly the `NO_PUBKEY` apt named. A signing key is a
trust anchor; installing one you have not checked defeats the purpose of
signatures.

**The principle:** when an error message names a cause, chase it before
theorising. And never install a key without verifying it, however routine.

## Step 6 — Verify live, and read the numbers critically

With the driver running, every assumed topic name matched. But two numbers
looked wrong:

```
pixel (424,240) -> z=0.175 m
17241 raw -> 723 kept
```

0.175 m is below the 0.3 m `z_min`, and 723 of 17 241 points survived. The code
was fine; the camera was pointed at something ~17 cm away and the default band
was cropping nearly everything. That is a tuning observation, and it was
reported as such rather than "it works".

**The principle:** a system that runs without crashing is not a system that
works. Check whether the output is *plausible for your setup*.

## Step 7 — Measure before optimising

The pipeline republished at 6 Hz against a 30 Hz driver. Rather than guess which
stage was slow, each was timed separately:

```
decode          53.8 ms
voxel         1800.7 ms     ← 93%
ransac          68.8 ms
```

One stage was essentially the whole problem. Any effort spent on the other four
would have been wasted, and "the RANSAC loop looks expensive" was the intuitive
and wrong guess.

**The principle:** profile first. Intuition about performance is unreliable, and
the cost of measuring is minutes.

## Step 8 — The mistakes

Four, all instructive.

**A benchmark that hung.** The first profiling run never finished. The synthetic
cloud was uniform random noise, which fills a far larger number of voxels than
any real scene, and the Python flood fill choked. The scene was rebuilt to be
realistic (a wall plus objects). *Synthetic test data has to resemble real data
in the dimension you are measuring.*

**A fix that appeared to do nothing.** After rewriting `voxel_downsample` the
benchmark barely moved. The cause was not the code:

```python
print(cp.__file__)                                  # .../install/...
print('voxel_packer' in open(cp.__file__).read())   # False
```

`colcon build --symlink-install` had left a stale copy in `install/`. Tests
imported from `src/` and passed; the benchmark sourced `install/` and ran the
old code. *When a change has no effect, confirm the file you edited is the file
being executed — before looking for a subtler cause.*

**A NameError.** The helper was renamed `voxel_packer` but one call site still
said `voxel_keys`. Nine tests failed immediately and named the problem. *This is
what tests are for — a thirty-second fix instead of a confusing runtime failure
later.*

**Bad arithmetic in a test.** A test asserted a median of `0.11` where the
correct answer was `1.1`. The test was wrong, not the code. *When a test fails,
the test is a suspect too — but verify which, do not assume.*

## Step 9 — Refactor under test

The optimisation rewrote two stages substantially. What made that safe was the
existing 43 tests, in particular `test_recovered_extents_match_ground_truth`.
Before: `0.15 × 0.15 × 0.20` and `0.10 × 0.15 × 0.15`. After: identical.

That is the difference between a refactor and a rewrite. Without the tests, a
7.7× speedup would have come with no way of knowing whether it broke the
clustering.

Tests were then **added** for the new helpers (`test_helpers.py`), because the
new code has failure modes the old code did not — key collisions, and label
propagation that fails to converge on a long chain.

**The principle:** tests are what let you change code aggressively. Write them
before you need them.

## Step 10 — Correct the record

Several docs said "not verified against hardware" — true when written, false
after the wrapper was installed. Those claims were updated rather than left to
rot.

`docs/05` still described the dict-based flood fill after it had been replaced.
Also fixed.

**The principle:** documentation that describes code which no longer exists is
worse than no documentation, because it is believed.

---

## The shape of it

```
inspect  →  probe dependencies  →  design around constraints
         →  build  →  verify against known answers
         →  integrate  →  verify live  →  measure  →  optimise under test
         →  correct the record
```

Nothing here is specific to RealSense. The parts that generalise are in
[10 — Method](10-method.md).

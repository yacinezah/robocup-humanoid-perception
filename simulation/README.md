# Deterministic perception replay

This directory contains the capture, synchronization, geometry and detector
adapter code used with the ITAndroids ROS/Gazebo stack. It does not contain robot
meshes, camera frames, protected replay identities or a self-contained simulator.

## Dependencies

- Ubuntu 20.04, ROS Noetic and Gazebo Classic 11 (historical version 11.15.1).
- [ITAndroids humanoid repository](https://gitlab.com/itandroids/projects/humanoid/itandroids-humanoide),
  pinned historical base `6740b3321d6521721f4e2937d0008a5f0d1f4c02`.
- The upstream robot/world assets, ROS messages and build dependencies.
- Explicit local detector exports. No model auto-download.

## Components

`scenarios/vision-suite.json` is the historical scenario recipe.
`ros/scenario_capture_node.py` coordinates the scenario and synchronized capture.
`ros/generate_synthetic_gt.py` constructs supported geometric ground truth.
`ros/detector_adapter.py` and `ros/replay_runner.py` expose deterministic detector
replay. `gazebo/` holds minimal camera/world scaffolding, not the complete robot.
`patches/` contains the research detector overlay, timing instrumentation and
perception-only player integration. Review them against the pinned upstream
source before applying with `git apply --check` in an isolated checkout.

Set `PERCEPTION_REPLAY_ROOT` to a new authorized capture directory. Inspect each
script's CLI and the scenario configuration before running ROS. No capture,
simulation launch or robot-service changes occur on package import or in CI.
These integration files require the upstream environment and were not rebuilt
against a new ROS installation during publication.

## Evidence boundary

The original suite contained 16 scenarios and 2,250 frames. Ball and goalpost
boxes and typed L/T centers have supported synthetic ground truth. Robot/X box
scores and exact occlusion reacquisition are unsupported. Camera projection,
visibility rules and known ground-plane assumptions must accompany results.

Recreating the same scenario does not produce independent final evidence. The
original perception replay lacks a valid odometry stream; localization studies
used separately declared synthetic command odometry. Simulator pose is ground
truth only. See the [benchmark report](../docs/reports/deterministic-gazebo-perception-benchmark.md).

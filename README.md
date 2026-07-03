# HoST Isaac Lab

An Isaac Lab port of [HoST: Humanoid Standing-up Control](https://github.com/OpenRobotLab/HoST),
built as an extension task for
[unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab).

The current task targets the Unitree G1 29-DoF model:

```text
Unitree-G1-29dof-HoST-Ground
```

## Current implementation

- G1 29-DoF ground standing-up task
- prone reset pose
- relative joint-position actions
- 94-value observation frames with six-frame history
- multiplicative upright/head-height objective
- regularization, style, and post-stand rewards
- upward assistance force and basic domain randomization
- RSL-RL PPO configuration

This repository contains only the HoST task extension. Isaac Lab,
`unitree_rl_lab`, robot assets, and training scripts remain external
dependencies.

## Installation

Install Isaac Lab and `unitree_rl_lab` first. The initial port follows the
versions currently used by upstream `unitree_rl_lab`:

- Isaac Sim 5.1
- Isaac Lab 2.3

Copy the root-level `host` package into a `unitree_rl_lab` checkout:

```bash
cp -r host \
  /path/to/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/
```

For example, when both repositories are in the same parent directory:

```bash
cp -r host_isaaclab/host unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/
```

Configure `UNITREE_MODEL_DIR` in:

```text
unitree_rl_lab/assets/robots/unitree.py
```

## Smoke test

From the `unitree_rl_lab` root:

```bash
./unitree_rl_lab.sh -p scripts/list_envs.py

./unitree_rl_lab.sh -p scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-HoST-Ground \
  --num_envs 32 \
  --max_iterations 2 \
  --headless
```

## Porting status

This is a migration baseline and has only received Windows-side static
validation so far. Ubuntu Isaac Lab runtime validation is still required.

The following original HoST features remain to be ported:

- four-critic reward-group PPO
- per-environment pulling-force/action-scale curriculum
- initial unactuated phase and randomized action delay
- full actuator and rigid-body domain randomization
- exact conversion of HoST auxiliary/keyframe bodies
- wall, slope, platform, and additional prone variants

## License

MIT. See [LICENSE](LICENSE).

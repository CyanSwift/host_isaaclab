"""Reset and assistance events for HoST."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def reset_joints_host(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    scale_range: tuple[float, float],
    offset_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset joints with HoST's multiplicative and additive randomization."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    default = asset.data.default_joint_pos[env_ids][:, joint_ids]
    scale = torch.empty_like(default).uniform_(*scale_range)
    offset = torch.empty_like(default).uniform_(*offset_range)
    positions = default * scale + offset
    limits = asset.data.soft_joint_pos_limits[env_ids][:, joint_ids]
    positions = torch.clamp(positions, min=limits[..., 0], max=limits[..., 1])
    velocities = torch.zeros_like(positions)
    asset.write_joint_state_to_sim(positions, velocities, joint_ids=joint_ids, env_ids=env_ids)


def apply_upward_assistance(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    force: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Apply a persistent upward force to selected robot bodies.

    Isaac Lab keeps external wrench buffers active until they are overwritten.
    This is the initial equivalent of HoST's vertical pulling assistance.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(asset.num_bodies))[body_ids]
    count = len(body_ids)
    forces = torch.zeros((len(env_ids), count, 3), device=env.device)
    torques = torch.zeros_like(forces)
    forces[..., 2] = force
    asset.set_external_force_and_torque(forces, torques, body_ids=body_ids, env_ids=env_ids)

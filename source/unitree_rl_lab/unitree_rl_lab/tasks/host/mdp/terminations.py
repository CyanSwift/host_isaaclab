"""HoST termination terms."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def excessive_joint_velocity(
    env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.any(torch.abs(robot.data.joint_vel[:, asset_cfg.joint_ids]) > limit, dim=1)


def excessive_base_velocity(
    env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.vector_norm(robot.data.root_lin_vel_b, dim=1) > limit

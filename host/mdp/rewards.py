"""Reward terms translated from HoST's ``host_ground.py``."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg


def _asset(env: ManagerBasedRLEnv, cfg: SceneEntityCfg) -> Articulation:
    return env.scene[cfg.name]


def _tolerance(
    value: torch.Tensor, lower: float, upper: float, margin: float, value_at_margin: float
) -> torch.Tensor:
    """Gaussian ``dm_control``-style tolerance used by the original HoST."""
    inside = (value >= lower) & (value <= upper)
    distance = torch.where(value < lower, lower - value, torch.clamp(value - upper, min=0.0))
    if margin <= 0.0:
        return inside.to(value.dtype)
    scale = torch.sqrt(torch.as_tensor(-2.0 * torch.log(torch.tensor(value_at_margin)), device=value.device))
    shaped = torch.exp(-0.5 * torch.square(distance / margin * scale))
    return torch.where(inside, torch.ones_like(shaped), shaped)


def stand_up_task(
    env: ManagerBasedRLEnv,
    target_head_height: float,
    head_margin: float,
    orientation_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    head_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Multiplicative HoST task reward: upright orientation × relative head height."""
    robot = _asset(env, asset_cfg)
    upright = _tolerance(
        -robot.data.projected_gravity_b[:, 2], orientation_threshold, float("inf"), 1.0, 0.05
    )
    head_z = robot.data.body_pos_w[:, head_cfg.body_ids, 2].mean(dim=1)
    feet_z = robot.data.body_pos_w[:, feet_cfg.body_ids, 2].mean(dim=1)
    height = _tolerance(head_z - feet_z, target_head_height, float("inf"), head_margin, 0.1)
    return upright * height


def joint_power_l1(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot = _asset(env, asset_cfg)
    return torch.sum(torch.abs(robot.data.joint_vel[:, asset_cfg.joint_ids] * robot.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_tracking_error(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot = _asset(env, asset_cfg)
    return torch.sum(torch.square(robot.data.joint_pos_target[:, asset_cfg.joint_ids] - robot.data.joint_pos[:, asset_cfg.joint_ids]), dim=1)


def joint_threshold(
    env: ManagerBasedRLEnv,
    lower: float | None = None,
    upper: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    pos = _asset(env, asset_cfg).data.joint_pos[:, asset_cfg.joint_ids]
    failed = torch.zeros(pos.shape[0], dtype=torch.bool, device=pos.device)
    if lower is not None:
        failed |= torch.any(pos < lower, dim=1)
    if upper is not None:
        failed |= torch.any(pos > upper, dim=1)
    return failed.float()


def paired_joint_deviation(
    env: ManagerBasedRLEnv,
    max_abs: float,
    both_abs: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Match HoST's bilateral hip threshold logic."""
    pos = torch.abs(_asset(env, asset_cfg).data.joint_pos[:, asset_cfg.joint_ids])
    return ((torch.max(pos, dim=1).values > max_abs) | (torch.min(pos, dim=1).values > both_abs)).float()


def joint_velocity_soft_limits(
    env: ManagerBasedRLEnv,
    soft_limit: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """HoST velocity-limit cost, clipped to one per joint."""
    robot = _asset(env, asset_cfg)
    ids = asset_cfg.joint_ids
    excess = torch.abs(robot.data.joint_vel[:, ids]) - robot.data.joint_vel_limits[:, ids] * soft_limit
    return torch.sum(excess.clamp(min=0.0, max=1.0), dim=1)


class action_smoothness_l2(ManagerTermBase):
    """Second-order action difference used by HoST."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_previous_action = torch.zeros_like(env.action_manager.action)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._previous_previous_action.zero_()
        else:
            self._previous_previous_action[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        current = env.action_manager.action
        previous = env.action_manager.prev_action
        value = torch.sum(torch.square(current - 2.0 * previous + self._previous_previous_action), dim=1)
        self._previous_previous_action.copy_(previous)
        return value


def shoulder_roll_deviation(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    pos = _asset(env, asset_cfg).data.joint_pos[:, asset_cfg.joint_ids]
    return ((pos[:, 0] < -0.02) | (pos[:, 1] > 0.02)).float()


def foot_displacement(
    env: ManagerBasedRLEnv,
    stand_height: float,
    sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = _asset(env, asset_cfg)
    foot = robot.data.body_pos_w[:, foot_cfg.body_ids, :].mean(dim=1)
    error = torch.sum(torch.square(robot.data.root_pos_w[:, :2] - foot[:, :2]), dim=1).clamp(min=0.3)
    return torch.exp(error * sigma) * (foot[:, 2] < 0.3) * (robot.data.root_pos_w[:, 2] > stand_height)


def shank_orientation(
    env: ManagerBasedRLEnv,
    base_height: float,
    knees_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = _asset(env, knees_cfg)
    knee = robot.data.body_pos_w[:, knees_cfg.body_ids]
    foot = robot.data.body_pos_w[:, feet_cfg.body_ids]
    direction = knee - foot
    vertical = direction[..., 2] / torch.linalg.vector_norm(direction, dim=-1).clamp(min=1.0e-6)
    shaped = _tolerance(vertical.mean(dim=1), 0.8, float("inf"), 1.0, 0.1)
    return shaped * (robot.data.root_pos_w[:, 2] > base_height)


def excessive_feet_distance(
    env: ManagerBasedRLEnv, max_distance: float, feet_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    feet = _asset(env, feet_cfg).data.body_pos_w[:, feet_cfg.body_ids]
    return (torch.linalg.vector_norm(feet[:, 0] - feet[:, 1], dim=1) > max_distance).float()


def ground_parallel(
    env: ManagerBasedRLEnv,
    threshold: float,
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward parallel ankle contact geometry, as in HoST's auxiliary ankle links."""
    robot = _asset(env, left_cfg)
    left_z = robot.data.body_pos_w[:, left_cfg.body_ids, 2] * 10.0
    right_z = robot.data.body_pos_w[:, right_cfg.body_ids, 2] * 10.0
    variance = 0.5 * (left_z.var(dim=1, unbiased=False) + right_z.var(dim=1, unbiased=False))
    return (variance < threshold).float()


def feet_height_similarity(
    env: ManagerBasedRLEnv,
    stand_height: float,
    sigma: float,
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = _asset(env, feet_cfg)
    feet_z = robot.data.body_pos_w[:, feet_cfg.body_ids, 2]
    difference = torch.abs(feet_z[:, 0] - feet_z[:, 1]).mul(10.0).clamp(min=0.2)
    return torch.exp(sigma * difference) * (robot.data.root_pos_w[:, 2] > stand_height)


def stable_base_motion(
    env: ManagerBasedRLEnv,
    stand_height: float,
    velocity: str,
    sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = _asset(env, asset_cfg)
    value = robot.data.root_ang_vel_b[:, :2] if velocity == "angular" else robot.data.root_lin_vel_b[:, :2]
    return torch.exp(sigma * torch.sum(torch.square(value), dim=1)) * (robot.data.root_pos_w[:, 2] > stand_height)


def target_upper_body_pose(
    env: ManagerBasedRLEnv,
    stand_height: float,
    sigma: float,
    target_positions: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = _asset(env, asset_cfg)
    ids = asset_cfg.joint_ids
    if isinstance(ids, slice):
        ids = list(range(robot.num_joints))[ids]
    target = torch.zeros_like(robot.data.joint_pos[:, ids])
    for column, joint_id in enumerate(ids):
        target[:, column] = target_positions.get(robot.joint_names[joint_id], 0.0)
    error = torch.sum(torch.square(robot.data.joint_pos[:, ids] - target), dim=1)
    return torch.exp(sigma * error) * (robot.data.root_pos_w[:, 2] > stand_height)


def target_orientation(
    env: ManagerBasedRLEnv, stand_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot = _asset(env, asset_cfg)
    return torch.exp(-5.0 * torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)) * (
        robot.data.root_pos_w[:, 2] > stand_height
    )


def target_base_height(
    env: ManagerBasedRLEnv,
    stand_height: float,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    height = _asset(env, asset_cfg).data.root_pos_w[:, 2]
    return torch.exp(-20.0 * torch.abs(height - target_height)) * (height > stand_height)

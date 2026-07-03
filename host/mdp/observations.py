"""Observations used by HoST.

The original policy observes one curriculum action-scale value in addition to
angular velocity, projected gravity, joint state, and the previous action.
The first Isaac Lab port keeps that value fixed at one; the force/action-scale
curriculum can later replace this term without changing the policy shape.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv


def action_scale(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the HoST action-scale observation (shape ``num_envs x 1``)."""
    return torch.ones((env.num_envs, 1), device=env.device)

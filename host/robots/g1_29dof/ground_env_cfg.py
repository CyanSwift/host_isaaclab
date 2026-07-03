"""Isaac Lab port of HoST's Unitree G1 ground standing-up task."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG
from unitree_rl_lab.tasks.host import mdp


@configclass
class HostGroundSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.7,
            restitution=0.3,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0.005, history_length=3, track_air_time=False
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.8, 0.8, 0.8))
    )


@configclass
class ActionsCfg:
    # HoST uses q_target = q_current + action. This is not the usual default-pose
    # offset used by locomotion policies.
    joint_position = mdp.RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, zero_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, noise=Unoise(n_min=-0.05, n_max=0.05))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        action_scale = ObsTerm(func=mdp.action_scale)

        def __post_init__(self):
            self.history_length = 6
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        action_scale = ObsTerm(func=mdp.action_scale)

        def __post_init__(self):
            self.history_length = 6
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 1.0),
            "dynamic_friction_range": (0.1, 1.0),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 64,
        },
    )
    torso_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-2.0, 5.0),
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_host,
        mode="reset",
        params={"scale_range": (0.9, 1.1), "offset_range": (-0.1, 0.1)},
    )
    upward_assistance = EventTerm(
        func=mdp.apply_upward_assistance,
        mode="reset",
        params={"force": 200.0, "asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )


@configclass
class RewardsCfg:
    # HoST task group: 2.5 * orientation * head-height.
    stand_up = RewTerm(
        func=mdp.stand_up_task,
        weight=2.5,
        params={
            "target_head_height": 1.0,
            "head_margin": 1.0,
            "orientation_threshold": 0.99,
            "head_cfg": SceneEntityCfg("robot", body_names="head_link"),
            "feet_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
        },
    )

    # Regularization group (HoST group weight 0.1).
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-8)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-3)
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.5e-7)
    joint_power = RewTerm(func=mdp.joint_power_l1, weight=-2.5e-6)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)
    joint_tracking = RewTerm(func=mdp.joint_tracking_error, weight=-2.5e-5)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)

    # Style group.
    waist_deviation = RewTerm(
        func=mdp.joint_threshold,
        weight=-10.0,
        params={
            "lower": -1.4,
            "upper": 1.4,
            "asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint"),
        },
    )
    hip_yaw_deviation = RewTerm(
        func=mdp.joint_threshold,
        weight=-10.0,
        params={"lower": -1.4, "upper": 1.4, "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_yaw_joint")},
    )
    hip_roll_deviation = RewTerm(
        func=mdp.joint_threshold,
        weight=-10.0,
        params={"lower": -1.4, "upper": 1.4, "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_roll_joint")},
    )
    shoulder_roll_deviation = RewTerm(
        func=mdp.shoulder_roll_deviation,
        weight=-2.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_shoulder_roll_joint", "right_shoulder_roll_joint"])},
    )
    left_foot_displacement = RewTerm(
        func=mdp.foot_displacement,
        weight=2.5,
        params={"stand_height": 0.65, "sigma": -2.0, "foot_cfg": SceneEntityCfg("robot", body_names="left_ankle_roll_link")},
    )
    right_foot_displacement = RewTerm(
        func=mdp.foot_displacement,
        weight=2.5,
        params={"stand_height": 0.65, "sigma": -2.0, "foot_cfg": SceneEntityCfg("robot", body_names="right_ankle_roll_link")},
    )
    knee_deviation = RewTerm(
        func=mdp.joint_threshold,
        weight=-0.25,
        params={"lower": -0.06, "upper": 2.85, "asset_cfg": SceneEntityCfg("robot", joint_names=".*_knee_joint")},
    )
    shank_orientation = RewTerm(
        func=mdp.shank_orientation,
        weight=10.0,
        params={
            "base_height": 0.45,
            "knees_cfg": SceneEntityCfg("robot", body_names=["left_knee_link", "right_knee_link"]),
            "feet_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
        },
    )
    ground_parallel = RewTerm(
        func=mdp.ground_parallel,
        weight=20.0,
        params={
            "threshold": 0.05,
            "left_cfg": SceneEntityCfg("robot", body_names="(left_ankle_roll_link|auxiliary_left_ankle_roll_link.*)"),
            "right_cfg": SceneEntityCfg("robot", body_names="(right_ankle_roll_link|auxiliary_right_ankle_roll_link.*)"),
        },
    )
    feet_distance = RewTerm(
        func=mdp.excessive_feet_distance,
        weight=-10.0,
        params={
            "max_distance": 0.9,
            "feet_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
        },
    )
    style_ang_vel_xy = RewTerm(
        func=mdp.stable_base_motion,
        weight=1.0,
        params={"stand_height": 0.45, "velocity": "angular", "sigma": -2.0},
    )

    # Post-task target group.
    target_ang_vel_xy = RewTerm(
        func=mdp.stable_base_motion,
        weight=10.0,
        params={"stand_height": 0.65, "velocity": "angular", "sigma": -2.0},
    )
    target_lin_vel_xy = RewTerm(
        func=mdp.stable_base_motion,
        weight=10.0,
        params={"stand_height": 0.65, "velocity": "linear", "sigma": -5.0},
    )
    feet_height = RewTerm(
        func=mdp.feet_height_similarity,
        weight=2.5,
        params={
            "stand_height": 0.65,
            "sigma": -2.0,
            "feet_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
        },
    )
    upper_body_pose = RewTerm(
        func=mdp.target_upper_body_pose,
        weight=10.0,
        params={
            "stand_height": 0.65,
            "sigma": -0.1,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist_.*_joint",
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            ),
        },
    )
    target_orientation = RewTerm(func=mdp.target_orientation, weight=10.0, params={"stand_height": 0.65})
    target_height = RewTerm(
        func=mdp.target_base_height,
        weight=10.0,
        params={"stand_height": 0.65, "target_height": 0.75},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    joint_velocity = DoneTerm(func=mdp.excessive_joint_velocity, params={"limit": 300.0})
    base_velocity = DoneTerm(func=mdp.excessive_base_velocity, params={"limit": 20.0})


@configclass
class HostG129DofGroundEnvCfg(ManagerBasedRLEnvCfg):
    scene: HostGroundSceneCfg = HostGroundSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.solver_type = 1
        self.sim.physx.min_position_iteration_count = 8
        self.sim.physx.min_velocity_iteration_count = 1
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # HoST starts G1 on the ground, rotated 90 degrees around -Y.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.5)
        half = math.sqrt(0.5)
        self.scene.robot.init_state.rot = (half, 0.0, -half, 0.0)  # Isaac Lab uses wxyz
        self.scene.robot.init_state.joint_pos = {
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_ankle_roll_joint": 0.0,
            "waist_.*_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.0,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.8,
            ".*_wrist_.*_joint": 0.0,
        }


@configclass
class HostG129DofGroundPlayEnvCfg(HostG129DofGroundEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False

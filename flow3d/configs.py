from dataclasses import dataclass


@dataclass
class FGLRConfig:
    means: float = 1.6e-4
    opacities: float = 1e-2
    scales: float = 5e-3
    quats: float = 1e-3
    colors: float = 1e-2
    motion_coefs: float = 1e-2
    traj_coefs: float = 1.6e-4


@dataclass
class BGLRConfig:
    means: float = 1.6e-4
    opacities: float = 5e-2
    scales: float = 5e-3
    quats: float = 1e-3
    colors: float = 1e-2


@dataclass
class MotionLRConfig:
    rots: float = 1.6e-4
    transls: float = 1.6e-4

@dataclass
class CameraScalesLRConfig:
    camera_scales: float = 1e-4

@dataclass
class CameraPoseLRConfig:
    Rs: float = 1e-3
    ts: float = 1e-3


@dataclass
class ModalJointLRConfig:
    delta_coordinate_real: float = 1e-4
    delta_coordinate_imag: float = 1e-4
    delta_phi_real: float = 1e-4
    delta_phi_imag: float = 1e-4


@dataclass
class ModalPhiRefinementLRConfig:
    delta_phi_real: float = 1e-4
    delta_phi_imag: float = 1e-4


@dataclass
class SceneLRConfig:
    fg: FGLRConfig
    bg: BGLRConfig
    motion_bases: MotionLRConfig
    camera_poses: CameraPoseLRConfig
    camera_scales: CameraScalesLRConfig
    modal_joint: ModalJointLRConfig
    modal_phi_refinement: ModalPhiRefinementLRConfig


@dataclass
class LossesConfig:
    w_rgb: float = 1.0
    w_depth_reg: float = 0.5
    w_depth_const: float = 0.0
    w_depth_grad: float = 1.0
    w_track: float = 0.0
    w_mask: float = 1.0
    w_smooth_bases: float = 0.1
    w_smooth_tracks: float = 2.0
    w_scale_var: float = 0.01
    w_z_accel: float = 1.0
    w_dct_coef: float = 1e-4
    w_local_iso_ray: float = 0.0
    w_local_iso_perp: float = 0.00
    w_local_iso_dist: float = 0.0
    local_iso_knn: int = 6
    local_iso_radius_mult: float = 2.0
    local_iso_huber_beta: float = 0.05
    local_iso_start_step: int = 100
    local_iso_edge_weight_temp: float = 1.0
    w_modal_flow: float = 1.0
    w_modal_rigidity: float = 0.1
    w_modal_mode_rigidity: float = 1.0
    w_modal_delta_phi_local: float = 0.05
    modal_structure_modes_per_step: int = 3
    modal_rigidity_huber_beta: float = 0.01
    w_modal_coordinate_prior: float = 0.1
    w_modal_coordinate_temporal: float = 0.01
    modal_coordinate_temporal_scale_sec: float = 0.5
    w_modal_phi_prior: float = 0.01
    modal_flow_charbonnier_epsilon_px: float = 0.5
    modal_flow_render_acc_min: float = 0.05

    # w_smooth_bases: float = 0.0
    # w_smooth_tracks: float = 0.0
    # w_scale_var: float = 0.0
    # w_z_accel: float = 0.0


@dataclass
class OptimizerConfig:
    max_steps: int = 5000
    ## Adaptive gaussian control
    warmup_steps: int = 200
    control_every: int = 100
    reset_opacity_every_n_controls: int = 30
    stop_control_by_screen_steps: int = 4000
    stop_control_steps: int = 4000
    ### Densify.
    densify_xys_grad_threshold: float = 0.0002
    densify_scale_threshold: float = 0.01
    densify_screen_threshold: float = 0.05
    stop_densify_steps: int = 15000
    ### Cull.
    cull_opacity_threshold: float = 0.1
    cull_scale_threshold: float = 0.5
    cull_screen_threshold: float = 0.15

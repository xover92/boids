import numpy as np
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class commands:
    method: ClassVar[str] = "reynolds"  # "reynolds", "couzin", "vicsek"
    obstacle_bool: ClassVar[bool] = True
    predator_bool: ClassVar[bool] = False
    moving_camera_bool: ClassVar[bool] = True
    gif_making_bool: ClassVar[bool] = True
    artistic_rendition_bool: ClassVar[bool] = False
    make_csv_bool: ClassVar[bool] = False
    plot_correlation_function: ClassVar[bool] = False
    compute_polarization: ClassVar[bool] = False


@dataclass(frozen=True)
class glob_const:
    n_boids: ClassVar[int] = 200
    boids_in_pos_ub: ClassVar[float] = 5.0
    fov_angle: ClassVar[float] = np.radians(145)
    cos_fov: ClassVar[float] = np.cos(fov_angle)
    time_steps: ClassVar[int] = 200


@dataclass(frozen=True)
class reynolds_const:
    coh_par: ClassVar[float] = 10
    ali_par: ClassVar[float] = 0.20
    sep_par: ClassVar[float] = 0.20
    max_speed: ClassVar[float] = 5.0
    min_speed: ClassVar[float] = 4.0
    max_delta: ClassVar[float] = 0.20
    action_range: ClassVar[float] = 20.0


@dataclass(frozen=True)
class couzin_const:
    ang_noi_par: ClassVar[float] = np.deg2rad(2)
    zoa: ClassVar[float] = 25
    zoo: ClassVar[float] = 20
    zor: ClassVar[float] = 1.0
    max_turn_angle: ClassVar[float] = np.deg2rad(10)
    speed: ClassVar[float] = 4.0


@dataclass(frozen=True)
class vicsek_const:
    ang_noi_par: ClassVar[float] = np.deg2rad(2)
    speed: ClassVar[float] = 4.0
    action_range: ClassVar[float] = 20.0


@dataclass(frozen=True)
class predator_const:
    max_speed: ClassVar[float] = 10.0
    min_speed: ClassVar[float] = 5.0
    max_delta: ClassVar[float] = 3.0
    att_par: ClassVar[float] = 20.0
    sep_par: ClassVar[float] = 10.0
    dist_par: ClassVar[float] = 40.0
    low_spawn: ClassVar[float] = 100.0
    high_spawn: ClassVar[float] = 120.0


@dataclass
class obstacles_const:
    n_cols: ClassVar[int] = 1
    n_per_col: ClassVar[int] = 30
    x_vals = np.linspace(0.0, 0.0, n_cols)
    z_vals = np.linspace(-30.0, 30.0, n_per_col)
    xx, zz = np.meshgrid(x_vals, z_vals)
    positions: ClassVar[np.ndarray] = np.column_stack((
        xx.flatten(),
        np.full(xx.size, -300.0),
        zz.flatten()
    ))
    action_range: ClassVar[float] = 5
    rep_par: ClassVar[float] = 2

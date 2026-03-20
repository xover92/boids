import numpy as np
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class commands:
    method: ClassVar[str] = "reynolds"  # "reynolds", "couzin", "vicsek"
    obstacle_bool: ClassVar[bool] = False 
    predator_bool: ClassVar[bool] = False
    moving_camera_bool: ClassVar[bool] = True
    gif_making_bool: ClassVar[bool] = True
    artistic_rendition_bool: ClassVar[bool] = True
    make_csv_bool: ClassVar[bool] = False
    plot_correlation_function: ClassVar[bool] = False
    compute_polarization: ClassVar[bool] = False

@dataclass(frozen=True)
class glob_const:
    n_boids: ClassVar[int] = 100
    boids_in_pos_ub: ClassVar[float] = n_boids/40 #fixed
    fov_angle: ClassVar[float] = np.radians(145) #fixed
    cos_fov: ClassVar[float] = np.cos(fov_angle) #fixed
    max_speed: ClassVar[float] = 5.0 #fixed
    time_steps: ClassVar[int] = 400


@dataclass(frozen=True)
class reynolds_const:
    coh_par: ClassVar[float] = 15
    ali_par: ClassVar[float] = 0.20
    sep_par: ClassVar[float] = 0.20
    action_range: ClassVar[float] = 20.0
    min_speed: ClassVar[float] = 4.0
    max_delta: ClassVar[float] = 0.25

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
    ang_noi_par: ClassVar[float] = np.deg2rad(10)
    speed: ClassVar[float] = 4.0
    action_range: ClassVar[float] = 20.0


@dataclass(frozen=True)
class predator_const:
    max_speed: ClassVar[float] = 40.0
    min_speed: ClassVar[float] = 0.0
    max_delta: ClassVar[float] = 10.0
    att_par: ClassVar[float] = 6
    sep_par: ClassVar[float] = 150.0
    dist_par: ClassVar[float] = 80.0
    low_spawn: ClassVar[float] = 50.0
    high_spawn: ClassVar[float] = 60.0


@dataclass
class obstacles_const:
    n_cols: ClassVar[int] = 3
    n_per_col: ClassVar[int] = 30
    y_vals = np.linspace(-3.0, 3.0, n_cols)
    z_vals = np.linspace(-30.0, 30.0, n_per_col)
    yy, zz = np.meshgrid(y_vals, z_vals)
    positions: ClassVar[np.ndarray] = np.column_stack((
        np.full(yy.size, 300.0),
        yy.flatten(),
        zz.flatten()
    ))
    action_range: ClassVar[float] = 10
    rep_par: ClassVar[float] = 3

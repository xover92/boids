import numpy as np
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class commands:
    method: ClassVar[str] = "reynolds"
    obstacle_bool: ClassVar[bool] = True
    predator_bool: ClassVar[bool] = False
    moving_camera_bool: ClassVar[bool] = True
    gif_making_bool: ClassVar[bool] = True
    artistic_rendition_bool: ClassVar[bool] = False
    make_csv_bool: ClassVar[bool] = True
    plot_correlation_function: ClassVar[bool] = True


@dataclass(frozen=True)
class glob_const:
    n_boids: ClassVar[int] = 300
    boids_in_vel_std: ClassVar[float] = 0.5
    boids_in_pos_std: ClassVar[float] = 10.0
    action_range: ClassVar[float] = 22.0
    fov_angle: ClassVar[float] = np.radians(135)
    cos_fov: ClassVar[float] = np.cos(fov_angle)
    time_steps: ClassVar[int] = 300


@dataclass(frozen=True)
class reynolds_const:
    coh_par: ClassVar[float] = 10
    ali_par: ClassVar[float] = 0.25
    sep_par: ClassVar[float] = 0.25
    noi_par: ClassVar[float] = 0.3
    max_speed: ClassVar[float] = 5.0
    min_speed: ClassVar[float] = 4.0
    max_delta: ClassVar[float] = 0.25


@dataclass(frozen=True)
class predator_const:
    max_speed: ClassVar[float] = 30.0
    min_speed: ClassVar[float] = 6.0
    max_delta: ClassVar[float] = 3.0
    att_par: ClassVar[float] = 6
    sep_par: ClassVar[float] = 150.0
    dist_par: ClassVar[float] = 80.0
    low_spawn: ClassVar[float] = 50.0
    high_spawn: ClassVar[float] = 60.0


@dataclass(frozen=True)
class couzin_const:
    noi_par: ClassVar[float] = 0.3
    zoa: ClassVar[float] = 25
    zoo: ClassVar[float] = 20
    zor: ClassVar[float] = 1.0
    max_turn_angle: ClassVar[float] = np.radians(2.0)
    speed: ClassVar[float] = 4.0


# @dataclass
# class obstacles_const:
#     positions: ClassVar[np.array] = np.array([
#         [300.0, 0.0, -25.0],
#         [300.0, 0.0, -25.0],
#         [300.0, 0.0, -20.0],
#         [300.0, 0.0, -20.0],
#         [300.0, 0.0, -15.0],
#         [300.0, 0.0, -15.0],
#         [300.0, 0.0, -10.0],
#         [300.0, 0.0, -10.0],
#         [300.0, 0.0, -5.0],
#         [300.0, 0.0, -5.0],
#         [300.0, 0.0, 0.0],
#         [300.0, 0.0, 0.0],
#         [300.0, 0.0, 25.0],
#         [300.0, 0.0, 25.0],
#         [300.0, 0.0, 20.0],
#         [300.0, 0.0, 20.0],
#         [300.0, 0.0, 15.0],
#         [300.0, 0.0, 15.0],
#         [300.0, 0.0, 10.0],
#         [300.0, 0.0, 10.0],
#         [300.0, 0.0, 5.0],
#         [300.0, 0.0, 5.0],
#     ])
#     action_range: ClassVar[float] = 15.0
#     rep_par: ClassVar[float] = 20.0


@dataclass
class obstacles_const:
    n_cols: ClassVar[int] = 1       
    n_per_col: ClassVar[int] = 20   
    y_vals = np.linspace(-0.0, 0.0, n_cols)
    z_vals = np.linspace(-40.0, 40.0, n_per_col)
    yy, zz = np.meshgrid(y_vals, z_vals)
    positions: ClassVar[np.ndarray] = np.column_stack((
        np.full(yy.size, 300.0),  
        yy.flatten(),             
        zz.flatten()              
    ))
    action_range: ClassVar[float] = 20.0
    rep_par: ClassVar[float] = 3.0
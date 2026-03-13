import numpy as np
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class glob_const:
    n_boids: ClassVar[int] = 300
    boids_in_vel_std: ClassVar[float] = 0.5
    boids_in_pos_std: ClassVar[float] = 10.0
    max_speed: ClassVar[float] = 5.0
    min_speed: ClassVar[float] = 3.0
    max_delta: ClassVar[float] = 1.0
    action_range: ClassVar[float] = 50.0
    max_turn_angle: ClassVar[float] = np.radians(5.0) 
    fov_angle: ClassVar[float] = np.radians(135)
    cos_fov: ClassVar[float] = np.cos(fov_angle)
    time_steps: ClassVar[int] = 200
    method: ClassVar[str] = "couzin"


@dataclass(frozen=True)
class predator_const:
    max_speed: ClassVar[float] = 30.0
    min_speed: ClassVar[float] = 0.0
    max_delta: ClassVar[float] = 10.0
    att_par: ClassVar[float] = 60.0
    sep_par: ClassVar[float] = 150.0
    dist_par: ClassVar[float] = 100.0


@dataclass(frozen=True)
class commands:
    obstacle_bool: ClassVar[bool] = False
    predator_bool: ClassVar[bool] = False
    moving_camera_bool: ClassVar[bool] = True
    gif_making_bool: ClassVar[bool] = True
    artistic_rendition_bool: ClassVar[bool] = False
    make_csv_bool: ClassVar[bool] = True
    plot_correlation_function: ClassVar[bool] = True

@dataclass(frozen=True)
class reynolds_const:
    coh_par: ClassVar[float] = 0.35
    ali_par: ClassVar[float] = 0.3
    sep_par: ClassVar[float] = 0.1
    noi_par: ClassVar[float] = 0.3

@dataclass(frozen=True)
class couzin_const:
    coh_par: ClassVar[float] = 1.0
    ali_par: ClassVar[float] = 1.0
    sep_par: ClassVar[float] = 1.0
    noi_par: ClassVar[float] = 1.0
    zoa: ClassVar[float] = 30
    zoo: ClassVar[float] = 25
    zor: ClassVar[float] = 5.0


@dataclass
class obstacles_const:
    positions: ClassVar[np.array] = np.array([
        [80.0, 10.0, -5.0],
        [80.0, 10.0, 0.0],
        [80.0, 10.0, 5.0],
        [80.0, 10.0, 10.0],
        [80.0, 10.0, 15.0],
        [80.0, 10.0, 20.0],
        [80.0, 10.0, 25.0],
        [70.0, 10.0, -5.0],
        [70.0, 10.0, 0.0],
        [70.0, 10.0, 5.0],
        [70.0, 10.0, 10.0],
        [70.0, 10.0, 15.0],
        [70.0, 10.0, 20.0],
        [70.0, 10.0, 25.0],
    ])
    action_range: ClassVar[float] = 20.0
    rep_par: ClassVar[float] = 20.0

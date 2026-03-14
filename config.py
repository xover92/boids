import numpy as np
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class glob_const:
    n_boids: ClassVar[int] = 400
    boids_in_vel_std: ClassVar[float] = 0.5
    boids_in_pos_std: ClassVar[float] = 15.0
    action_range: ClassVar[float] = 80.0
    fov_angle: ClassVar[float] = np.radians(135)
    cos_fov: ClassVar[float] = np.cos(fov_angle)
    time_steps: ClassVar[int] = 200


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
class commands:
    method: ClassVar[str] = "couzin"
    obstacle_bool: ClassVar[bool] = False
    predator_bool: ClassVar[bool] = True
    moving_camera_bool: ClassVar[bool] = True
    gif_making_bool: ClassVar[bool] = True
    artistic_rendition_bool: ClassVar[bool] = False
    make_csv_bool: ClassVar[bool] = False
    plot_correlation_function: ClassVar[bool] = False

@dataclass(frozen=True)
class reynolds_const:
    coh_par: ClassVar[float] = 2
    ali_par: ClassVar[float] = 0.1
    sep_par: ClassVar[float] = 0.05
    noi_par: ClassVar[float] = 0.30
    max_speed: ClassVar[float] = 5.0
    min_speed: ClassVar[float] = 4.0
    max_delta: ClassVar[float] = 0.05

@dataclass(frozen=True)
class couzin_const:
    noi_par: ClassVar[float] = 1.0
    zoa: ClassVar[float] = 25
    zoo: ClassVar[float] = 15
    zor: ClassVar[float] = 3.0
    max_turn_angle: ClassVar[float] = np.radians(5.0) 
    speed: ClassVar[float] = 4.0



@dataclass
class obstacles_const:
    positions: ClassVar[np.array] = np.array([
        [80.0, 80.0, 65.0],
        [80.0, 80.0, 70.0],
        [80.0, 80.0, 75.0],
        [80.0, 80.0, 80.0],
        [80.0, 80.0, 85.0],
        [80.0, 80.0, 90.0],
        [80.0, 80.0, 95.0],
        [70.0, 80.0, 65.0],
        [70.0, 80.0, 70.0],
        [70.0, 80.0, 75.0],
        [70.0, 80.0, 80.0],
        [70.0, 80.0, 85.0],
        [70.0, 80.0, 90.0],
        [70.0, 80.0, 95.0],
    ])
    action_range: ClassVar[float] = 50.0
    rep_par: ClassVar[float] = 15.0

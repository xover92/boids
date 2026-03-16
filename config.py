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
    n_boids: ClassVar[int] = 400
    boids_in_vel_std: ClassVar[float] = 0.5
    boids_in_pos_std: ClassVar[float] = 10.0
    action_range: ClassVar[float] = 15.0
    fov_angle: ClassVar[float] = np.radians(135)
    cos_fov: ClassVar[float] = np.cos(fov_angle)
    time_steps: ClassVar[int] = 400


@dataclass(frozen=True)
class reynolds_const:
    coh_par: ClassVar[float] = 3.0
    ali_par: ClassVar[float] = 0.35
    sep_par: ClassVar[float] = 0.3
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


@dataclass
class obstacles_const:
    positions: ClassVar[np.array] = np.array([
        [180.0, 180.0, 165.0],
        [180.0, 170.0, 170.0],
        [180.0, 180.0, 175.0],
        [180.0, 170.0, 180.0],
        [180.0, 180.0, 185.0],
        [180.0, 170.0, 190.0],
        [180.0, 180.0, 195.0],
    ])
    action_range: ClassVar[float] = 10.0
    rep_par: ClassVar[float] = 25.0

import numpy as np
from dataclasses import dataclass
from typing import ClassVar
import matplotlib.pyplot as plt
import matplotlib.animation as animation


@dataclass(frozen=True)
class glob_const:
    n_boids: ClassVar[int] = 100
    boid_init_loc: ClassVar[np.ndarray] = np.array([3.0, 0.0, 0.0])
    boid_init_scale: ClassVar[float] = 0.5
    spawn_length: ClassVar[int] = 40
    max_speed: ClassVar[float] = 5.0
    min_speed: ClassVar[float] = 2.0
    max_delta: ClassVar[float] = 1.0
    action_range: ClassVar[float] = 50.0
    fov_angle: ClassVar[float] = np.radians(135)
    cos_fov: ClassVar[float] = np.cos(fov_angle)
    time_steps: ClassVar[int] = 300
    method: ClassVar[str] = "reynolds"


@dataclass(frozen=True)
class commands:
    obstacle_bool: ClassVar[bool] = False
    predator_bool: ClassVar[bool] = False
    moving_camera_bool: ClassVar[bool] = True
    gif_making_bool: ClassVar[bool] = False
    artistic_rendition_bool: ClassVar[bool] = False
    make_csv_bool: ClassVar[bool] = True
    plot_correlation_function: ClassVar[bool] = True

@dataclass(frozen=True)
class reynolds_const:
    coh_par: ClassVar[float] = 0.10
    ali_par: ClassVar[float] = 0.25
    sep_par: ClassVar[float] = 0.35
    noi_par: ClassVar[float] = 0.30


@dataclass(frozen=True)
class couzin_const:
    coh_par: ClassVar[float] = 0.1
    ali_par: ClassVar[float] = 0.1
    sep_par: ClassVar[float] = 0.1
    noi_par: ClassVar[float] = 0.1
    zoa: ClassVar[float] = 25
    zoo: ClassVar[float] = 10
    zor: ClassVar[float] = 1.0


@dataclass(frozen=True)
class vicsek_const:
    mex_delta: ClassVar[float] = 0.2
    noi_par: ClassVar[float] = 0.8


@dataclass(frozen=True)
class predator_const:
    max_speed: ClassVar[float] = 20.0
    min_speed: ClassVar[float] = 5.0
    max_delta: ClassVar[float] = 1.5
    att_par: ClassVar[float] = 0.8
    sep_par: ClassVar[float] = 25.0
    dist_par: ClassVar[float] = 20.0
    init_pos: ClassVar[np.ndarray] = np.array([[100.0, 20.0, 20.0]])
    init_vel: ClassVar[np.ndarray] = np.array([[-10.0, 0.0, 0.0]])


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
        # [70.0, 10.0, -5.0],
        # [70.0, 10.0, 0.0],
        # [70.0, 10.0, 5.0],
        # [70.0, 10.0, 10.0],
        # [70.0, 10.0, 15.0],
        # [70.0, 10.0, 20.0],
        # [70.0, 10.0, 25.0],
    ])
    action_range: ClassVar[float] = 10.0
    rep_par: ClassVar[float] = 6.0

import numpy as np
from dataclasses import dataclass
from typing import ClassVar
from itertools import product

@dataclass(frozen=True)
class glob_const:
    n_boids: ClassVar[int] = 100
    boids_in_vel_std: ClassVar[float] = 0.5
    boids_in_pos_std: ClassVar[float] = 30.0
    max_speed: ClassVar[float] = 50.0
    min_speed: ClassVar[float] = 2.0
    max_delta: ClassVar[float] = 5
    action_range: ClassVar[float] = 20.0
    max_turn_angle: ClassVar[float] = np.radians(5.0) 
    fov_angle: ClassVar[float] = np.radians(135)
    cos_fov: ClassVar[float] = np.cos(fov_angle)
    time_steps: ClassVar[int] = 300
    method: ClassVar[str] = "reynolds"


@dataclass(frozen=True)
class predator_const:
    max_speed: ClassVar[float] = 40.0
    min_speed: ClassVar[float] = 0.0
    max_delta: ClassVar[float] = 10.0
    att_par: ClassVar[float] = 6
    sep_par: ClassVar[float] = 550.0
    dist_par: ClassVar[float] = 50.0


@dataclass(frozen=True)
class commands:
    obstacle_bool: ClassVar[bool] = True
    predator_bool: ClassVar[bool] = False
    moving_camera_bool: ClassVar[bool] = True
    gif_making_bool: ClassVar[bool] = True
    artistic_rendition_bool: ClassVar[bool] = False
    make_csv_bool: ClassVar[bool] = True
    plot_correlation_function: ClassVar[bool] = True #makecsv must be true too

@dataclass(frozen=True)
class reynolds_const:
    coh_par: ClassVar[float] = 1
    ali_par: ClassVar[float] = 0.25
    sep_par: ClassVar[float] = 0.005
    noi_par: ClassVar[float] = 0.30

@dataclass(frozen=True)
class couzin_const:
    coh_par: ClassVar[float] = 1.0
    ali_par: ClassVar[float] = 1.0
    sep_par: ClassVar[float] = 1.0
    noi_par: ClassVar[float] = 1.0
    zoa: ClassVar[float] = 35
    zoo: ClassVar[float] = 15
    zor: ClassVar[float] = 3.0

y_vals = list(range(-160, 160, 5))
z_vals = list(range(-140, 140, 5))

res = np.array([[280.0, y, z] for y, z in product(y_vals, z_vals)])

@dataclass
class obstacles_const:
    positions: ClassVar[np.array] = res
    action_range: ClassVar[float] = 30.0
    rep_par: ClassVar[float] = 20.0

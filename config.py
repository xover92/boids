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
    make_csv_bool: ClassVar[bool] = True
    plot_correlation_function: ClassVar[bool] = True
    compute_polarization: ClassVar[bool] = True


@dataclass(frozen=True)
class glob_const:
    n_boids: ClassVar[int] = 1000
    boids_in_pos_ub: ClassVar[float] = n_boids/40  # fixed
    fov_angle: ClassVar[float] = np.radians(145)  # fixed
    cos_fov: ClassVar[float] = np.cos(fov_angle)  # fixed
    max_speed: ClassVar[float] = 4.0  # fixed
    time_steps: ClassVar[int] = 500


@dataclass(frozen=True)
class reynolds_const:
    coh_par: ClassVar[float] = 5.5
    ali_par: ClassVar[float] = 0.8
    sep_par: ClassVar[float] = 0.5
    action_range: ClassVar[float] = 20.0
    min_speed: ClassVar[float] = 1.0
    max_delta: ClassVar[float] = 0.25


@dataclass(frozen=True)
class vicsek_const:
    ang_noi_par: ClassVar[float] = np.deg2rad(10)
    action_range: ClassVar[float] = 20.0


@dataclass(frozen=True)
class couzin_const:
    ang_noi_par: ClassVar[float] = np.deg2rad(0.1)
    zoa: ClassVar[float] = 15
    zoo: ClassVar[float] = 10
    zor: ClassVar[float] = 1.0
    max_turn_angle: ClassVar[float] = np.deg2rad(1)


@dataclass(frozen=True)
class predator_const:
    max_speed: ClassVar[float] = 25.0  # fixed
    min_speed: ClassVar[float] = 3.0  # fixed
    max_delta: ClassVar[float] = 1.5  # fixed
    att_par: ClassVar[float] = 2.0  # fixed
    sep_par: ClassVar[float] = 5
    dist_par: ClassVar[float] = 60.0


@dataclass
class obstacles_const:
    obstacle_type: ClassVar[str] = "wall"  # "wall", "column", "custom"
    obstacle_dim: ClassVar[float] = glob_const.n_boids//10  # fixed
    match obstacle_type:
        case "wall":
            n_cols: ClassVar[int] = obstacle_dim*2
            n_per_col: ClassVar[int] = obstacle_dim*2
            x_vals = np.linspace(-obstacle_dim, obstacle_dim, n_cols)
            z_vals = np.linspace(-obstacle_dim, obstacle_dim, n_per_col)
            xx, zz = np.meshgrid(x_vals, z_vals)
            positions: ClassVar[np.ndarray] = np.column_stack((
                xx.flatten(),
                np.full(xx.size, 400.0),
                zz.flatten()
            ))

        case "column":
            n_cols: ClassVar[int] = 1
            n_per_col: ClassVar[int] = obstacle_dim*3
            x_vals = np.linspace(0, 0, n_cols)
            z_vals = np.linspace(-obstacle_dim, obstacle_dim, n_per_col)
            xx, zz = np.meshgrid(x_vals, z_vals)
            positions: ClassVar[np.ndarray] = np.column_stack((
                xx.flatten(),
                np.full(xx.size, 200.0),
                zz.flatten()
            ))

        case "custom":
            n_cols: ClassVar[int] = 1
            n_per_col: ClassVar[int] = 30
            y_vals = np.linspace(-50.0, 50.0, n_cols)
            z_vals = np.linspace(-30.0, 30.0, n_per_col)
            yy, zz = np.meshgrid(y_vals, z_vals)
            positions: ClassVar[np.ndarray] = np.column_stack((
                np.full(yy.size, 300.0),
                yy.flatten(),
                zz.flatten()
            ))

    action_range: ClassVar[float] = 35
    rep_par: ClassVar[float] = 0.0045

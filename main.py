import config as cfg
import simulation as sml
import animation as anm
import auxiliary_functions as aux
import numpy as np
import flock_statistics as sts

flock = sml.FlockState()
pos_history = np.zeros((cfg.glob_const.time_steps, cfg.glob_const.n_boids, 3))
vel_history = np.zeros((cfg.glob_const.time_steps, cfg.glob_const.n_boids, 3))
predator = sml.Predator()
pred_pos_history = np.zeros((cfg.glob_const.time_steps, 1, 3))
pred_vel_history = np.zeros((cfg.glob_const.time_steps, 1, 3))


# Main cycle
for t in range(cfg.glob_const.time_steps):

    pos_history[t] = flock.pos.copy()
    vel_history[t] = flock.vel.copy()

    pred_pos_history[t] = predator.pos.copy()
    pred_vel_history[t] = predator.vel.copy()

    sml.update_flock(flock, predator, cfg.commands.method)

print(
    f"The flock dataset has been created: {pos_history.shape}")
if cfg.commands.predator_bool == True:
    print(f"The predator dataset has been created: {pred_pos_history.shape}")

# aux.verify_all_vel_are_constant(vel_history)

if cfg.commands.make_csv_bool:
    sts.make_csv(pos_history, vel_history)

if cfg.commands.gif_making_bool:
    anm.make_gif(pos_history, pred_pos_history)

if cfg.commands.plot_correlation_function:
    aux.plot_correlation_function()

if cfg.commands.compute_polarization:
    print("Computing polarization over time...")

    aux.plot_polarization_over_time()
import config as cfg
import simulation as sml
import animation as anm
import numpy as np

flock = sml.FlockState()
predator = sml.Predator()
pos_history = np.zeros((cfg.glob_const.time_steps, cfg.glob_const.n_boids, 3))
vel_history = np.zeros((cfg.glob_const.time_steps, cfg.glob_const.n_boids, 3))
pred_pos_history = np.zeros((cfg.glob_const.time_steps, 1, 3))
pred_vel_history = np.zeros((cfg.glob_const.time_steps, 1, 3))

# Main cycle
for t in range(cfg.glob_const.time_steps):

    pos_history[t] = flock.pos.copy()
    vel_history[t] = flock.vel.copy()

    pred_pos_history[t] = predator.pos.copy()
    pred_vel_history[t] = predator.vel.copy()

    sml.update_flock(flock, predator, cfg.glob_const.method)

print(
    f"The datasets have been created: {pos_history.shape}, {pred_pos_history.shape}")

anm.make_gif(pos_history, pred_pos_history)

import dataclasses
import config as cfg
import simulation as sml
import animation as anm
import numpy as np
import pandas as pd
import flock_statistics as sts
import matplotlib.pyplot as plt

flock = sml.FlockState()
predator = sml.Predator()
pos_history = np.zeros((cfg.glob_const.time_steps, cfg.glob_const.n_boids, 3))
vel_history = np.zeros((cfg.glob_const.time_steps, cfg.glob_const.n_boids, 3))
pred_pos_history = np.zeros((cfg.glob_const.time_steps, 1, 3))
pred_vel_history = np.zeros((cfg.glob_const.time_steps, 1, 3))


def get_class_vars(cls):
    # Iterates over the annotations to get the actual values from the class
    return {field: getattr(cls, field) for field in cls.__annotations__}


to_exclude = ['moving_camera_bool', 'gif_making_bool', 'artistic_rendition_bool', 'make_csv_bool',
              'plot_correlation_function', 'fov_angle', 'cos_fov', 'boids_in_vel_std', 'boids_in_pos_std', 'noi_par', 'method', 'predator_bool', 'obstacle_bool']
# Merge both classes
if cfg.commands.method == 'reynolds':
    all_params = {**get_class_vars(cfg.glob_const), **get_class_vars(
        cfg.commands), **get_class_vars(cfg.reynolds_const)}
elif cfg.commands.method == 'couzin':
    all_params = {**get_class_vars(cfg.glob_const), **get_class_vars(
        cfg.commands), **get_class_vars(cfg.couzin_const)}

# Create the legend string (filtering out booleans if you want it cleaner)
params_text = "\n".join(
    [f"{k}: {v}" for k, v in all_params.items() if k not in to_exclude])

# Main cycle
for t in range(cfg.glob_const.time_steps):

    pos_history[t] = flock.pos.copy()
    vel_history[t] = flock.vel.copy()

    pred_pos_history[t] = predator.pos.copy()
    pred_vel_history[t] = predator.vel.copy()

    sml.update_flock(flock, predator, cfg.commands.method)

print(
    f"The datasets have been created: {pos_history.shape}, {pred_pos_history.shape}")

# # Calculate the norm for each boid at each timestep
# # vel_history shape is (time_steps, n_boids, 3)
# norms = np.linalg.norm(vel_history, axis=2)

# # Check if all norms are approximately equal to the first one
# all_equal = np.allclose(norms, norms[0, 0])

# print(f"Are all velocity norms constant? {all_equal}")
# print(f"Min norm: {norms.min():.4f}, Max norm: {norms.max():.4f}, Average norm: {norms.mean():.4f}")

if cfg.commands.make_csv_bool:
    sts.make_csv(pos_history, vel_history)

if cfg.commands.gif_making_bool:
    anm.make_gif(pos_history, pred_pos_history)

if cfg.commands.make_csv_bool:
    sts.make_csv(pos_history, vel_history)

if cfg.commands.plot_correlation_function:
    df_original = pd.read_csv("flock_history.csv")

    # Initialize a list to store the C(r) Series for each step
    all_correlations = []

    cropped_steps = np.arange(
        max((df_original['step'].unique())+1)/4, max(df_original['step'].unique()))

    all_results_list = []

    print(f"Processing {len(cropped_steps)} steps")

    # Run the loop using on timesteps
    for step in cropped_steps:

        res = sts.compute_spatial_correlation(df_original, step, n_bins=10)

        step_df = res.reset_index()
        step_df.columns = ['r', 'c_r']

        all_results_list.append(step_df)

    full_data = pd.concat(all_results_list, ignore_index=True)

    # Redo the binning and averaging on the global dataset

    master_bins = np.linspace(0, full_data['r'].max(), 50)
    bin_centers = master_bins[:-1] + np.diff(master_bins) / 2

    full_data['master_r_bin'] = pd.cut(
        full_data['r'], bins=master_bins, labels=bin_centers)

    # final average
    final_c_r = full_data.groupby('master_r_bin', observed=True)['c_r'].mean()

    # plotting
    plt.figure(figsize=(10, 6))
    plt.plot(final_c_r.index, final_c_r.values, marker='s',
             markersize=4, linestyle='-', color='teal', label=params_text)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Distance $r$')
    plt.ylabel('Averaged Correlation $C(r)$')
    if cfg.commands.obstacle_bool and cfg.commands.predator_bool == False:
        plt.title('Final Global Spatial Correlation, with ' + cfg.commands.method.capitalize() + ' Method and Obstacles')
    elif cfg.commands.predator_bool and cfg.commands.obstacle_bool == False:
        plt.title('Final Global Spatial Correlation, with ' + cfg.commands.method.capitalize() + ' Method and Predator')
    elif cfg.commands.predator_bool and cfg.commands.obstacle_bool:
        plt.title('Final Global Spatial Correlation, with ' + cfg.commands.method.capitalize() + ' Method, Predator and Obstacles')
    else:
        plt.title('Final Global Spatial Correlation, with ' + cfg.commands.method.capitalize() + ' Method')
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.show()

print("Computing polarization over time...")

try:
    df_original = pd.read_csv("flock_history.csv")

    all_steps = df_original['step'].unique()

    polarization_over_time = []

    for step in all_steps:
        pol = sts.compute_polarization(df_original, step)
        polarization_over_time.append(pol)

    plt.figure(figsize=(10, 5))
    plt.plot(all_steps, polarization_over_time, marker='',
             linestyle='-', color='darkorange', linewidth=2, label=params_text)

    plt.ylim(0, 1.05)
    plt.xlabel('Time Step')
    plt.ylabel('Polarization')
    if cfg.commands.obstacle_bool and cfg.commands.predator_bool == False:
        plt.title('Polarization over time, with ' + cfg.commands.method.capitalize() + ' Method and Obstacles')
    elif cfg.commands.predator_bool and cfg.commands.obstacle_bool == False:
        plt.title('Polarization over time, with ' + cfg.commands.method.capitalize() + ' Method and Predator')
    elif cfg.commands.predator_bool and cfg.commands.obstacle_bool:
        plt.title('Polarization over time, with ' + cfg.commands.method.capitalize() + ' Method, Predator and Obstacles')
    else:
        plt.title('Polarization over time, with ' + cfg.commands.method.capitalize() + ' Method')
    plt.grid(True, alpha=0.4)
    plt.legend(loc='upper right', fontsize=8)
    plt.show()

except FileNotFoundError:
    print("Error. No csv so no polarization")

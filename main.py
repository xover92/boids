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

# Main cycle
for t in range(cfg.glob_const.time_steps):

    pos_history[t] = flock.pos.copy()
    vel_history[t] = flock.vel.copy()

    pred_pos_history[t] = predator.pos.copy()
    pred_vel_history[t] = predator.vel.copy()

    sml.update_flock(flock, predator, cfg.commands.method)

print(
    f"The datasets have been created: {pos_history.shape}, {pred_pos_history.shape}")

if cfg.commands.gif_making_bool:
    anm.make_gif(pos_history, pred_pos_history)

if cfg.commands.make_csv_bool:
    sml.make_csv(pos_history, vel_history)
    
if cfg.commands.plot_correlation_function:
    df_original = pd.read_csv("flock_history.csv")
    
    #Initialize a list to store the C(r) Series for each step
    all_correlations = []

    cropped_steps = np.arange(max((df_original['step'].unique())+1)/10, max(df_original['step'].unique()))
    
    all_results_list = []
    
    print(f"Processing {len(cropped_steps)} steps")

    #Run the loop using on timesteps
    for step in cropped_steps:
       
        res = sts.compute_spatial_correlation(df_original, step, n_bins=50)
        
    
        step_df = res.reset_index()
        step_df.columns = ['r', 'c_r']
        
        all_results_list.append(step_df)

  
    full_data = pd.concat(all_results_list, ignore_index=True)

    # Redo the binning and averaging on the global dataset

    master_bins = np.linspace(0, full_data['r'].max(), 50)
    bin_centers = master_bins[:-1] + np.diff(master_bins) / 2


    full_data['master_r_bin'] = pd.cut(full_data['r'], bins=master_bins, labels=bin_centers)

    # final average
    final_c_r = full_data.groupby('master_r_bin', observed=True)['c_r'].mean()

    # plotting
    plt.figure(figsize=(10, 6))
    plt.plot(final_c_r.index, final_c_r.values, marker='s', markersize=4, linestyle='-', color='teal')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Distance $r$')
    plt.ylabel('Averaged Correlation $C(r)$')
    plt.title('Final Global Spatial Correlation (Time-Averaged)')
    plt.grid(True, alpha=0.3)
    plt.show()


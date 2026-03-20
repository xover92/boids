import config as cfg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import flock_statistics as sts


def get_class_vars(cls):
    # Iterates over the annotations to get the actual values from the class
    return {field: getattr(cls, field) for field in cls.__annotations__}


def names_in_legend():
    # Dictionnary
    name_mapping = {
        # Globals
        'n_boids': 'Boids number',
        'max_speed': 'Max speed',
        'fov_angle': 'Field of view',
        
        # Reynolds
        'coh_par': 'Cohesion par',
        'ali_par': 'Alignement par',
        'sep_par': 'Separation par',
        'max_delta': 'Max acceleration',
        'min_speed': 'Min speed',
        
        # Couzin 
        'zoa': 'ZOA',
        'zoo': 'ZOO',
        'zor': 'ZOR',
        'max_turn_angle': 'Max steering angle',
    
        # Vicsek & Couzin
        'action_range': 'Action range',
        'ang_noi_par': 'Angular noise',
    }
    angles_to_convert = ['max_turn_angle', 'ang_noi_par', 'fov_angle']

    if cfg.commands.method == 'reynolds':
        all_params = {**get_class_vars(cfg.glob_const), **get_class_vars(cfg.reynolds_const)}
    elif cfg.commands.method == 'couzin':
        all_params = {**get_class_vars(cfg.glob_const), **get_class_vars(cfg.couzin_const)}
    elif cfg.commands.method == 'vicsek':
        all_params = {**get_class_vars(cfg.glob_const), **get_class_vars(cfg.vicsek_const)}

    legend_lines = []

    # Iterating on dictionnary 
    for param_key, display_name in name_mapping.items():
        if param_key in all_params:
            v = all_params[param_key]
            
            if param_key in angles_to_convert:
                val_deg = np.degrees(v)
                val_str = f"{val_deg:.1f}°"
            elif isinstance(v, float):
                val_str = f"{v:.2f}"
            else:
                val_str = str(v)

            legend_lines.append(f"{display_name}: {val_str}")

    return "\n".join(legend_lines)


def verify_all_vel_are_constant(vel_history):
    # Calculate the norm for each boid at each timestep
    # vel_history shape is (time_steps, n_boids, 3)
    norms = np.linalg.norm(vel_history, axis=2)

    # Check if all norms are approximately equal to the first one
    all_equal = np.allclose(norms, norms[0, 0])

    print(f"Are all velocity norms constant? {all_equal}")
    print(
        f"Min norm: {norms.min():.4f}, Max norm: {norms.max():.4f}, Average norm: {norms.mean():.4f}")


legend_text = names_in_legend()


def plot_polarization_over_time():
    try:
        df_original = pd.read_csv("flock_history.csv")

        all_steps = df_original['step'].unique()

        polarization_over_time = []

        for step in all_steps:
            pol = sts.compute_polarization(df_original, step)
            polarization_over_time.append(pol)

        plt.figure(figsize=(10, 5))
        plt.plot(all_steps, polarization_over_time, marker='',
                 linestyle='-', color='darkorange', linewidth=2, label=legend_text)

        plt.ylim(0, 1.05)
        plt.xlabel('Time Step', fontsize=15)
        plt.ylabel('Polarization', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        # Set the title based on the presence of obstacles and predators
        if cfg.commands.obstacle_bool and cfg.commands.predator_bool == False:
            plt.title('Polarization over time, with ' +
                      cfg.commands.method.capitalize() + ' Method and Obstacles', fontsize=15)
        elif cfg.commands.predator_bool and cfg.commands.obstacle_bool == False:
            plt.title('Polarization over time, with ' +
                      cfg.commands.method.capitalize() + ' Method and Predator', fontsize=15)
        elif cfg.commands.predator_bool and cfg.commands.obstacle_bool:
            plt.title('Polarization over time, with ' +
                      cfg.commands.method.capitalize() + ' Method, Predator and Obstacles', fontsize=15)
        else:
            plt.title('Polarization over time, with ' +
                      cfg.commands.method.capitalize() + ' Method', fontsize=15)

        plt.grid(True, alpha=0.4)
        plt.legend(loc='lower left', fontsize=15)
        plt.show()

    except FileNotFoundError:
        print("Error. No csv so no polarization")


def plot_correlation_function():
    df_original = pd.read_csv("flock_history.csv")

    cropped_steps = np.arange(
        max((df_original['step'].unique())+1)//4, max(df_original['step'].unique()))

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
             markersize=4, linestyle='-', color='teal', label=legend_text)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Distance $r$', fontsize=15)
    plt.ylabel('Averaged Correlation $C(r)$', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # Set the title based on the presence of obstacles and predators
    if cfg.commands.obstacle_bool and cfg.commands.predator_bool == False:
        plt.title('Final Global Spatial Correlation, with ' +
                  cfg.commands.method.capitalize() + ' Method and Obstacles', fontsize=15)
    elif cfg.commands.predator_bool and cfg.commands.obstacle_bool == False:
        plt.title('Final Global Spatial Correlation, with ' +
                  cfg.commands.method.capitalize() + ' Method and Predator', fontsize=15)
    elif cfg.commands.predator_bool and cfg.commands.obstacle_bool:
        plt.title('Final Global Spatial Correlation, with ' +
                  cfg.commands.method.capitalize() + ' Method, Predator and Obstacles', fontsize=15)
    else:
        plt.title('Final Global Spatial Correlation, with ' +
                  cfg.commands.method.capitalize() + ' Method', fontsize=15)

    plt.legend(loc='lower left', fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.show()
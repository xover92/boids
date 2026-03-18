import pandas as pd
import numpy as np
import simulation as sml


def make_csv(pos_history, vel_history):

    n_steps, n_boids, _ = pos_history.shape

    pos_flat = pos_history.reshape(-1, 3)
    vel_flat = vel_history.reshape(-1, 3)

    time_indices = np.repeat(np.arange(n_steps), n_boids)
    boid_ids = np.tile(np.arange(n_boids), n_steps)

    df = pd.DataFrame({
        'step': time_indices,
        'boid_id': boid_ids,
        'pos_x': pos_flat[:, 0],
        'pos_y': pos_flat[:, 1],
        'pos_z': pos_flat[:, 2],
        'vel_x': vel_flat[:, 0],
        'vel_y': vel_flat[:, 1],
        'vel_z': vel_flat[:, 2]
    })

    df['u_x'] = df['vel_x'] - df.groupby('step')['vel_x'].transform('mean')
    df['u_y'] = df['vel_y'] - df.groupby('step')['vel_y'].transform('mean')
    df['u_z'] = df['vel_z'] - df.groupby('step')['vel_z'].transform('mean')

    df.to_csv("flock_history.csv", index=False)
    print("File flock_history.csv successfully created")


def compute_spatial_correlation(df_original, step, n_bins=300):

    df = df_original[df_original['step'] == step]
    pos = df[['pos_x', 'pos_y', 'pos_z']].values
    u_raw = df[['u_x', 'u_y', 'u_z']].values
    n = len(df)
    u_vers = sml.versor(u_raw)

    dist_vects = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(dist_vects, axis=-1)
    uiuj_matrix = np.inner(u_vers, u_vers)

    i, j = np.triu_indices(n, k=1)
    r_list = dist_matrix[i, j]
    uiuj_list = uiuj_matrix[i, j]

    df_rough = pd.DataFrame({
        'r': r_list,
        'u_iu_j': uiuj_list
    })

    bins = np.linspace(0, df_rough['r'].max(), n_bins)
    df_rough['r_bin'] = pd.cut(
        df_rough['r'], bins=bins, labels=bins[:-1] + np.diff(bins)/2)

    c_r = (df_rough.groupby('r_bin', observed=True)['u_iu_j'].mean())

    return c_r


def compute_polarization(df_original, step):
    df = df_original[df_original['step'] == step]
    vel = df[['vel_x', 'vel_y', 'vel_z']].values

    vel_vers = sml.versor(vel)

    pol = np.linalg.norm(np.mean(vel_vers, axis=0))

    return pol
    
    
    
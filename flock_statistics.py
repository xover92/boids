import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    # Filter the dataframe for the specific time step
    df = df_original[df_original['step'] == step]
    
    # Extract coordinates and velocity fluctuations as NumPy arrays
    pos = df[['pos_x', 'pos_y', 'pos_z']].values
    u = df[['u_x', 'u_y', 'u_z']].values
    n=len(df)
    
    r_list=[]
    uiuj_list=[]
    
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2 + (pos[i,2]-pos[j,2])**2)
            uiuj= u[i,0]*u[j,0] + u[i,1]*u[j,1] + u[i,2]*u[j,2]
    
            r_list.append(dist)
            uiuj_list.append(uiuj)
    
    
    df_rough = pd.DataFrame({
        'r':r_list, 
        'u_iu_j':uiuj_list
    })

    bins = np.linspace(0, df_rough['r'].max(), n_bins)
    df_rough['r_bin'] = pd.cut(df_rough['r'], bins=bins, labels=bins[:-1] + np.diff(bins)/2)

    c_r = df_rough.groupby('r_bin', observed=True)['u_iu_j'].mean()

    c_0 = np.mean(np.sum(u**2, axis=1))
    
    c_r = c_r / c_0
    
    return c_r


def compute_polarization(df_original, step):
    df=df_original[df_original['step'] == step]
    v = df[['vel_x', 'vel_y', 'vel_z']].values
    n=len(df)
    
    total_v_x=df[['vel_x']].sum()
    total_v_y=df[['vel_y']].sum()
    total_v_z=df[['vel_z']].sum()
    
    polarization = np.sqrt(total_v_x**2+total_v_y**2+total_v_z**2)
    return polarization
    
    
    
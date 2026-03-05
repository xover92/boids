import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_spatial_correlation(df_original, step, n_bins=50):
    # Filter the dataframe for the specific time step
    df = df_original[df_original['step'] == step]
    
    # Extract coordinates and velocity fluctuations as NumPy arrays
    pos = df[['pos_x', 'pos_y', 'pos_z']].values
    u = df[['u_x', 'u_y', 'u_z']].values
    c0 = 1
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

    return c_r
import config as cfg
import simulation as sml
import numpy as np
from dataclasses import dataclass
from typing import ClassVar
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors


def make_gif(pos_history, pred_pos_history):
        if cfg.glob_const.artistic_rendition_bool==False:
            # Creating the animation
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Setting field of view limits
            # min_val = min(pos_history.min(), pred_pos_history.min())
            # max_val = max(pos_history.max(), pred_pos_history.max())
            # padding = (max_val - min_val) * 0.05
            # lim_inf = min_val - padding
            # lim_sup = max_val + padding
            # ax.set_xlim(lim_inf, lim_sup)
            # ax.set_ylim(lim_inf, lim_sup)
            # ax.set_zlim(lim_inf, lim_sup)
            ax.set_title("3D flock simulation", fontsize=14)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            
        
        if cfg.glob_const.artistic_rendition_bool==True:
             fig = plt.figure(figsize=(12, 9))
    
            # MAKING THE GRADIENT
             ax_bg = fig.add_axes([0, 0, 1, 1]) 
             gradient = np.linspace(0, 1, 256).reshape(-1, 1)

             cmap_colors = ["#1a0b15", "#d34727", "#ff7e5f"]
             sunset_cmap = mcolors.LinearSegmentedColormap.from_list("vespero", cmap_colors)
             
             ax_bg.imshow(gradient, aspect='auto', cmap=sunset_cmap, origin='upper')
             ax_bg.set_axis_off() # Nascondiamo l'asse del gradiente
 
             # transparent grid
             ax = fig.add_subplot(111, projection='3d')
             ax.set_facecolor('none')      
             ax.patch.set_alpha(0.0)      
             fig.patch.set_alpha(0.0)      
             
             ax.set_axis_off()           

            # CLOUDS
             def draw_clouds(n_clusters=50):
                 for _ in range(n_clusters):
                     cx, cy, cz = (np.random.rand(3) - 0.5) * 100
                     n_fluff = 500
    
                     fluff_x = cx + np.random.normal(0, 100, n_fluff)
                     fluff_y = cy + np.random.normal(0, 20, n_fluff)
                     fluff_z = cz + np.random.normal(0, 15, n_fluff)
                    
                     ax.scatter(fluff_x, fluff_y, fluff_z, 
                             c='white', marker='o', 
                             s=np.random.randint(300, 1000), 
                             alpha=0.02,
                             edgecolors='none', depthshade=False)
 
             draw_clouds(20)


    # Drawing the boids
        flock_scatter = ax.scatter(pos_history[0, :, 0],
                            pos_history[0, :, 1],
                            pos_history[0, :, 2],
                            c='black', marker='^', s=20, alpha=0.7,)


            # Drawing the predator
        if cfg.glob_const.predator_bool==True:
                pred_scatter = ax.scatter(pred_pos_history[0, :, 0],
                                    pred_pos_history[0, :, 1],
                                    pred_pos_history[0, :, 2],
                                    c='red', marker='^', s=60, alpha=1.0,)


            # Drawing obstacles
        if cfg.glob_const.obstacle_bool==True:
                ax.scatter(cfg.obstacles_const.positions[:, 0],
                        cfg.obstacles_const.positions[:, 1],
                        cfg.obstacles_const.positions[:, 2],
                        c='red', marker='o', s=200, label="Obstacles") 


        # # Moving the boids
        if cfg.glob_const.moving_camera_bool==False:
            def animate(frame):

                flock_current_pos = pos_history[frame]
                flock_scatter._offsets3d = (
                flock_current_pos[:, 0], flock_current_pos[:, 1], flock_current_pos[:, 2])
                    
                pred_current_pos = pred_pos_history[frame]
                pred_scatter._offsets3d = (pred_current_pos[:, 0], pred_current_pos[:, 1], pred_current_pos[:, 2])
                    
                return flock_scatter, pred_scatter



            # Moving the boids with dynamic camera
        if cfg.glob_const.moving_camera_bool==True:
            def animate(frame):

                flock_current_pos = pos_history[frame]
                flock_scatter._offsets3d = (
                    flock_current_pos[:, 0], flock_current_pos[:, 1], flock_current_pos[:, 2])
                    
                pred_current_pos = pred_pos_history[frame]
                pred_scatter._offsets3d = (pred_current_pos[:, 0], pred_current_pos[:, 1], pred_current_pos[:, 2])
                    
                centroid = flock_current_pos.mean(axis=0)
                    
                window = 40.0 
                    
                ax.set_xlim(centroid[0] - window, centroid[0] + window)
                ax.set_ylim(centroid[1] - window, centroid[1] + window)
                ax.set_zlim(centroid[2] - window, centroid[2] + window)
                    
                return flock_scatter, pred_scatter


            # Creating the gif
        ani = animation.FuncAnimation(
            fig, animate, frames=cfg.glob_const.time_steps, interval=100, blit=False)

        # Saving the gif
        print("The gif is loading")
        ani.save("animation_boids.gif", writer='pillow', fps=30)
        print("The gif is ready")





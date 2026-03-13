import config as cfg
import numpy as np
import pandas as pd


# Inizializing boids' initial positions and velocities
class FlockState:
    def __init__(self):
        self.pos = np.random.normal(
            loc=0, scale=cfg.glob_const.boids_in_pos_std, size=(cfg.glob_const.n_boids, 3))
        self.vel = np.random.normal(
            loc=cfg.glob_const.min_speed, scale=cfg.glob_const.boids_in_vel_std, size=(cfg.glob_const.n_boids, 3))
        if cfg.glob_const.method == "couzin":
            self.vel = versor(self.vel)*cfg.glob_const.min_speed


# Inizializing predator's initial position e velocity
class Predator:
    def __init__(self):
        self.pos = np.random.uniform(low=30, high=50,  size=(1, 3))
        self.vel = np.random.uniform(
            low=cfg.predator_const.min_speed, high=cfg.predator_const.max_speed, size=(1, 3))


# Compute distance vectors matrix (n, n, 3), distance norms matrix (n, n)
# and fov matrix (n, n) between boids
def compute_distances_and_fov(pos, vel):
    dist_vects = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_norms = np.linalg.norm(dist_vects, axis=-1)

    dot_prods = (vel[:, np.newaxis, :] * (-dist_vects)).sum(axis=-1)
    vel_norms = np.linalg.norm(vel, axis=-1)[:, np.newaxis]
    cos_angles = dot_prods / np.maximum(vel_norms * dist_norms, 1e-9)

    return dist_vects, dist_norms, cos_angles


# Checking max/min speed and speed variations
def apply_kinematic_limits(vel, vel_delta, max_delta, min_speed, max_speed):

    delta_norm = np.linalg.norm(vel_delta, axis=1, keepdims=True)
    vel_delta_limited = np.where(
        delta_norm > max_delta,
        (vel_delta / np.maximum(delta_norm, 1e-9)) * max_delta,
        vel_delta)
    new_vel = vel + vel_delta_limited

    speed = np.linalg.norm(new_vel, axis=1, keepdims=True)
    new_vel = np.where(
        speed > max_speed,
        (new_vel / speed) * cfg.glob_const.max_speed,
        new_vel)

    new_vel = np.where(
        speed < cfg.glob_const.min_speed,
        (new_vel / np.maximum(speed, 1e-9)) * min_speed,
        new_vel)

    return new_vel


# General function to compute versors
def versor(vector):
    safe_norm = np.maximum(np.linalg.norm(vector, keepdims=True, axis=1), 1e-9)
    return vector/safe_norm


# Clamping function
def clamp(vel_prov, vel):
    norm_prov = np.linalg.norm(vel_prov, axis=1, keepdims=True)
    capacity = np.maximum(cfg.glob_const.max_delta - norm_prov, 0.0)
    norm = np.linalg.norm(vel, axis=1, keepdims=True)
    vel_prov += np.where(
        norm < capacity,
        vel,
        versor(vel) * capacity
    )
    return vel_prov


# Reynolds model
def compute_reynolds(pos, vel, dist_vects, dist_norms, cos_angles, predator_state):

    mask = (dist_norms < cfg.glob_const.action_range) & (
        dist_norms > 0) & (cos_angles > cfg.glob_const.cos_fov)
    mask_3d = mask[:, :, np.newaxis]
    n_neighbors = mask.sum(axis=1)
    has_neighbors = n_neighbors > 0

    # Inizializing velocities vectors to zero
    coh_vel = np.zeros_like(vel)
    ali_vel = np.zeros_like(vel)
    sep_vel = np.zeros_like(vel)
    vel_prov = np.zeros_like(vel)

    # Cohesion
    sum_pos = (pos[np.newaxis, :, :] * mask_3d).sum(axis=1)
    centroid = sum_pos[has_neighbors] / n_neighbors[has_neighbors, np.newaxis]
    neighb_dist = centroid - pos[has_neighbors]
    neighb_dist_norm = np.linalg.norm((neighb_dist), keepdims=True, axis=1)
    coh_vel[has_neighbors] = (neighb_dist) * cfg.reynolds_const.coh_par / (neighb_dist_norm**3)

    # Alignement
    sum_vel = (vel[np.newaxis, :, :] * mask_3d).sum(axis=1)
    mean_vel = sum_vel[has_neighbors] / n_neighbors[has_neighbors, np.newaxis]
    ali_vel[has_neighbors] = (mean_vel - vel[has_neighbors]) * cfg.reynolds_const.ali_par

    # Separation
    safe_distance_sq = np.where(dist_norms == 0, 1.0, dist_norms**3)
    repulsion = dist_vects / safe_distance_sq[:, :, np.newaxis]
    sep_vel = (repulsion * mask_3d).sum(axis=1) * cfg.reynolds_const.sep_par

    # Computing clamping for each interaction 
    if cfg.commands.obstacle_bool == True:
        vel_prov = clamp(vel_prov, compute_obstacle_avoidance(pos))

    if cfg.commands.predator_bool == True:
        vel_prov = clamp(vel_prov, compute_predator_avoidance(
            pos, predator_state.pos))

    vel_prov = clamp(vel_prov, sep_vel)

    vel_prov = clamp(vel_prov, ali_vel)

    vel_prov = clamp(vel_prov, coh_vel)

    vel_delta = vel_prov

    return vel_delta


# Couzin model
def compute_couzin(pos, vel, dist_vects, dist_norms, cos_angles):

    fov_mask = cos_angles > cfg.glob_const.cos_fov
    mask_rep = (dist_norms < cfg.couzin_const.zor) & (dist_norms > 0) & fov_mask
    mask_ali = (dist_norms >= cfg.couzin_const.zor) & (
        dist_norms < cfg.couzin_const.zoo) & fov_mask
    mask_coh = (dist_norms >= cfg.couzin_const.zoo) & (
        dist_norms <= cfg.couzin_const.zoa) & fov_mask
    mask_3d_rep = mask_rep[:, :, np.newaxis]
    mask_3d_ali = mask_ali[:, :, np.newaxis]
    mask_3d_coh = mask_coh[:, :, np.newaxis]

    n_neighbors_rep = mask_rep.sum(axis=1)
    n_neighbors_ali = mask_ali.sum(axis=1)
    n_neighbors_coh = mask_coh.sum(axis=1)

    has_neighbors_rep = n_neighbors_rep > 0
    has_neighbors_ali = n_neighbors_ali > 0
    has_neighbors_coh = n_neighbors_coh > 0

    coh_vel = np.zeros_like(vel)
    ali_vel = np.zeros_like(vel)
    sep_vel = np.zeros_like(vel)

    # Repulsion
    repulsion = versor(dist_vects)
    sep_vel[has_neighbors_rep] = (
        repulsion * mask_3d_rep).sum(axis=1)[has_neighbors_rep] * cfg.couzin_const.sep_par

    # Alignment
    align = versor(vel)
    ali_vel[has_neighbors_ali] =(
        align * mask_3d_ali).sum(axis=1)[has_neighbors_ali] * cfg.couzin_const.ali_par


    sum_vel = (vel[np.newaxis, :, :] * mask_3d_ali).sum(axis=1)
    mean_vel = sum_vel[has_neighbors_ali] / \
        n_neighbors_ali[has_neighbors_ali, np.newaxis]
    ali_vel[has_neighbors_ali] = (
        mean_vel - vel[has_neighbors_ali]) * cfg.couzin_const.ali_par

    # Cohesion
    safe_distance = np.maximum(dist_norms, 1e-9)
    cohesion = dist_vects / safe_distance[:, :, np.newaxis]
    coh_vel[has_neighbors_coh] = -(
        cohesion * mask_3d_coh).sum(axis=1)[has_neighbors_coh] * cfg.couzin_const.coh_par

    # sum_pos = (pos[np.newaxis, :, :] * mask_3d_coh).sum(axis=1)
    # centroid = sum_pos[has_neighbors_coh] / \
    #     n_neighbors_coh[has_neighbors_coh, np.newaxis]
    # coh_vel[has_neighbors_coh] = (
    #     centroid - pos[has_neighbors_coh]) * cfg.couzin_const.coh_par

   # Combine alignment and cohesion for boids that do not need to repel.
    # If a boid has both, average them. If it has only one, use it. If neither, it defaults to 0.
    ali_coh_combined = np.where(
        has_neighbors_ali[:, np.newaxis] & has_neighbors_coh[:, np.newaxis],
        (ali_vel + coh_vel) / 2.0,
        np.where(
            has_neighbors_ali[:, np.newaxis],
            ali_vel,
            coh_vel
        )
    )

    # Repulsion has absolute priority. If a boid needs to repel, ignore alignment/cohesion.
    vel_prov = np.where(
        has_neighbors_rep[:, np.newaxis],
        sep_vel,
        ali_coh_combined
    )

    # White noise
    noise = 0
    # noise = np.random.normal(
    #     loc=0.0, scale=cfg.couzin_const.noi_par, size=(cfg.glob_const.n_boids, 3))
    # vel_delta = vel_prov + noise

    target_dir = vel_prov + noise
    # dir_norm = np.linalg.norm(target_dir, axis=1, keepdims=True)
    target_vel = versor(target_dir) * \
        cfg.glob_const.max_speed
    vel_delta = target_vel - vel

    return vel_delta


# Avoiding obstacles
def compute_obstacle_avoidance(pos):

    obs_vel = np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - \
        cfg.obstacles_const.positions[np.newaxis, :, :]
    distance = np.linalg.norm(diff, axis=-1)
    mask = (distance < cfg.obstacles_const.action_range) & (distance > 0)
    mask_3d = mask[:, :, np.newaxis]

    safe_distance_sq = np.where(distance == 0, 1.0, distance**3)
    repulsion = diff / safe_distance_sq[:, :, np.newaxis]
    obs_vel = (repulsion * mask_3d).sum(axis=1) * cfg.obstacles_const.rep_par

    return obs_vel


# Avoiding predator
def compute_predator_avoidance(flock_pos, pred_pos):

    pred_avoid_vel = np.zeros_like(flock_pos)
    diff = flock_pos[:, np.newaxis, :] - pred_pos[np.newaxis, :, :]
    distance = np.linalg.norm(diff, axis=-1)
    mask = (distance < cfg.predator_const.dist_par) & (distance > 0)
    mask_3d = mask[:, :, np.newaxis]

    safe_distance_sq = np.where(distance == 0, 1.0, distance**3)
    repulsion = diff / safe_distance_sq[:, :, np.newaxis]
    pred_avoid_vel = (repulsion * mask_3d).sum(axis=1) * \
        cfg.predator_const.sep_par

    return pred_avoid_vel


# Predator dynamic
def predator_move(flock_pos, pred_pos, pred_vel):

    # Attraction to the flock's centroid
    sum_pos = (flock_pos[np.newaxis, :, :]).sum(axis=1)
    centroid = sum_pos / cfg.glob_const.n_boids
    pred_vel = (centroid - pred_pos) * cfg.predator_const.att_par

    return pred_vel


# Main function
def update_flock(flock_state: FlockState, predator_state: Predator, method: str):

    diff, distance, cos_angle = compute_distances_and_fov(
        flock_state.pos, flock_state.vel)

    # if not hasattr(update_flock, "counter"):
    #     update_flock.counter = 0

    match method.lower():
        case "reynolds":
            vel_delta = compute_reynolds(
                flock_state.pos, flock_state.vel, diff, distance, cos_angle, predator_state)
        case "couzin":
            vel_delta = compute_couzin(
                flock_state.pos, flock_state.vel, diff, distance, cos_angle)
        case _:
            raise ValueError(
                f"Method '{method}' is invalid. Choose between 'reynolds' or 'couzin'.")

    pred_avoid_vel = np.zeros_like(flock_state.pos)

    pred_vel_delta = np.zeros_like([[0, 0, 0]])

    if cfg.commands.predator_bool == True:
        pred_vel_delta = predator_move(
            flock_state.pos, predator_state.pos, predator_state.vel)

    final_vel_delta = vel_delta

    flock_state.vel += final_vel_delta
    # flock_state.vel = apply_kinematic_limits(
    #     flock_state.vel, final_vel_delta, cfg.glob_const.max_delta, cfg.glob_const.min_speed, cfg.glob_const.max_speed)

    # random white noise
    noise = np.random.normal(
        scale=cfg.glob_const.boids_in_vel_std/5, loc=0, size=flock_state.vel.shape)
    norm_flock_vel = np.linalg.norm(flock_state.vel, keepdims=True, axis=1)
    flock_state.vel = versor(noise+flock_state.vel)*norm_flock_vel

    # if update_flock.counter %50 == 0:
    #         # 1. Create a random noise vector
    #     common_noise = np.random.normal(scale=5, size=3)

    #     # 2. Select random indices for half the flock (e.g., 100 out of 200)
    #     num_boids = flock_state.vel.shape[0]
    #     indices = np.random.choice(num_boids, size=num_boids // 2, replace=False)

    #     # 3. Apply the noise only to those specific boids
    #     flock_state.vel[indices] += common_noise

    if cfg.commands.predator_bool == True:
        predator_state.vel = apply_kinematic_limits(
            predator_state.vel, pred_vel_delta, cfg.predator_const.max_delta, cfg.predator_const.min_speed, cfg.predator_const.max_speed)

    flock_state.pos += flock_state.vel

    predator_state.pos += predator_state.vel

    # update_flock.counter+=1


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

import config as cfg
import numpy as np


# General function to compute versors
def versor(vector):
    safe_norm = np.maximum(np.linalg.norm(
        vector, keepdims=True, axis=1), 1e-10)
    return vector/safe_norm


# Inizializing boids' initial positions and velocities
class FlockState:
    def __init__(self):
        self.pos = np.random.normal(
            loc=0, scale=cfg.glob_const.boids_in_pos_std, size=(cfg.glob_const.n_boids, 3))
        self.vel = np.random.normal(
            loc=cfg.reynolds_const.min_speed, scale=cfg.glob_const.boids_in_vel_std, size=(cfg.glob_const.n_boids, 3))
        self.vel = versor(self.vel)*cfg.reynolds_const.min_speed
        if cfg.commands.method=="couzin":
            self.vel = versor(self.vel)*cfg.couzin_const.speed


# Inizializing predator's initial position e velocity
class Predator:
    def __init__(self):
        self.pos = np.random.uniform(
            low=cfg.predator_const.low_spawn, high=cfg.predator_const.high_spawn,  size=(1, 3))
        self.vel = np.random.uniform(
            low=cfg.predator_const.min_speed, high=cfg.predator_const.max_speed, size=(1, 3))
        self.vel = versor(self.vel)*cfg.predator_const.min_speed


# Compute distance vectors matrix (n, n, 3), distance norms matrix (n, n)
# and fov matrix (n, n) between boids
def compute_distances_and_fov(pos, vel):
    dist_vects = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_norms = np.linalg.norm(dist_vects, axis=-1)

    dot_prods = (vel[:, np.newaxis, :] * (-dist_vects)).sum(axis=-1)
    vel_norms = np.linalg.norm(vel, axis=-1)[:, np.newaxis]
    cos_angles = dot_prods / np.maximum(vel_norms * dist_norms, 1e-9)

    return dist_vects, dist_norms, cos_angles


# Clamping function
def clamp(vel_prov, vel):
    norm_prov = np.linalg.norm(vel_prov, axis=1, keepdims=True)
    capacity = np.maximum(cfg.reynolds_const.max_delta - norm_prov, 0.0)
    norm = np.linalg.norm(vel, axis=1, keepdims=True)
    vel_prov += np.where(
        norm < capacity,
        vel,
        versor(vel) * capacity
    )
    return vel_prov


def limit_turn_angle(vel, target_vel, max_angle_rad):

    vel_vers = versor(vel)
    target_vel_vers = versor(target_vel)

    # Computing angle between previous vel and target vel
    dot_prod = (vel_vers * target_vel_vers).sum(axis=1, keepdims=True)
    dot_prod = np.clip(dot_prod, -1.0, 1.0)
    delta_angle = np.arccos(dot_prod)
    mask_turn = delta_angle > max_angle_rad

    # Computing versor on turning plane and clamping
    perp_vel = target_vel_vers - (dot_prod * vel_vers)
    perp_vel_vers = versor(perp_vel)
    clamped_vel = np.cos(max_angle_rad) * vel_vers + \
        np.sin(max_angle_rad) * perp_vel_vers
    final_vel = np.where(mask_turn, clamped_vel, target_vel_vers)

    return final_vel


# Avoiding obstacles
def compute_obstacle_avoidance(pos):

    obs_dist_vects = pos[:, np.newaxis, :] - \
        cfg.obstacles_const.positions[np.newaxis, :, :]
    obs_dist_norms = np.linalg.norm(obs_dist_vects, axis=-1)
    mask = (obs_dist_norms < cfg.obstacles_const.action_range) & (
        obs_dist_norms > 0)
    mask_3d = mask[:, :, np.newaxis]
    safe_distance_sq = np.where(obs_dist_norms == 0, 1.0, obs_dist_norms**3)
    repulsion = obs_dist_vects / safe_distance_sq[:, :, np.newaxis]
    obs_avoid_vel = (repulsion * mask_3d).sum(axis=1) * \
        cfg.obstacles_const.rep_par

    return obs_avoid_vel


# Avoiding predator
def compute_predator_avoidance(flock_pos, pred_pos):

    pred_dist_vects = flock_pos[:, np.newaxis, :] - pred_pos[np.newaxis, :, :]
    pred_dist_norms = np.linalg.norm(pred_dist_vects, axis=-1)
    mask = (pred_dist_norms < cfg.predator_const.dist_par) & (
        pred_dist_norms > 0)
    mask_3d = mask[:, :, np.newaxis]
    safe_distance_sq = np.where(pred_dist_norms == 0, 1.0, pred_dist_norms**3)
    repulsion = pred_dist_vects / safe_distance_sq[:, :, np.newaxis]
    pred_avoid_vel = (repulsion * mask_3d).sum(axis=1) * \
        cfg.predator_const.sep_par

    return pred_avoid_vel


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
    prov_vel = np.zeros_like(vel)

    # Cohesion
    sum_pos = (pos[np.newaxis, :, :] * mask_3d).sum(axis=1)
    centroid = sum_pos[has_neighbors] / n_neighbors[has_neighbors, np.newaxis]
    neighb_dist = centroid - pos[has_neighbors]
    neighb_dist_norm = np.linalg.norm((neighb_dist), keepdims=True, axis=1)
    coh_vel[has_neighbors] = (neighb_dist) * \
        cfg.reynolds_const.coh_par / (neighb_dist_norm**3)

    # Alignement
    sum_vel = (vel[np.newaxis, :, :] * mask_3d).sum(axis=1)
    mean_vel = sum_vel[has_neighbors] / n_neighbors[has_neighbors, np.newaxis]
    ali_vel[has_neighbors] = (
        mean_vel - vel[has_neighbors]) * cfg.reynolds_const.ali_par

    # Separation
    safe_distance_sq = np.where(dist_norms == 0, 1.0, dist_norms**3)
    repulsion = dist_vects / safe_distance_sq[:, :, np.newaxis]
    sep_vel = (repulsion * mask_3d).sum(axis=1) * cfg.reynolds_const.sep_par

    # Computing clamping for each interaction
    if cfg.commands.obstacle_bool == True:
        prov_vel = clamp(prov_vel, compute_obstacle_avoidance(pos))

    if cfg.commands.predator_bool == True:
        prov_vel = clamp(prov_vel, compute_predator_avoidance(
            pos, predator_state.pos))

    prov_vel = clamp(prov_vel, sep_vel)

    prov_vel = clamp(prov_vel, ali_vel)

    prov_vel = clamp(prov_vel, coh_vel)

    return prov_vel


# Couzin model
def compute_couzin(pos, vel, dist_vects, dist_norms, cos_angles, predator_state):

    fov_mask = cos_angles > cfg.glob_const.cos_fov
    mask_rep = (dist_norms < cfg.couzin_const.zor) & (
        dist_norms > 0) & fov_mask
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
    avoid_obs_vel = np.zeros_like(vel)
    avoid_pred_vel = np.zeros_like(vel)

    # Repulsion
    repulsion = versor(dist_vects)
    sep_vel[has_neighbors_rep] = (
        repulsion * mask_3d_rep).sum(axis=1)[has_neighbors_rep]

    # Alignment
    align = versor(vel)
    ali_vel[has_neighbors_ali] = (
        align * mask_3d_ali).sum(axis=1)[has_neighbors_ali]

    # Cohesion
    cohesion = versor(dist_vects)
    coh_vel[has_neighbors_coh] = -(
        cohesion * mask_3d_coh).sum(axis=1)[has_neighbors_coh]

    if cfg.commands.obstacle_bool == True:
        avoid_obs_vel = compute_obstacle_avoidance(pos)

    if cfg.commands.predator_bool == True:
        avoid_pred_vel = compute_predator_avoidance(
            pos, predator_state.pos)

    gen_avoid_vel = avoid_obs_vel + avoid_pred_vel

    # Couzin zone model
    target_vel = np.where(
        has_neighbors_rep[:, np.newaxis],
        versor(sep_vel + gen_avoid_vel),
        np.where(
            has_neighbors_ali[:,
                              np.newaxis] & has_neighbors_coh[:, np.newaxis],
            versor((ali_vel + coh_vel) / 2.0 + gen_avoid_vel),
            np.where(
                has_neighbors_ali[:, np.newaxis],
                versor(ali_vel + gen_avoid_vel),
                np.where(
                    has_neighbors_coh[:, np.newaxis],
                    versor(coh_vel + gen_avoid_vel),
                    versor(vel + gen_avoid_vel)
                )
            )
        )
    )

    # Usa la nuova funzione per trovare la direzione reale limitando l'angolo
    new_vel = limit_turn_angle(
        vel, target_vel, cfg.couzin_const.max_turn_angle) * cfg.couzin_const.speed
    prov_vel = new_vel - vel

    return prov_vel


# Predator dynamic
def predator_move(flock_pos, pred_pos):

    # Attraction to the flock's centroid
    sum_pos = (flock_pos[np.newaxis, :, :]).sum(axis=1)
    centroid = sum_pos / cfg.glob_const.n_boids
    prov_vel = (centroid - pred_pos) * cfg.predator_const.att_par
    prov_vel = np.where(
        np.linalg.norm(prov_vel, keepdims=True, axis=1) > cfg.predator_const.max_delta,
             versor(prov_vel) * cfg.predator_const.max_delta,
             prov_vel)

    return prov_vel


# Main function
def update_flock(flock_state: FlockState, predator_state: Predator, method: str):

    dist_vects, dist_norms, cos_angles = compute_distances_and_fov(
        flock_state.pos, flock_state.vel)

    match method.lower():
        case "reynolds":
            boids_prov_vel = compute_reynolds(
                flock_state.pos, flock_state.vel, dist_vects, dist_norms, cos_angles, predator_state)
        case "couzin":
            boids_prov_vel = compute_couzin(
                flock_state.pos, flock_state.vel, dist_vects, dist_norms, cos_angles, predator_state)
        case _:
            raise ValueError(
                f"Method '{method}' is invalid. Choose between 'reynolds' or 'couzin'.")

    flock_state.vel += boids_prov_vel

    # Boids' velocity limit
    boids_speed = np.linalg.norm(flock_state.vel, axis=1, keepdims=True)
    flock_state.vel = np.where(
        boids_speed > cfg.reynolds_const.max_speed,
        (flock_state.vel / boids_speed) * cfg.reynolds_const.max_speed,
        flock_state.vel
    )

    # Random white noise
    noise = np.random.normal(
        scale=cfg.glob_const.boids_in_vel_std/5, loc=0, size=flock_state.vel.shape)
    norm_flock_vel = np.linalg.norm(flock_state.vel, keepdims=True, axis=1)
    flock_state.vel = versor(noise+flock_state.vel)*norm_flock_vel

    if cfg.commands.predator_bool == True:
        pred_prov_vel = predator_move(
            flock_state.pos, predator_state.pos)
        predator_state.vel += pred_prov_vel
        # Predator's velocity limit
        pred_speed = np.linalg.norm(predator_state.vel, axis=1, keepdims=True)
        predator_state.vel = np.where(
            pred_speed > cfg.predator_const.max_speed,
            (predator_state.vel / pred_speed) * cfg.predator_const.max_speed,
            predator_state.vel
        )

    flock_state.pos += flock_state.vel

    predator_state.pos += predator_state.vel

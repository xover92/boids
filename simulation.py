import config as cfg
import numpy as np


# General function to compute versors
def versor(vector):
    safe_norm = np.maximum(np.linalg.norm(vector, keepdims=True, axis=1), 1e-9)
    return vector/safe_norm


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
        (new_vel / speed) * max_speed,
        new_vel)

    new_vel = np.where(
        speed < min_speed,
        (new_vel / np.maximum(speed, 1e-9)) * min_speed,
        new_vel)

    return new_vel


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


def limit_turn_angle(vel_old, vel_target, max_angle_rad):
    """
    Limita la sterzata vettoriale a un angolo massimo max_angle_rad.
    Entrambi gli array devono avere shape (N, 3).
    """
    # 1. Troviamo i versori (direzioni pure)
    v_old = versor(vel_old)
    v_targ = versor(vel_target)

    # 2. Calcoliamo il prodotto scalare per trovare l'angolo
    # np.clip serve ad evitare errori di approssimazione float fuori da [-1, 1]
    dot = (v_old * v_targ).sum(axis=1, keepdims=True)
    dot = np.clip(dot, -1.0, 1.0)
    angle = np.arccos(dot)

    # Maschera: chi sta cercando di virare troppo forte?
    mask_turn = angle > max_angle_rad

    # 3. Calcoliamo il vettore ortogonale (piano di virata)
    # Rimuoviamo da v_targ la componente parallela a v_old
    v_perp = v_targ - (dot * v_old)
    
    # Normalizziamo il vettore ortogonale in modo sicuro
    norm_perp = np.linalg.norm(v_perp, axis=1, keepdims=True)
    safe_norm_perp = np.maximum(norm_perp, 1e-9)
    v_perp_versor = v_perp / safe_norm_perp

    # 4. Calcoliamo il vettore ruotato esattamente del max_angle_rad
    v_clamped = np.cos(max_angle_rad) * v_old + np.sin(max_angle_rad) * v_perp_versor

    # 5. Applichiamo la virata limitata solo a chi supera l'angolo
    final_dir = np.where(mask_turn, v_clamped, v_targ)

    return final_dir


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
        vel_prov = clamp(vel_prov, compute_obstacle_avoidance(pos))

    if cfg.commands.predator_bool == True:
        vel_prov = clamp(vel_prov, compute_predator_avoidance(
            pos, predator_state.pos))

    vel_prov = clamp(vel_prov, sep_vel)

    vel_prov = clamp(vel_prov, ali_vel)

    vel_prov = clamp(vel_prov, coh_vel)

    return vel_prov


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
    has_no_neighbours = (~has_neighbors_rep) & (~has_neighbors_ali) & (~has_neighbors_coh)

    coh_vel = np.zeros_like(vel)
    ali_vel = np.zeros_like(vel)
    sep_vel = np.zeros_like(vel)
    avoid_obs_vel = np.zeros_like(vel)
    avoid_pred_vel = np.zeros_like(vel)

    # Repulsion
    repulsion = versor(dist_vects)
    sep_vel[has_neighbors_rep] = (
        repulsion * mask_3d_rep).sum(axis=1)[has_neighbors_rep] * cfg.couzin_const.sep_par

    # Alignment
    align = versor(vel)
    ali_vel[has_neighbors_ali] = (
        align * mask_3d_ali).sum(axis=1)[has_neighbors_ali] * cfg.couzin_const.ali_par

    # Cohesion
    cohesion = versor(dist_vects)
    coh_vel[has_neighbors_coh] = -(
        cohesion * mask_3d_coh).sum(axis=1)[has_neighbors_coh] * cfg.couzin_const.coh_par

    if cfg.commands.obstacle_bool == True:
        avoid_obs_vel = compute_obstacle_avoidance(pos)

    if cfg.commands.predator_bool == True:
        avoid_pred_vel = compute_predator_avoidance(
            pos, predator_state.pos)
        
    gen_avoid_vel = avoid_obs_vel + avoid_pred_vel

# Couzin zone model: calcoliamo la DIREZIONE BERSAGLIO
    target_dir = np.where(
        has_neighbors_rep[:, np.newaxis],
        versor(sep_vel + gen_avoid_vel),
        np.where(
            has_neighbors_ali[:, np.newaxis] & has_neighbors_coh[:, np.newaxis],
            versor(ali_vel + coh_vel + gen_avoid_vel) / 2.0, 
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
    actual_dir = limit_turn_angle(vel, target_dir, cfg.glob_const.max_turn_angle)

    target_vel = actual_dir * cfg.glob_const.min_speed
    
    vel_delta = target_vel - vel

    return vel_delta


# Avoiding obstacles
def compute_obstacle_avoidance(pos):

    obs_avoid_vel = np.zeros_like(pos)
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

    pred_avoid_vel = np.zeros_like(flock_pos)
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


# Predator dynamic
def predator_move(flock_pos, pred_pos, pred_vel):

    # Attraction to the flock's centroid
    sum_pos = (flock_pos[np.newaxis, :, :]).sum(axis=1)
    centroid = sum_pos / cfg.glob_const.n_boids
    pred_vel = (centroid - pred_pos) * cfg.predator_const.att_par

    return pred_vel


# Main function
def update_flock(flock_state: FlockState, predator_state: Predator, method: str):

    dist_vects, dist_norms, cos_angles = compute_distances_and_fov(
        flock_state.pos, flock_state.vel)

    match method.lower():
        case "reynolds":
            vel_prov = compute_reynolds(
                flock_state.pos, flock_state.vel, dist_vects, dist_norms, cos_angles, predator_state)
        case "couzin":
            vel_prov = compute_couzin(
                flock_state.pos, flock_state.vel, dist_vects, dist_norms, cos_angles, predator_state)
        case _:
            raise ValueError(
                f"Method '{method}' is invalid. Choose between 'reynolds' or 'couzin'.")

    # Random white noise
    flock_state.vel += vel_prov

    # Velocity limit
    speed = np.linalg.norm(flock_state.vel, axis=1, keepdims=True)
    flock_state.vel = np.where(
        speed > cfg.glob_const.max_speed,
        (flock_state.vel / speed) * cfg.glob_const.max_speed,
        flock_state.vel
    )

    noise = np.random.normal(
        scale=cfg.glob_const.boids_in_vel_std/5, loc=0, size=flock_state.vel.shape)
    norm_flock_vel = np.linalg.norm(flock_state.vel, keepdims=True, axis=1)
    flock_state.vel = versor(noise+flock_state.vel)*norm_flock_vel

    if cfg.commands.predator_bool == True:
        pred_vel_delta = np.zeros_like([[0, 0, 0]])
        pred_vel_delta = predator_move(
            flock_state.pos, predator_state.pos, predator_state.vel)
        predator_state.vel = apply_kinematic_limits(
            predator_state.vel, pred_vel_delta, cfg.predator_const.max_delta, cfg.predator_const.min_speed, cfg.predator_const.max_speed)

    flock_state.pos += flock_state.vel

    predator_state.pos += predator_state.vel





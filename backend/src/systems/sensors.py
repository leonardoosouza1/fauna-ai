"""
Sistema Sensorial Multi-Espécie — Projeto Fauna v2.

Gera observações para cada agente com capacidades sensoriais alométricas:
  1. Propriocepção: estado interno + massa + sexo + cooldown reprodutivo
  2. Vizinhança: identifica ESPÉCIE, tamanho e saúde dos vizinhos
  3. Ambiente: recursos (C3/C4), cobertura vegetal, água e odores
  4. Percepção de Predadores: "Ecologia do Medo" via scent_map

Parâmetros (visual_range, fov) agora vêm dos arrays de espécie.
"""

import jax
import jax.numpy as jnp

from src.core.types import Array, DTYPE_COMPUTE
from src.systems.physics import compute_distances, find_nearest_neighbors


# ──────────────────────────────────────────────
# Configurações Padrão (Fallback)
# ──────────────────────────────────────────────
DEFAULT_NUM_NEIGHBORS = 5


def sense_proprioception(
    velocity: Array,
    energy: Array,
    age: Array,
    mass: Array,
    max_energy: Array,
    reproductive_cooldown: Array,
    sex: Array,
) -> Array:
    """
    Gera vetor de propriocepção estendido.

    Returns:
        proprio: (dim,)
          - vel_x, vel_y (2)
          - speed (1)
          - energy_pct (1)
          - age_pct (1)
          - mass_norm (1)
          - repro_ready (1)
          - is_male (1)
    """
    speed = jnp.linalg.norm(velocity)
    
    # Normalizações
    energy_pct = jnp.clip(energy / max_energy, 0.0, 1.0)
    age_pct = jnp.clip(age / 50000.0, 0.0, 1.0)
    mass_norm = jnp.log1p(mass) / 5.0  # Log-scale para massa (0.5kg a 50kg)
    
    repro_ready = (reproductive_cooldown <= 0).astype(DTYPE_COMPUTE)
    is_male = sex.astype(DTYPE_COMPUTE)

    return jnp.concatenate([
        velocity,                          # (2,)
        jnp.array([speed]),                # (1,)
        jnp.array([energy_pct]),           # (1,)
        jnp.array([age_pct]),              # (1,)
        jnp.array([mass_norm]),            # (1,)
        jnp.array([repro_ready]),          # (1,)
        jnp.array([is_male]),              # (1,)
    ]).astype(DTYPE_COMPUTE)


def sense_neighbors(
    agent_position: Array,
    all_positions: Array,
    all_species: Array,
    all_masses: Array,
    neighbor_indices: Array,
    neighbor_distances: Array,
    sensor_range: float,
    fov: float = jnp.pi * 2,
    velocity: Array | None = None,
) -> Array:
    """
    Percepção de vizinhos com identificação de espécie.

    Args:
        sensor_range: Alcance visual deste agente específico (alométrico)
        fov: Campo de visão (radianos)
    
    Returns:
        neighbor_obs: Para cada vizinho:
          - rel_pos_x, rel_pos_y (normalizado)
          - distance (normalizado)
          - species_id (one-hot ou embedding simplificado? Vamos usar ID / 10.0)
          - mass (log-normalizado)
          - is_predator (flag simples baseada em species_id >= 4)
    """
    k = neighbor_indices.shape[0]
    
    # Dados dos vizinhos
    neigh_pos = all_positions[neighbor_indices]
    neigh_species = all_species[neighbor_indices]
    neigh_mass = all_masses[neighbor_indices]
    
    # Posições relativas
    rel_pos = neigh_pos - agent_position
    
    # Checagem de FOV (se velocity fornecida para orientação)
    in_fov = jnp.ones((k, 1))
    if velocity is not None:
        agent_dir = jnp.arctan2(velocity[1], velocity[0])
        neigh_dir = jnp.arctan2(rel_pos[:, 1], rel_pos[:, 0])
        angle_diff = jnp.abs(jnp.angle(jnp.exp(1j * (neigh_dir - agent_dir))))
        in_fov = (angle_diff <= fov / 2.0).reshape(k, 1)

    # Normalizar distância
    dist_norm = jnp.clip(neighbor_distances / sensor_range, 0.0, 1.0).reshape(k, 1)
    
    # Mascarar fora do alcance e FOV
    visible = (neighbor_distances <= sensor_range).reshape(k, 1) & in_fov
    visible = visible.astype(DTYPE_COMPUTE)
    
    rel_pos_norm = (rel_pos / sensor_range) * visible
    dist_norm = dist_norm * visible
    
    # Features de espécie
    # Id normalizado (simples)
    sp_feat = (neigh_species / 10.0).reshape(k, 1) * visible
    
    # Massa log
    mass_feat = (jnp.log1p(neigh_mass) / 5.0).reshape(k, 1) * visible
    
    # Flag predador (hardcoded > 3 por enquanto, ou passado via args)
    is_pred = ((neigh_species >= 4).astype(DTYPE_COMPUTE)).reshape(k, 1) * visible
    
    # Concatenar features por vizinho: 2 + 1 + 1 + 1 + 1 = 6 features
    obs = jnp.concatenate([
        rel_pos_norm, dist_norm, sp_feat, mass_feat, is_pred
    ], axis=-1).flatten()
    
    return obs.astype(DTYPE_COMPUTE)


def sense_environment(
    position: Array,
    biomass_c4: Array,
    biomass_c3: Array,
    cover_density: Array,
    water_mask: Array,
    scent_map: Array,
    world_bounds: float = 500.0,
) -> Array:
    """
    Percepção ambiental local (vegetação, cobertura, água, cheiro).

    Amostra no local exato do agente (sem gradiente visual por enquanto, 
    para economizar inputs, mas poderia ter).

    Returns:
        env_obs: (6,)
          - local_c4
          - local_c3
          - local_cover
          - on_water
          - local_scent
          - scent_gradient_x (simulado)
          - scent_gradient_y
    """
    grid_h, grid_w = biomass_c4.shape
    
    # World -> Grid
    normalized = (position + world_bounds) / (2.0 * world_bounds)
    gx = jnp.clip((normalized[0] * grid_w).astype(jnp.int32), 0, grid_w - 1)
    gy = jnp.clip((normalized[1] * grid_h).astype(jnp.int32), 0, grid_h - 1)
    
    # Amostrar valores
    c4 = biomass_c4[gy, gx] / 600.0  # Norm
    c3 = biomass_c3[gy, gx] / 200.0  # Norm
    cover = cover_density[gy, gx]
    water = water_mask[gy, gx].astype(DTYPE_COMPUTE)
    scent = jnp.clip(scent_map[gy, gx] / 10.0, 0.0, 1.0)
    
    # Gradiente de odor (cheirar para saber de onde vem o predador)
    # Amostrar vizinhos simples
    sx_p = jnp.clip(gx + 1, 0, grid_w - 1)
    sx_m = jnp.clip(gx - 1, 0, grid_w - 1)
    sy_p = jnp.clip(gy + 1, 0, grid_h - 1)
    sy_m = jnp.clip(gy - 1, 0, grid_h - 1)
    
    scent_dx = (scent_map[gy, sx_p] - scent_map[gy, sx_m]) / 20.0
    scent_dy = (scent_map[sy_p, gx] - scent_map[sy_m, gx]) / 20.0
    
    return jnp.array([c4, c3, cover, water, scent, scent_dx, scent_dy], dtype=DTYPE_COMPUTE)


def build_observation(
    agent_idx: int,
    positions: Array,
    velocities: Array,
    energies: Array,
    ages: Array,
    is_active: Array,
    # Novos campos state.v2
    species_ids: Array,
    masses: Array,
    max_energies: Array,
    reproductive_cooldowns: Array,
    sexes: Array,
    # Ambiente v2
    biomass_c4: Array,
    biomass_c3: Array,
    cover_density: Array,
    water_mask: Array,
    scent_map: Array,
    # Params per-agent
    visual_ranges: Array,
    fovs: Array,
    num_neighbors: int = DEFAULT_NUM_NEIGHBORS,
) -> Array:
    """
    Constrói vetor de observação completo para um agente (v2).
    """
    own_pos = positions[agent_idx]
    own_vel = velocities[agent_idx]
    own_active = is_active[agent_idx]
    
    # Se inativo, retorna zeros
    # (Otimização: jax.lax.cond ou apenas mascarar no final)
    
    # 1. Propriocepção
    proprio = sense_proprioception(
        own_vel, energies[agent_idx], ages[agent_idx], 
        masses[agent_idx], max_energies[agent_idx], 
        reproductive_cooldowns[agent_idx], sexes[agent_idx]
    )
    
    # 2. Vizinhança (K-NN)
    # Calcular distâncias globais é caro O(N^2), mas necessário para física anyway.
    # Assumindo que distances matrix já foi computada externamente ou 
    # recomputamos aqui para este agente (O(N)).
    
    # Para performance, vamos computar distâncias apenas deste agente para os outros
    diff = positions - own_pos
    dists = jnp.linalg.norm(diff, axis=-1)
    # Mascarar self e inativos
    dists = jnp.where(is_active & (jnp.arange(len(positions)) != agent_idx), dists, jnp.inf)
    
    # Top-K
    near_dists, near_indices = jax.lax.top_k(-dists, num_neighbors)
    near_dists = -near_dists
    
    neighbor_obs = sense_neighbors(
        own_pos, positions, species_ids, masses,
        near_indices, near_dists,
        sensor_range=visual_ranges[agent_idx],
        fov=fovs[agent_idx],
        velocity=own_vel
    )
    
    # 3. Ambiente
    env_obs = sense_environment(
        own_pos, biomass_c4, biomass_c3, cover_density, water_mask, scent_map
    )
    
    # Concatenar
    obs = jnp.concatenate([proprio, neighbor_obs, env_obs])
    
    # Mascarar se inativo (retorna zeros)
    return jnp.where(own_active, obs, jnp.zeros_like(obs))


def build_all_observations(
    positions: Array,
    velocities: Array,
    energies: Array,
    ages: Array,
    is_active: Array,
    species_ids: Array,
    masses: Array,
    max_energies: Array,
    reproductive_cooldowns: Array,
    sexes: Array,
    biomass_c4: Array,
    biomass_c3: Array,
    cover_density: Array,
    water_mask: Array,
    scent_map: Array,
    visual_ranges: Array,
    fovs: Array,
    num_neighbors: int = DEFAULT_NUM_NEIGHBORS,
) -> Array:
    """
    Vetorização vmap de build_observation.
    """
    max_agents = positions.shape[0]
    indices = jnp.arange(max_agents)
    
    def _single(idx):
        return build_observation(
            idx, positions, velocities, energies, ages, is_active,
            species_ids, masses, max_energies, reproductive_cooldowns, sexes,
            biomass_c4, biomass_c3, cover_density, water_mask, scent_map,
            visual_ranges, fovs, num_neighbors
        )
        
    return jax.vmap(_single)(indices)


def compute_obs_dim(num_neighbors: int = DEFAULT_NUM_NEIGHBORS) -> int:
    """
    Calcula dimensão do vetor obs.
    Proprio: 2+1+1+1+1+1+1 = 8
    Neighbors: k * 6
    Env: 7
    Total = 15 + 6k
    Ex: k=5 -> 45 inputs
    """
    return 8 + (num_neighbors * 6) + 7

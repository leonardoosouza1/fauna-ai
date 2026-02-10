"""
Sistema Sensorial — Projeto Fauna.

Gera observações para cada agente a partir do estado do mundo.
Cada agente percebe:
  1. Propriocepção: velocidade, energia, idade (estado interno)
  2. Vizinhança: posição relativa dos k vizinhos mais próximos
  3. Recursos: gradiente de comida local
  4. Direção: orientação relativa ao recurso mais próximo

Design:
  - Vetorizado com JAX (vmap sobre agentes)
  - Dimensão de observação configurável
  - Compatível com bfloat16
"""

import jax
import jax.numpy as jnp

from src.core.types import Array, DTYPE_COMPUTE
from src.systems.physics import compute_distances, find_nearest_neighbors


# ──────────────────────────────────────────────
# Configurações do Sensor
# ──────────────────────────────────────────────
DEFAULT_SENSOR_RANGE = 15.0  # Distância máxima de percepção
DEFAULT_NUM_NEIGHBORS = 5    # Vizinhos percebidos


def sense_proprioception(
    velocity: Array,
    energy: Array,
    age: Array,
) -> Array:
    """
    Gera vetor de propriocepção de um único agente.

    Returns:
        proprio: (proprio_dim,) — velocidade + energia + idade normalizadas.
    """
    speed = jnp.linalg.norm(velocity)
    # Normalizar valores para [-1, 1] ou [0, 1]
    energy_norm = jnp.clip(energy / 200.0, 0.0, 1.0)
    age_norm = jnp.clip(age / 10000.0, 0.0, 1.0)

    # Concatenar: velocidade + speed + energia + idade
    return jnp.concatenate([
        velocity,                          # (spatial_dim,)
        jnp.array([speed]),                # (1,)
        jnp.array([energy_norm]),          # (1,)
        jnp.array([age_norm]),             # (1,)
    ]).astype(DTYPE_COMPUTE)


def sense_neighbors(
    agent_position: Array,
    all_positions: Array,
    neighbor_indices: Array,
    neighbor_distances: Array,
    sensor_range: float = DEFAULT_SENSOR_RANGE,
) -> Array:
    """
    Gera vetor de percepção de vizinhança para um agente.

    Args:
        agent_position: (spatial_dim,) — posição do agente
        all_positions: (max_agents, spatial_dim) — todas as posições
        neighbor_indices: (k,) — índices dos k vizinhos
        neighbor_distances: (k,) — distâncias aos k vizinhos
        sensor_range: Distância máxima de percepção

    Returns:
        neighbor_obs: (k * (spatial_dim + 1),) — posições relativas + distâncias
    """
    k = neighbor_indices.shape[0]
    spatial_dim = agent_position.shape[0]

    # Posições relativas dos vizinhos
    neighbor_positions = all_positions[neighbor_indices]  # (k, spatial_dim)
    relative_positions = neighbor_positions - agent_position  # (k, spatial_dim)

    # Normalizar pela distância máxima de percepção
    relative_positions = relative_positions / sensor_range

    # Normalizar distâncias
    norm_distances = jnp.clip(
        neighbor_distances / sensor_range, 0.0, 1.0,
    ).reshape(k, 1)

    # Mascarar vizinhos fora do alcance
    in_range = (neighbor_distances < sensor_range).reshape(k, 1).astype(DTYPE_COMPUTE)
    relative_positions = relative_positions * in_range
    norm_distances = norm_distances * in_range

    # Flatten: (k, spatial_dim + 1) → (k * (spatial_dim + 1),)
    neighbor_obs = jnp.concatenate(
        [relative_positions, norm_distances], axis=-1,
    ).flatten()

    return neighbor_obs.astype(DTYPE_COMPUTE)


def sense_food_gradient(
    position: Array,
    food_grid: Array,
    world_bounds: float = 50.0,
) -> Array:
    """
    Percepção do gradiente local de comida.

    Amostra a food_grid nas 4 direções cardinais do agente
    para estimar o gradiente de recursos.

    Args:
        position: (2,) — posição do agente
        food_grid: (grid_h, grid_w) — mapa de recursos
        world_bounds: Limites do mundo

    Returns:
        food_obs: (5,) — valor local + gradientes NE/SE/SW/NW
    """
    grid_h, grid_w = food_grid.shape

    # Converter coordenada mundo → grid
    def world_to_grid(pos):
        normalized = (pos + world_bounds) / (2.0 * world_bounds)
        gx = jnp.clip(normalized[0] * grid_w, 0, grid_w - 1).astype(jnp.int32)
        gy = jnp.clip(normalized[1] * grid_h, 0, grid_h - 1).astype(jnp.int32)
        return gx, gy

    gx, gy = world_to_grid(position)

    # Valor local
    local_food = food_grid[gy, gx]

    # Gradientes (amostrar vizinhos no grid)
    step = 1
    north = food_grid[jnp.clip(gy - step, 0, grid_h - 1), gx]
    south = food_grid[jnp.clip(gy + step, 0, grid_h - 1), gx]
    east = food_grid[gy, jnp.clip(gx + step, 0, grid_w - 1)]
    west = food_grid[gy, jnp.clip(gx - step, 0, grid_w - 1)]

    # Gradiente: diferença direcional
    grad_ns = (south - north)  # positivo = mais comida ao sul
    grad_ew = (east - west)    # positivo = mais comida a leste

    return jnp.array([
        local_food, grad_ns, grad_ew, north, east,
    ]).astype(DTYPE_COMPUTE)


def build_observation(
    agent_idx: int,
    positions: Array,
    velocities: Array,
    energies: Array,
    ages: Array,
    is_active: Array,
    food_grid: Array,
    num_neighbors: int = DEFAULT_NUM_NEIGHBORS,
    sensor_range: float = DEFAULT_SENSOR_RANGE,
) -> Array:
    """
    Constrói o vetor de observação completo para um agente.

    Args:
        agent_idx: Índice do agente no array
        positions: (max_agents, spatial_dim)
        velocities: (max_agents, spatial_dim)
        energies: (max_agents,)
        ages: (max_agents,)
        is_active: (max_agents,)
        food_grid: (grid_h, grid_w)
        num_neighbors: Número de vizinhos a perceber
        sensor_range: Alcance sensorial

    Returns:
        obs: (obs_dim,) — vetor de observação completo
    """
    spatial_dim = positions.shape[-1]
    own_pos = positions[agent_idx]
    own_vel = velocities[agent_idx]
    own_energy = energies[agent_idx]
    own_age = ages[agent_idx].astype(jnp.float32)

    # 1. Propriocepção
    proprio = sense_proprioception(own_vel, own_energy, own_age)

    # 2. Vizinhança
    distances = compute_distances(positions, is_active)
    agent_dists = distances[agent_idx]
    neigh_idx, neigh_dists = find_nearest_neighbors(
        agent_dists[None, :], k=num_neighbors,
    )
    neigh_idx = neigh_idx[0]
    neigh_dists = neigh_dists[0]

    neighbor_obs = sense_neighbors(
        own_pos, positions, neigh_idx, neigh_dists, sensor_range,
    )

    # 3. Comida
    food_obs = sense_food_gradient(own_pos, food_grid)

    # Concatenar tudo
    return jnp.concatenate([proprio, neighbor_obs, food_obs])


def build_all_observations(
    positions: Array,
    velocities: Array,
    energies: Array,
    ages: Array,
    is_active: Array,
    food_grid: Array,
    num_neighbors: int = DEFAULT_NUM_NEIGHBORS,
    sensor_range: float = DEFAULT_SENSOR_RANGE,
) -> Array:
    """
    Constrói observações para TODOS os agentes vetorizados com vmap.

    Returns:
        observations: (max_agents, obs_dim)
    """
    max_agents = positions.shape[0]

    def _single_obs(idx):
        return build_observation(
            idx, positions, velocities, energies, ages,
            is_active, food_grid, num_neighbors, sensor_range,
        )

    # vmap sobre os índices de agentes
    indices = jnp.arange(max_agents)
    observations = jax.vmap(_single_obs)(indices)

    return observations


def compute_obs_dim(
    spatial_dim: int = 2,
    num_neighbors: int = DEFAULT_NUM_NEIGHBORS,
) -> int:
    """
    Calcula a dimensão total do vetor de observação.

    Componentes:
        - Propriocepção: spatial_dim + 3 (vel + speed + energy + age)
        - Vizinhança: num_neighbors * (spatial_dim + 1)
        - Comida: 5
    """
    proprio_dim = spatial_dim + 3  # velocidade + speed + energia + idade
    neighbor_dim = num_neighbors * (spatial_dim + 1)
    food_dim = 5
    return proprio_dim + neighbor_dim + food_dim

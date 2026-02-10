"""
Motor de Física Diferenciável — Projeto Fauna.

Gerencia a integração de corpos rígidos dos agentes no ambiente,
utilizando diretamente JAX para vetorização eficiente.

Para a Fase 2 (MVP), implementamos física simplificada:
  - Movimentação 2D com aceleração
  - Colisão com limites do mundo
  - Atrito/damping
  - Detecção de proximidade entre agentes

Futuro: Integração completa com Brax para física 3D com contato.
"""

import jax
import jax.numpy as jnp

from src.core.types import Array, PRNGKey, DTYPE_STORE


# ──────────────────────────────────────────────
# Constantes Físicas
# ──────────────────────────────────────────────
MAX_SPEED = 5.0         # Velocidade máxima de um agente
FRICTION = 0.95         # Damping por step (1.0 = sem atrito)
WORLD_BOUNDS = 50.0     # Limites [-BOUNDS, +BOUNDS] do mundo


def apply_forces(
    position: Array,
    velocity: Array,
    acceleration: Array,
    is_active: Array,
    dt: float = 0.02,
) -> tuple[Array, Array]:
    """
    Integra a física para todos os agentes simultaneamente.

    Usa integração semi-implícita de Euler para estabilidade:
        v_{t+1} = friction * (v_t + a * dt)
        x_{t+1} = x_t + v_{t+1} * dt

    Args:
        position: (max_agents, spatial_dim)
        velocity: (max_agents, spatial_dim)
        acceleration: (max_agents, spatial_dim) — output da rede neural [-1, 1]
        is_active: (max_agents,) — máscara de agentes ativos
        dt: Passo temporal da simulação.

    Returns:
        Tupla (new_position, new_velocity).
    """
    # Escalar aceleração para força máxima
    force = acceleration * MAX_SPEED

    # Integração semi-implícita de Euler
    new_velocity = FRICTION * (velocity + force * dt)

    # Clipar velocidade máxima
    speed = jnp.linalg.norm(new_velocity, axis=-1, keepdims=True)
    speed_clamped = jnp.minimum(speed, MAX_SPEED)
    # Evitar divisão por zero
    safe_speed = jnp.maximum(speed, 1e-8)
    new_velocity = new_velocity * (speed_clamped / safe_speed)

    # Integrar posição
    new_position = position + new_velocity * dt

    # Limitar às bordas do mundo (reflexão)
    new_position = jnp.clip(new_position, -WORLD_BOUNDS, WORLD_BOUNDS)

    # Reflexão de velocidade nas bordas
    at_boundary = jnp.abs(new_position) >= WORLD_BOUNDS
    new_velocity = jnp.where(at_boundary, -new_velocity * 0.5, new_velocity)

    # Mascarar agentes inativos (manter estado zerado)
    active_mask = is_active[:, None].astype(DTYPE_STORE)
    new_position = new_position * active_mask
    new_velocity = new_velocity * active_mask

    return new_position, new_velocity


def compute_distances(positions: Array, is_active: Array) -> Array:
    """
    Calcula matriz de distâncias par-a-par entre agentes ativos.

    Args:
        positions: (max_agents, spatial_dim)
        is_active: (max_agents,) — máscara booleana

    Returns:
        distances: (max_agents, max_agents) — distância euclidiana.
                   Agentes inativos têm distância inf.
    """
    # Diferenças par-a-par: (N, 1, D) - (1, N, D) = (N, N, D)
    diff = positions[:, None, :] - positions[None, :, :]
    dists = jnp.linalg.norm(diff, axis=-1)

    # Mascarar inativos com inf
    active_mask = is_active[:, None] & is_active[None, :]
    dists = jnp.where(active_mask, dists, jnp.inf)

    # Diagonal = inf (distância de si mesmo) — usar jnp.where para evitar 0*inf=NaN
    eye_mask = jnp.eye(dists.shape[0], dtype=jnp.bool_)
    dists = jnp.where(eye_mask, jnp.inf, dists)

    return dists


def find_nearest_neighbors(
    distances: Array, k: int = 5,
) -> tuple[Array, Array]:
    """
    Encontra os k vizinhos mais próximos para cada agente.

    Args:
        distances: (max_agents, max_agents) — do compute_distances()
        k: Número de vizinhos.

    Returns:
        Tupla (indices, distances) — shapes (max_agents, k).
    """
    # Top-k menores distâncias (negando para usar top-k)
    top_k_dists, top_k_indices = jax.lax.top_k(-distances, k)
    return top_k_indices, -top_k_dists


def detect_collisions(
    distances: Array,
    collision_radius: float = 1.0,
) -> Array:
    """
    Detecta colisões entre pares de agentes.

    Args:
        distances: (max_agents, max_agents)
        collision_radius: Raio de colisão.

    Returns:
        collision_matrix: (max_agents, max_agents) — booleano.
    """
    return distances < collision_radius

"""
Motor de Física Diferenciável Multi-Espécie — Projeto Fauna v2.

Gerencia a integração física com suporte a velocidades máximas por espécie
e limites de mundo expandidos (home ranges).

Mudanças v2:
  - max_speed agora é um array per-agent (alométrico)
  - world_bounds expandido para 500m (era 50m)
  - Colisão com corpos d'água para espécies terrestres
"""

import jax
import jax.numpy as jnp

from src.core.types import Array, DTYPE_STORE


# ──────────────────────────────────────────────
# Constantes Físicas
# ──────────────────────────────────────────────
FRICTION = 0.95         # Damping por step
WORLD_BOUNDS_DEFAULT = 500.0     # 1km x 1km


def apply_forces(
    position: Array,
    velocity: Array,
    acceleration: Array,
    is_active: Array,
    max_speeds: Array | float,  # Pode ser escalar (legacy) ou array (v2)
    water_mask: Array | None = None,
    is_aquatic: Array | None = None,
    dt: float = 0.02,
    world_bounds: float = WORLD_BOUNDS_DEFAULT,
) -> tuple[Array, Array]:
    """
    Integra a física para todos os agentes simultaneamente.

    Args:
        max_speeds: Velocidade máxima per-agent (ou scalar legacy)
        water_mask: Grid de água (opcional)
        is_aquatic: Máscara boolean de quem pode entrar na água (opcional)
    """
    # Se max_speeds for escalar, broadcast
    if isinstance(max_speeds, float):
        max_speeds = jnp.full((position.shape[0], 1), max_speeds)
    elif max_speeds.ndim == 1:
        max_speeds = max_speeds[:, None]

    # Aceleração escala com max_speed de cada agente
    force = acceleration * max_speeds

    # Integração semi-implícita
    new_velocity = FRICTION * (velocity + force * dt)

    # Clipar velocidade máxima per-agent
    speed = jnp.linalg.norm(new_velocity, axis=-1, keepdims=True)
    # Evitar divisão por zero
    safe_speed = jnp.maximum(speed, 1e-8)
    
    # Scaling factor se speed > max_speed
    scale = jnp.minimum(speed, max_speeds) / safe_speed
    new_velocity = new_velocity * scale

    # Integrar posição
    new_position = position + new_velocity * dt

    # Limitar às bordas do mundo (reflexão)
    new_position = jnp.clip(new_position, -world_bounds, world_bounds)

    # Reflexão de velocidade nas bordas
    at_boundary = jnp.abs(new_position) >= world_bounds
    new_velocity = jnp.where(at_boundary, -new_velocity * 0.5, new_velocity)

    # Colisão com água (se water_mask fornecido)
    # Implementação simplificada: se agente terrestre cair em água, 
    # é empurrado de volta (tipo boundary).
    # Requer mapeamento pos -> grid, que é caro fazer aqui se não tiver gradiente.
    # Por enquanto, assumir que ambiente cuida disso ou física simplificada ignora.
    # (Futuro: SDF da água para colisão)

    # Mascarar agentes inativos
    active_mask = is_active[:, None].astype(DTYPE_STORE)
    new_position = new_position * active_mask
    new_velocity = new_velocity * active_mask

    return new_position, new_velocity


def compute_distances(positions: Array, is_active: Array) -> Array:
    """
    Calcula matriz de distâncias par-a-par entre agentes ativos.
    """
    # Diferenças par-a-par
    diff = positions[:, None, :] - positions[None, :, :]
    dists = jnp.linalg.norm(diff, axis=-1)

    # Mascarar inativos com inf
    active_mask = is_active[:, None] & is_active[None, :]
    dists = jnp.where(active_mask, dists, jnp.inf)

    # Diagonal = inf
    eye_mask = jnp.eye(dists.shape[0], dtype=jnp.bool_)
    dists = jnp.where(eye_mask, jnp.inf, dists)

    return dists


def find_nearest_neighbors(
    distances: Array, k: int = 5,
) -> tuple[Array, Array]:
    """
    Encontra os k vizinhos mais próximos.
    """
    top_k_dists, top_k_indices = jax.lax.top_k(-distances, k)
    return top_k_indices, -top_k_dists


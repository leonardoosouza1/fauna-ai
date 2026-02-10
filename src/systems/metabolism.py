"""
Sistema Metabólico — Projeto Fauna.

Gerencia o ciclo de vida dos agentes:
  - Consumo de energia (custo de existência + movimento + pensamento)
  - Alimentação (recuperação de energia ao encontrar recursos)
  - Morte (energia ≤ 0 → desativação do slot)
  - Reprodução (energia ≥ threshold → divisão do agente)

O metabolismo é a pressão seletiva principal:
agentes que não encontram comida ou desperdiçam energia morrem,
criando a seleção natural que impulsiona a evolução.
"""

import jax
import jax.numpy as jnp

from src.core.types import Array, PRNGKey, DTYPE_STORE


# ──────────────────────────────────────────────
# Constantes Metabólicas
# ──────────────────────────────────────────────
BASAL_COST = 0.1          # Custo de existir por step
MOVEMENT_COST_FACTOR = 0.05  # Custo proporcional à velocidade
THINK_COST = 0.02         # Custo de processamento neural
FOOD_ENERGY = 30.0        # Energia ganha ao comer
REPRODUCTION_THRESHOLD = 150.0  # Energia mínima para reprodução
REPRODUCTION_COST = 80.0  # Energia gasta na reprodução
MAX_ENERGY = 200.0        # Energia máxima
MAX_AGE = 50000           # Idade máxima antes de morte natural


def compute_energy_costs(
    velocity: Array,
    is_active: Array,
) -> Array:
    """
    Calcula o custo energético de cada agente por step.

    Args:
        velocity: (max_agents, spatial_dim)
        is_active: (max_agents,)

    Returns:
        costs: (max_agents,) — custo total por agente
    """
    # Custo de movimento (proporcional à velocidade)
    speed = jnp.linalg.norm(velocity, axis=-1)
    movement_cost = speed * MOVEMENT_COST_FACTOR

    # Custo total
    total_cost = (BASAL_COST + movement_cost + THINK_COST) * is_active.astype(DTYPE_STORE)

    return total_cost


def consume_food(
    positions: Array,
    food_grid: Array,
    is_active: Array,
    eating_radius: float = 2.0,
    world_bounds: float = 50.0,
) -> tuple[Array, Array]:
    """
    Agentes próximos a recursos coletam comida.

    Args:
        positions: (max_agents, 2)
        food_grid: (grid_h, grid_w)
        is_active: (max_agents,)
        eating_radius: Raio de coleta
        world_bounds: Limites do mundo

    Returns:
        energy_gained: (max_agents,) — energia ganha por cada agente
        new_food_grid: (grid_h, grid_w) — grid atualizado
    """
    grid_h, grid_w = food_grid.shape

    # Converter posição do mundo para coordenadas do grid
    normalized = (positions + world_bounds) / (2.0 * world_bounds)
    gx = jnp.clip((normalized[:, 0] * grid_w).astype(jnp.int32), 0, grid_w - 1)
    gy = jnp.clip((normalized[:, 1] * grid_h).astype(jnp.int32), 0, grid_h - 1)

    # Verificar se há comida no local
    food_at_agent = food_grid[gy, gx]
    can_eat = (food_at_agent > 0) & is_active

    # Energia ganha
    energy_gained = jnp.where(can_eat, FOOD_ENERGY, 0.0).astype(DTYPE_STORE)

    # Remover comida consumida do grid
    new_food_grid = food_grid
    # Usar scan para atualizar grid sequencialmente (evitar race conditions)
    def _consume_one(grid, idx):
        active = can_eat[idx]
        grid = jnp.where(
            active,
            grid.at[gy[idx], gx[idx]].add(-1.0),
            grid,
        )
        return grid, None

    new_food_grid, _ = jax.lax.scan(
        _consume_one, new_food_grid, jnp.arange(positions.shape[0]),
    )
    new_food_grid = jnp.maximum(new_food_grid, 0.0)

    return energy_gained, new_food_grid


def update_metabolism(
    energy: Array,
    age: Array,
    velocity: Array,
    is_active: Array,
    energy_gained: Array,
) -> tuple[Array, Array, Array]:
    """
    Atualiza energia, idade e status de vida dos agentes.

    Args:
        energy: (max_agents,)
        age: (max_agents,)
        velocity: (max_agents, spatial_dim)
        is_active: (max_agents,)
        energy_gained: (max_agents,) — energia obtida da alimentação

    Returns:
        new_energy, new_age, new_is_active
    """
    # Custos
    costs = compute_energy_costs(velocity, is_active)

    # Atualizar energia
    new_energy = energy - costs + energy_gained
    new_energy = jnp.clip(new_energy, 0.0, MAX_ENERGY)

    # Incrementar idade
    new_age = age + is_active.astype(jnp.int32)

    # Determinar mortes (energia zerada OU idade máxima)
    alive = (new_energy > 0) & (new_age < MAX_AGE) & is_active
    new_is_active = alive

    # Zerar energia de agentes mortos
    new_energy = new_energy * alive.astype(DTYPE_STORE)

    return new_energy, new_age, new_is_active


def respawn_food(
    food_grid: Array,
    rng: PRNGKey,
    respawn_rate: float = 0.01,
    max_food_per_cell: float = 5.0,
) -> tuple[Array, PRNGKey]:
    """
    Regenera comida no mundo com probabilidade respawn_rate por célula.

    Args:
        food_grid: (grid_h, grid_w)
        rng: Chave PRNG para aleatoriedade
        respawn_rate: Probabilidade de respawn por célula
        max_food_per_cell: Capacidade máxima de uma célula

    Returns:
        new_food_grid, new_rng
    """
    rng, key = jax.random.split(rng)

    # Gerar máscara de respawn
    spawn_mask = jax.random.uniform(key, food_grid.shape) < respawn_rate

    # Adicionar comida onde a máscara é True
    new_food = food_grid + spawn_mask.astype(DTYPE_STORE)
    new_food = jnp.minimum(new_food, max_food_per_cell)

    return new_food, rng


def check_reproduction_eligibility(
    energy: Array,
    is_active: Array,
) -> Array:
    """
    Verifica quais agentes podem se reproduzir.

    Returns:
        can_reproduce: (max_agents,) — booleano
    """
    return (energy >= REPRODUCTION_THRESHOLD) & is_active


def reproduce_agent(
    parent_energy: float,
) -> tuple[float, float]:
    """
    Divide energia entre pai e filho durante reprodução.

    Returns:
        (parent_remaining_energy, child_initial_energy)
    """
    parent_remaining = parent_energy - REPRODUCTION_COST
    child_energy = REPRODUCTION_COST * 0.5  # Filho recebe 50% do custo
    return parent_remaining, child_energy

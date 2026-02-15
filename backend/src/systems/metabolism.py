"""
Sistema Metabólico Alométrico — Projeto Fauna v2.

Gerencia o ciclo de vida dos agentes com escalonamento massa-dependente:
  - TMB via Lei de Kleiber: BMR = 3.5 × M^0.75 (Watts)
  - COT (Custo de Transporte): 10.7 × M^(-0.32) × distância
  - Eficiência digestiva sazonal (55% verão, 40% inverno)
  - Termorregulação para espécies pequenas
  - Energia máxima proporcional à massa

O metabolismo é a pressão seletiva principal:
agentes que não encontram comida ou desperdiçam energia morrem,
criando a seleção natural que impulsiona a evolução.

Referências:
    - Kleiber (1947) — scaling metabólico
    - Taylor et al. (1982) — COT terrestre
    - Speakman (2005) — FMR mamíferos neotropicais
"""

import jax
import jax.numpy as jnp

from src.core.types import Array, PRNGKey, DTYPE_STORE


# ──────────────────────────────────────────────
# Constantes Metabólicas (Kleiber-based)
# ──────────────────────────────────────────────
KLEIBER_ALPHA = 3.5           # BMR = α × M^0.75 (Watts)
KLEIBER_EXPONENT = 0.75       # Expoente alométrico
COT_ALPHA = 10.7              # COT = α × M^(-0.32) J/(kg·m)
COT_EXPONENT = -0.32
THINK_COST_FRACTION = 0.05    # 5% da TMB vai para processamento neural

# Conversão: 1 Watt = 1 J/s. Se 1 step ≈ 0.1 dia = 8640s:
# BMR_per_step = BMR_watts * 8640 / 1000 = BMR * 8.64 kJ/step
SECONDS_PER_STEP = 8640.0     # 0.1 dia = 8640 segundos
J_TO_KJ = 0.001

# Sazonalidade
SUMMER_EFFICIENCY = 0.55      # Eficiência digestiva no verão
WINTER_EFFICIENCY = 0.40      # Eficiência digestiva no inverno

# Termorregulação
THERMONEUTRAL_LOWER = 15.0    # Temperatura abaixo da qual há custo extra (°C)
THERMO_COST_COEFF = 0.002     # kJ extra por °C abaixo de 15°C por kg de massa

# Reprodução
MAX_AGE = 50000               # Idade máxima antes de morte natural (steps)


# ══════════════════════════════════════════════
# Custos Energéticos
# ══════════════════════════════════════════════

def compute_bmr_cost(mass: Array) -> Array:
    """
    TMB por step via Kleiber: BMR_kJ/step = 3.5 × M^0.75 × 8.64.

    Args:
        mass: (max_agents,) — massa corporal em kg

    Returns:
        bmr_cost: (max_agents,) — custo basal em kJ/step
    """
    bmr_watts = KLEIBER_ALPHA * jnp.power(mass, KLEIBER_EXPONENT)
    bmr_kj_per_step = bmr_watts * SECONDS_PER_STEP * J_TO_KJ
    return bmr_kj_per_step.astype(DTYPE_STORE)


def compute_movement_cost(
    velocity: Array,
    mass: Array,
    dt: float = 1.0,
) -> Array:
    """
    COT (Custo de Transporte): 10.7 × M^(-0.32) × M × distance.

    Args:
        velocity: (max_agents, spatial_dim)
        mass: (max_agents,) — massa corporal em kg
        dt: Passo temporal

    Returns:
        movement_cost: (max_agents,) — custo de movimento em kJ
    """
    speed = jnp.linalg.norm(velocity, axis=-1)
    distance = speed * dt

    # COT total = COT_per_kg_m × mass × distance (em Joules)
    cot_per_kg_m = COT_ALPHA * jnp.power(jnp.maximum(mass, 0.01), COT_EXPONENT)
    cost_joules = cot_per_kg_m * mass * distance

    return (cost_joules * J_TO_KJ).astype(DTYPE_STORE)


def compute_think_cost(mass: Array) -> Array:
    """
    Custo neural como fração da TMB.

    Args:
        mass: (max_agents,)

    Returns:
        think_cost: (max_agents,) — em kJ/step
    """
    bmr = compute_bmr_cost(mass)
    return (bmr * THINK_COST_FRACTION).astype(DTYPE_STORE)


def compute_thermoregulation_cost(
    mass: Array,
    temperature: Array,
) -> Array:
    """
    Custo de termorregulação abaixo da zona termoneutra.

    Espécies pequenas (Preá ~0.55kg) gastam mais energia
    proporcionalmente para manter temperatura corporal.

    Args:
        mass: (max_agents,) — kg
        temperature: escalar — °C

    Returns:
        thermo_cost: (max_agents,) — kJ/step
    """
    temp_deficit = jnp.maximum(THERMONEUTRAL_LOWER - temperature, 0.0)
    # Custo inversamente proporcional à massa (superfície/volume)
    surface_area_ratio = jnp.power(jnp.maximum(mass, 0.01), -0.33)
    cost = temp_deficit * THERMO_COST_COEFF * mass * surface_area_ratio
    return cost.astype(DTYPE_STORE)


def compute_energy_costs(
    velocity: Array,
    is_active: Array,
    mass: Array | None = None,
    temperature: Array | None = None,
) -> Array:
    """
    Calcula o custo energético total de cada agente por step.

    Se mass é None, usa custo fixo legacy (retrocompatibilidade).

    Args:
        velocity: (max_agents, spatial_dim)
        is_active: (max_agents,)
        mass: (max_agents,) — massa corporal (opcional)
        temperature: escalar — temperatura (opcional)

    Returns:
        costs: (max_agents,) — custo total por agente (kJ/step)
    """
    if mass is None:
        # Modo legacy (sem alometria)
        speed = jnp.linalg.norm(velocity, axis=-1)
        total_cost = (0.1 + speed * 0.05 + 0.02)
        return (total_cost * is_active.astype(DTYPE_STORE)).astype(DTYPE_STORE)

    # Modo alométrico
    bmr = compute_bmr_cost(mass)
    movement = compute_movement_cost(velocity, mass)
    think = compute_think_cost(mass)

    total = bmr + movement + think

    # Termorregulação (se temperatura disponível)
    if temperature is not None:
        thermo = compute_thermoregulation_cost(mass, temperature)
        total = total + thermo

    return (total * is_active.astype(DTYPE_STORE)).astype(DTYPE_STORE)


# ══════════════════════════════════════════════
# Alimentação
# ══════════════════════════════════════════════

def consume_food(
    positions: Array,
    food_grid: Array,
    is_active: Array,
    eating_radius: float = 2.0,
    world_bounds: float = 500.0,
) -> tuple[Array, Array]:
    """
    Agentes próximos a recursos coletam comida (modo legacy).

    Mantido para retrocompatibilidade com pipeline original.
    Para multi-espécie, usar environment.consume_vegetation().
    """
    grid_h, grid_w = food_grid.shape

    normalized = (positions + world_bounds) / (2.0 * world_bounds)
    gx = jnp.clip((normalized[:, 0] * grid_w).astype(jnp.int32), 0, grid_w - 1)
    gy = jnp.clip((normalized[:, 1] * grid_h).astype(jnp.int32), 0, grid_h - 1)

    food_at_agent = food_grid[gy, gx]
    can_eat = (food_at_agent > 0) & is_active

    energy_gained = jnp.where(can_eat, 30.0, 0.0).astype(DTYPE_STORE)

    new_food_grid = food_grid

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


# ══════════════════════════════════════════════
# Atualização Metabólica Principal
# ══════════════════════════════════════════════

def update_metabolism(
    energy: Array,
    age: Array,
    velocity: Array,
    is_active: Array,
    energy_gained: Array,
    mass: Array | None = None,
    max_energy: Array | None = None,
    temperature: Array | None = None,
) -> tuple[Array, Array, Array]:
    """
    Atualiza energia, idade e status de vida dos agentes.

    v2: Usa custos alométricos se mass é fornecido.
        Max energy per-agent se max_energy fornecido.

    Args:
        energy: (max_agents,) — energia atual
        age: (max_agents,) — idade em steps
        velocity: (max_agents, spatial_dim)
        is_active: (max_agents,)
        energy_gained: (max_agents,) — energia obtida da alimentação
        mass: (max_agents,) — massa corporal (opcional)
        max_energy: (max_agents,) — capacidade máxima (opcional)
        temperature: escalar — temperatura (opcional)

    Returns:
        (new_energy, new_age, new_is_active)
    """
    # Custos
    costs = compute_energy_costs(velocity, is_active, mass, temperature)

    # Atualizar energia
    new_energy = energy - costs + energy_gained

    # Capacidade máxima per-agent (ou global legacy)
    if max_energy is not None:
        cap = max_energy
    else:
        cap = jnp.full_like(new_energy, 200.0)

    new_energy = jnp.clip(new_energy, 0.0, cap)

    # Incrementar idade
    new_age = age + is_active.astype(jnp.int32)

    # Determinar mortes (energia zerada OU idade máxima)
    alive = (new_energy > 0) & (new_age < MAX_AGE) & is_active
    new_is_active = alive

    # Zerar energia de agentes mortos
    new_energy = new_energy * alive.astype(DTYPE_STORE)

    return new_energy, new_age, new_is_active


# ══════════════════════════════════════════════
# Reprodução
# ══════════════════════════════════════════════

def check_reproduction_eligibility(
    energy: Array,
    is_active: Array,
    max_energy: Array | None = None,
    reproductive_cooldown: Array | None = None,
) -> Array:
    """
    Verifica quais agentes podem se reproduzir.

    v2: Threshold relativo a max_energy (75%) + cooldown check.

    Returns:
        can_reproduce: (max_agents,) — booleano
    """
    if max_energy is not None:
        threshold = max_energy * 0.75
    else:
        threshold = jnp.full_like(energy, 150.0)

    eligible = (energy >= threshold) & is_active

    if reproductive_cooldown is not None:
        eligible = eligible & (reproductive_cooldown <= 0)

    return eligible


def reproduce_agent(
    parent_energy: float,
    energy_cost_fraction: float = 0.3,
) -> tuple[float, float]:
    """
    Divide energia entre pai e filho durante reprodução.

    Args:
        parent_energy: Energia do pai
        energy_cost_fraction: Fração da energia gasta (espécie-specific)

    Returns:
        (parent_remaining_energy, child_initial_energy)
    """
    cost = parent_energy * energy_cost_fraction
    parent_remaining = parent_energy - cost
    child_energy = cost * 0.5   # Filho recebe 50% do custo
    return parent_remaining, child_energy


# ══════════════════════════════════════════════
# Regeneração de Comida (Legacy)
# ══════════════════════════════════════════════

def respawn_food(
    food_grid: Array,
    rng: PRNGKey,
    respawn_rate: float = 0.01,
    max_food_per_cell: float = 5.0,
) -> tuple[Array, PRNGKey]:
    """
    Regenera comida no mundo (modo legacy).
    Para multi-espécie, usar environment.grow_vegetation().
    """
    rng, key = jax.random.split(rng)
    spawn_mask = jax.random.uniform(key, food_grid.shape) < respawn_rate
    new_food = food_grid + spawn_mask.astype(DTYPE_STORE)
    new_food = jnp.minimum(new_food, max_food_per_cell)
    return new_food, rng

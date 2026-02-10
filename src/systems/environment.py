"""
Sistema Ambiental Sazonal — Projeto Fauna v2.

Gerencia a dinâmica do ambiente:
  - Ciclo sazonal (verão/inverno) com temperatura e fotoperíodo
  - Crescimento de gramíneas C3 (inverno) e C4 (verão)
  - Cobertura vegetal que afeta probabilidade de fuga
  - Mapa de odores de predadores com decay exponencial
  - Presença de água (corpos d'água estáticos)

Parametrizado por BiomeConfig para escalabilidade regional.

Referências:
    - ANPP Pampa: 0.53 kg/m²/ano (Semmartin et al., 2004)
    - Growth C4: pico 4 g/m²/day no verão (IBGE Pampa)
    - Growth C3: pico 0.8 g/m²/day no inverno (festuca, aveia)
"""

import jax
import jax.numpy as jnp

from src.core.types import Array, PRNGKey, DTYPE_STORE


# ──────────────────────────────────────────────
# Constantes Ambientais
# ──────────────────────────────────────────────
YEAR_LENGTH_STEPS = 3650       # 1 ano ≈ 3650 steps (1 step ≈ 0.1 dia)
TWO_PI = 2.0 * jnp.pi

# Produtividade de gramíneas (g/m²/step, escalado de g/m²/day)
C4_MAX_GROWTH = 4.0 / 10.0    # 4 g/m²/dia → por step
C3_MAX_GROWTH = 0.8 / 10.0    # 0.8 g/m²/dia → por step

# Limites de biomassa (g/m²)
MAX_BIOMASS_C4 = 600.0        # Máximo standing crop
MAX_BIOMASS_C3 = 200.0
MAX_COVER_DENSITY = 1.0

# Odor
SCENT_DECAY_RATE = 0.995       # Decay por step (~200 steps para cair 50%)
SCENT_DEPOSIT_AMOUNT = 1.0     # Intensidade de depósito por predador/step


# ══════════════════════════════════════════════
# Ciclo Sazonal
# ══════════════════════════════════════════════

def advance_season(
    season_phase: Array,
    mean_temp: float = 18.0,
    amplitude: float = 10.0,
) -> tuple[Array, Array]:
    """
    Avança a fase sazonal em 1 step e calcula temperatura.

    Args:
        season_phase: Fase atual [0, 2π]
        mean_temp: Temperatura média anual (°C)
        amplitude: Amplitude sazonal (°C)

    Returns:
        (new_phase, temperature)
    """
    new_phase = (season_phase + TWO_PI / YEAR_LENGTH_STEPS) % TWO_PI
    temperature = mean_temp + amplitude * jnp.sin(new_phase)
    return new_phase, temperature


def get_season_factor(season_phase: Array) -> tuple[Array, Array]:
    """
    Calcula fatores de crescimento sazonal para C3 e C4.

    C4 (gramíneas tropicais): pico no verão (phase ≈ π/2)
    C3 (gramíneas temperadas): pico no inverno (phase ≈ 3π/2)

    Returns:
        (c4_factor, c3_factor) — cada um em [0, 1]
    """
    c4_factor = jnp.clip(jnp.sin(season_phase), 0.0, 1.0)   # Verão
    c3_factor = jnp.clip(-jnp.sin(season_phase), 0.0, 1.0)  # Inverno
    return c4_factor, c3_factor


# ══════════════════════════════════════════════
# Crescimento de Vegetação
# ══════════════════════════════════════════════

def grow_vegetation(
    biomass_c4: Array,
    biomass_c3: Array,
    cover_density: Array,
    season_phase: Array,
    rng: PRNGKey,
) -> tuple[Array, Array, Array, PRNGKey]:
    """
    Crescimento das gramíneas C3 e C4 baseado na estação.

    Modelo logístico: growth = G_max * factor * (1 - B / B_max)
    Cobertura vegetal cresce proporcionalmente à biomassa total.

    Args:
        biomass_c4: (H, W) — biomassa C4 atual
        biomass_c3: (H, W) — biomassa C3 atual
        cover_density: (H, W) — cobertura vegetal [0, 1]
        season_phase: escalar — fase sazonal
        rng: Chave PRNG

    Returns:
        (new_c4, new_c3, new_cover, new_rng)
    """
    rng, key = jax.random.split(rng)
    c4_factor, c3_factor = get_season_factor(season_phase)

    # Crescimento logístico C4
    c4_growth = C4_MAX_GROWTH * c4_factor * (1.0 - biomass_c4 / MAX_BIOMASS_C4)
    c4_growth = jnp.maximum(c4_growth, 0.0)

    # Crescimento logístico C3
    c3_growth = C3_MAX_GROWTH * c3_factor * (1.0 - biomass_c3 / MAX_BIOMASS_C3)
    c3_growth = jnp.maximum(c3_growth, 0.0)

    # Estocasticidade espacial (heterogeneidade local)
    noise = jax.random.uniform(key, biomass_c4.shape, minval=0.8, maxval=1.2)
    c4_growth = c4_growth * noise

    new_c4 = jnp.clip(biomass_c4 + c4_growth, 0.0, MAX_BIOMASS_C4)
    new_c3 = jnp.clip(biomass_c3 + c3_growth, 0.0, MAX_BIOMASS_C3)

    # Cobertura vegetal: média ponderada da biomassa normalizada
    total_biomass_frac = (new_c4 / MAX_BIOMASS_C4 + new_c3 / MAX_BIOMASS_C3) / 2.0
    new_cover = jnp.clip(
        cover_density * 0.99 + total_biomass_frac * 0.01,
        0.0, MAX_COVER_DENSITY,
    )

    return new_c4, new_c3, new_cover, rng


def consume_vegetation(
    biomass_c4: Array,
    biomass_c3: Array,
    positions: Array,
    is_active: Array,
    species_id: Array,
    mass: Array,
    world_bounds: float = 500.0,
) -> tuple[Array, Array, Array]:
    """
    Herbívoros consomem biomassa da célula em que se encontram.

    Consumo proporcional à massa do animal (Kleiber intake scaling).
    Herbívoros (species_id 1-3 no Pampa) comem C4 preferencialmente,
    depois C3 se C4 não disponível.

    Returns:
        (energy_gained, new_c4, new_c3)
    """
    grid_h, grid_w = biomass_c4.shape

    # Converter posição do mundo para coordenadas do grid
    normalized = (positions + world_bounds) / (2.0 * world_bounds)
    gx = jnp.clip((normalized[:, 0] * grid_w).astype(jnp.int32), 0, grid_w - 1)
    gy = jnp.clip((normalized[:, 1] * grid_h).astype(jnp.int32), 0, grid_h - 1)

    # Quem é herbívoro? (diet_type 1 = HERBIVORE)
    # No Pampa: species_id 1 (Preá), 2 (Veado), 3 (Capivara)
    is_herbivore = (species_id >= 1) & (species_id <= 3) & is_active

    # Consumo diário ∝ mass^0.75 (Daily Metabolic Intake)
    # Convertido para g/step de biomassa consumida
    daily_intake_g_step = 0.05 * mass ** 0.75  # g por step
    intake = daily_intake_g_step * is_herbivore.astype(DTYPE_STORE)

    # Verificar disponibilidade C4 primeiro
    c4_available = biomass_c4[gy, gx]
    c3_available = biomass_c3[gy, gx]

    # Come C4 preferencialmente
    c4_eaten = jnp.minimum(intake, c4_available)
    remaining_intake = intake - c4_eaten

    # Se sobrou apetite, come C3
    c3_eaten = jnp.minimum(remaining_intake, c3_available)

    total_eaten = c4_eaten + c3_eaten

    # Energia ganha (kJ) — eficiência digestiva ~55%
    efficiency = 0.55
    energy_density_kj_per_g = 0.018  # ~18 kJ/kg MS → 0.018 kJ/g
    energy_gained = total_eaten * efficiency * energy_density_kj_per_g

    # Remover biomassa consumida do grid (scan sequencial)
    def _consume_one_c4(grid, idx):
        active = is_herbivore[idx]
        grid = jnp.where(
            active,
            grid.at[gy[idx], gx[idx]].add(-c4_eaten[idx]),
            grid,
        )
        return grid, None

    def _consume_one_c3(grid, idx):
        active = is_herbivore[idx]
        grid = jnp.where(
            active,
            grid.at[gy[idx], gx[idx]].add(-c3_eaten[idx]),
            grid,
        )
        return grid, None

    new_c4, _ = jax.lax.scan(_consume_one_c4, biomass_c4, jnp.arange(positions.shape[0]))
    new_c3, _ = jax.lax.scan(_consume_one_c3, biomass_c3, jnp.arange(positions.shape[0]))

    new_c4 = jnp.maximum(new_c4, 0.0)
    new_c3 = jnp.maximum(new_c3, 0.0)

    return energy_gained, new_c4, new_c3


# ══════════════════════════════════════════════
# Mapa de Odores (Ecologia do Medo)
# ══════════════════════════════════════════════

def update_scent_map(
    scent_map: Array,
    positions: Array,
    is_active: Array,
    species_id: Array,
    world_bounds: float = 500.0,
) -> Array:
    """
    Atualiza mapa de odores: predadores depositam, odores decaem.

    Predadores (species_id >= 4) depositam feromônios que
    influenciam o comportamento de fuga de herbívoros.

    Args:
        scent_map: (H, W) — mapa de intensidade de odor
        positions: (max_agents, 2) — posições dos agentes
        is_active: (max_agents,) — máscara de ativos
        species_id: (max_agents,) — IDs de espécie
        world_bounds: Limites do mundo

    Returns:
        new_scent_map: (H, W)
    """
    grid_h, grid_w = scent_map.shape

    # Decay exponencial global
    new_scent = scent_map * SCENT_DECAY_RATE

    # Predadores depositam odor (species_id >= 4)
    is_predator = (species_id >= 4) & is_active

    # Converter posições para grid
    normalized = (positions + world_bounds) / (2.0 * world_bounds)
    gx = jnp.clip((normalized[:, 0] * grid_w).astype(jnp.int32), 0, grid_w - 1)
    gy = jnp.clip((normalized[:, 1] * grid_h).astype(jnp.int32), 0, grid_h - 1)

    # Depositar odor
    def _deposit(grid, idx):
        active = is_predator[idx]
        grid = jnp.where(
            active,
            grid.at[gy[idx], gx[idx]].add(SCENT_DEPOSIT_AMOUNT),
            grid,
        )
        return grid, None

    new_scent, _ = jax.lax.scan(_deposit, new_scent, jnp.arange(positions.shape[0]))
    new_scent = jnp.clip(new_scent, 0.0, 10.0)

    return new_scent


# ══════════════════════════════════════════════
# Inicialização do Ambiente
# ══════════════════════════════════════════════

def initialize_environment(
    grid_size: tuple[int, int] = (128, 128),
    initial_c4: float = 200.0,
    initial_c3: float = 80.0,
    forest_fraction: float = 0.25,
    water_fraction: float = 0.05,
    rng_seed: int = 42,
) -> dict:
    """
    Inicializa o ambiente do bioma com biomassa, cobertura e água.

    Returns:
        dict com: biomass_c4, biomass_c3, cover_density, water_mask
    """
    rng = jax.random.PRNGKey(rng_seed)
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

    # Biomassa inicial com variação espacial
    c4_noise = jax.random.uniform(rng1, grid_size, minval=0.5, maxval=1.5)
    c3_noise = jax.random.uniform(rng2, grid_size, minval=0.5, maxval=1.5)

    biomass_c4 = jnp.clip(initial_c4 * c4_noise, 0.0, MAX_BIOMASS_C4).astype(DTYPE_STORE)
    biomass_c3 = jnp.clip(initial_c3 * c3_noise, 0.0, MAX_BIOMASS_C3).astype(DTYPE_STORE)

    # Cobertura florestal (patches de floresta)
    cover_noise = jax.random.uniform(rng3, grid_size)
    cover_density = (cover_noise < forest_fraction).astype(DTYPE_STORE)

    # Água (corpos d'água estáticos — rios/lagoas)
    water_noise = jax.random.uniform(rng4, grid_size)
    water_mask = (water_noise < water_fraction)

    return {
        "biomass_c4": biomass_c4,
        "biomass_c3": biomass_c3,
        "cover_density": cover_density,
        "water_mask": water_mask,
    }


def compute_total_food_grid(biomass_c4: Array, biomass_c3: Array) -> Array:
    """
    Computa food_grid retrocompatível como soma de biomassa normalizada.

    Returns:
        food_grid: (H, W) — biomassa total em unidades de "comida original"
    """
    total = biomass_c4 + biomass_c3
    # Normalizar para escala compatível com food_grid original (0-5)
    return jnp.clip(total / 160.0, 0.0, 5.0).astype(DTYPE_STORE)

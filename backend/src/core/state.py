"""
Estado do Mundo e dos Agentes — Projeto Fauna v2.

Dataclasses imutáveis compatíveis com JAX pytrees.
Toda atualização gera um novo estado: f(State_t) -> State_{t+1}.

Versão v2: Extende o estado para suportar multi-espécie com
parâmetros alométricos, sazonalidade e biomas escaláveis.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from src.core.types import Array, PRNGKey, DTYPE_COMPUTE, DTYPE_STORE


# ──────────────────────────────────────────────
# Estado Neural (cérebro CfC/NCP de um agente)
# ──────────────────────────────────────────────
class BrainState(NamedTuple):
    """Estado oculto da rede neural líquida."""
    hidden: Array          # (hidden_dim,) — estado oculto CfC
    workspace: Array       # (workspace_dim,) — Global Workspace (GWT)


# ──────────────────────────────────────────────
# Estado Físico (corpo no ambiente)
# ──────────────────────────────────────────────
class PhysicsState(NamedTuple):
    """Propriedades físicas de um agente no ambiente."""
    position: Array        # (2,) ou (3,) — posição no mundo
    velocity: Array        # (2,) ou (3,) — velocidade linear
    rotation: Array        # escalar ou (4,) quaternion


# ──────────────────────────────────────────────
# Metadados do Agente (biologia, economia, ciclo de vida)
# ──────────────────────────────────────────────
class AgentMeta(NamedTuple):
    """
    Informações metabólicas e de ciclo de vida.

    Campos adicionais v2: mass, max_energy, reproductive_cooldown, sex
    para suportar metabolismo alométrico e reprodução espécie-específica.
    """
    energy: Array                   # escalar — energia corrente (kJ)
    age: Array                      # escalar — idade em timesteps
    species_id: Array               # escalar — identificador de espécie
    is_active: Array                # escalar bool — slot ativo ou vago
    # ── Novos campos v2 ──
    mass: Array                     # escalar — massa corporal (kg)
    max_energy: Array               # escalar — capacidade energética máxima (kJ)
    reproductive_cooldown: Array    # escalar — steps restantes até poder reproduzir
    sex: Array                      # escalar — 0=fêmea, 1=macho


# ──────────────────────────────────────────────
# Estado completo de um agente
# ──────────────────────────────────────────────
class AgentState(NamedTuple):
    """Estado completo de um único agente (imutável)."""
    brain: BrainState
    physics: PhysicsState
    meta: AgentMeta


# ──────────────────────────────────────────────
# Estado do Ambiente
# ──────────────────────────────────────────────
class EnvState(NamedTuple):
    """
    Estado do ambiente com recursos bioma-específicos.

    v2: Substitui food_grid genérico por biomassa C3/C4 com
    sazonalidade, cobertura vegetal e mapas de odor de predadores.
    """
    # ── Vegetação ──
    biomass_c4: Array       # (grid_h, grid_w) — gramíneas de verão (g/m²)
    biomass_c3: Array       # (grid_h, grid_w) — gramíneas de inverno (g/m²)
    cover_density: Array    # (grid_h, grid_w) — cobertura vegetal [0, 1]
    water_mask: Array       # (grid_h, grid_w) — presença de água (bool)

    # ── Odor de predadores ──
    scent_map: Array        # (grid_h, grid_w) — intensidade de odor predador

    # ── Clima/Sazonalidade ──
    season_phase: Array     # escalar [0, 2π] — fase sazonal
    temperature: Array      # escalar — temperatura atual (°C)

    # ── Contadores ──
    step: Array             # escalar — timestep global

    # ── Retrocompatibilidade ──
    food_grid: Array        # (grid_h, grid_w) — biomassa total disponível


# ──────────────────────────────────────────────
# Estado do Mundo (contém todos os agentes + ambiente)
# ──────────────────────────────────────────────
class WorldState(NamedTuple):
    """Estado global da simulação. Tamanho fixo para compatibilidade XLA."""
    agents: AgentState     # Cada campo tem shape (max_agents, ...)
    env: EnvState
    rng: PRNGKey           # Chave PRNG para reprodutibilidade


# ══════════════════════════════════════════════
# Funções Factory
# ══════════════════════════════════════════════

def create_brain_state(hidden_dim: int, workspace_dim: int = 16) -> BrainState:
    """Cria estado neural zerado para um agente."""
    return BrainState(
        hidden=jnp.zeros(hidden_dim, dtype=DTYPE_COMPUTE),
        workspace=jnp.zeros(workspace_dim, dtype=DTYPE_COMPUTE),
    )


def create_agent_state(
    hidden_dim: int,
    spatial_dim: int = 2,
    workspace_dim: int = 16,
    initial_energy: float = 100.0,
    species_id: int = 0,
    mass: float = 1.0,
    max_energy: float = 200.0,
    sex: int = 0,
) -> AgentState:
    """Cria estado completo de um agente com valores iniciais."""
    return AgentState(
        brain=create_brain_state(hidden_dim, workspace_dim),
        physics=PhysicsState(
            position=jnp.zeros(spatial_dim, dtype=DTYPE_STORE),
            velocity=jnp.zeros(spatial_dim, dtype=DTYPE_STORE),
            rotation=jnp.zeros(1, dtype=DTYPE_STORE),
        ),
        meta=AgentMeta(
            energy=jnp.array(initial_energy, dtype=DTYPE_STORE),
            age=jnp.array(0, dtype=jnp.int32),
            species_id=jnp.array(species_id, dtype=jnp.int32),
            is_active=jnp.array(True, dtype=jnp.bool_),
            mass=jnp.array(mass, dtype=DTYPE_STORE),
            max_energy=jnp.array(max_energy, dtype=DTYPE_STORE),
            reproductive_cooldown=jnp.array(0, dtype=jnp.int32),
            sex=jnp.array(sex, dtype=jnp.int32),
        ),
    )


def create_world_state(
    max_agents: int,
    hidden_dim: int,
    grid_size: tuple[int, int] = (128, 128),
    spatial_dim: int = 2,
    workspace_dim: int = 16,
    rng_seed: int = 42,
    biome_id: int = 1,
) -> WorldState:
    """
    Cria estado do mundo com slots fixos para max_agents.

    Arrays têm tamanho fixo para compilação XLA.
    Slots inativos são marcados com is_active=False.

    v2: Grid 128×128, bioma configurável, campos ambientais estendidos.
    """
    rng = jax.random.PRNGKey(rng_seed)

    # Criar arrays batched (max_agents, ...) — todos inativos por padrão
    agents = AgentState(
        brain=BrainState(
            hidden=jnp.zeros((max_agents, hidden_dim), dtype=DTYPE_COMPUTE),
            workspace=jnp.zeros((max_agents, workspace_dim), dtype=DTYPE_COMPUTE),
        ),
        physics=PhysicsState(
            position=jnp.zeros((max_agents, spatial_dim), dtype=DTYPE_STORE),
            velocity=jnp.zeros((max_agents, spatial_dim), dtype=DTYPE_STORE),
            rotation=jnp.zeros((max_agents, 1), dtype=DTYPE_STORE),
        ),
        meta=AgentMeta(
            energy=jnp.zeros(max_agents, dtype=DTYPE_STORE),
            age=jnp.zeros(max_agents, dtype=jnp.int32),
            species_id=jnp.zeros(max_agents, dtype=jnp.int32),
            is_active=jnp.zeros(max_agents, dtype=jnp.bool_),
            mass=jnp.zeros(max_agents, dtype=DTYPE_STORE),
            max_energy=jnp.zeros(max_agents, dtype=DTYPE_STORE),
            reproductive_cooldown=jnp.zeros(max_agents, dtype=jnp.int32),
            sex=jnp.zeros(max_agents, dtype=jnp.int32),
        ),
    )

    env = EnvState(
        biomass_c4=jnp.zeros(grid_size, dtype=DTYPE_STORE),
        biomass_c3=jnp.zeros(grid_size, dtype=DTYPE_STORE),
        cover_density=jnp.zeros(grid_size, dtype=DTYPE_STORE),
        water_mask=jnp.zeros(grid_size, dtype=jnp.bool_),
        scent_map=jnp.zeros(grid_size, dtype=DTYPE_STORE),
        season_phase=jnp.array(0.0, dtype=DTYPE_STORE),
        temperature=jnp.array(18.0, dtype=DTYPE_STORE),
        step=jnp.array(0, dtype=jnp.int32),
        food_grid=jnp.zeros(grid_size, dtype=DTYPE_STORE),
    )

    return WorldState(agents=agents, env=env, rng=rng)


def activate_agents_multispecies(
    world: WorldState,
    species_arrays: dict,
    world_bounds: float = 500.0,
    rng: PRNGKey | None = None,
) -> WorldState:
    """
    Ativa agentes multi-espécie com parâmetros do bioma.

    Args:
        world: Estado do mundo com slots vazios.
        species_arrays: Output de build_species_arrays() — dict com arrays
            species_id, mass, max_speed, max_energy, etc.
        world_bounds: Limites do mundo (±bounds).
        rng: Chave PRNG para posições aleatórias.

    Returns:
        WorldState com agentes ativados por espécie.
    """
    if rng is None:
        rng = world.rng

    rng_pos, rng_sex, rng_next = jax.random.split(rng, 3)

    sp_ids = species_arrays["species_id"]
    num_agents = int(jnp.sum(sp_ids > 0))
    spatial_dim = world.agents.physics.position.shape[-1]

    # Posições aleatórias dentro dos limites do mundo
    positions = jax.random.uniform(
        rng_pos,
        shape=(num_agents, spatial_dim),
        minval=-world_bounds,
        maxval=world_bounds,
    )

    # Sexo aleatório 50/50
    sexes = jax.random.randint(rng_sex, (num_agents,), 0, 2)

    max_agents = world.agents.meta.energy.shape[0]

    # Atualizar slots
    new_position = world.agents.physics.position.at[:num_agents].set(positions)
    new_species_id = world.agents.meta.species_id.at[:max_agents].set(sp_ids)
    new_mass = world.agents.meta.mass.at[:max_agents].set(species_arrays["mass"])
    new_max_energy = world.agents.meta.max_energy.at[:max_agents].set(
        species_arrays["max_energy"]
    )

    # Energia inicial = 70% da capacidade máxima
    initial_energies = species_arrays["max_energy"] * 0.7
    new_energy = world.agents.meta.energy.at[:max_agents].set(initial_energies)

    # Ativar apenas agentes com species_id > 0
    new_is_active = sp_ids > 0

    # Sexo
    new_sex = world.agents.meta.sex.at[:num_agents].set(sexes)

    new_agents = world.agents._replace(
        physics=world.agents.physics._replace(position=new_position),
        meta=world.agents.meta._replace(
            energy=new_energy,
            species_id=new_species_id,
            mass=new_mass,
            max_energy=new_max_energy,
            is_active=new_is_active,
            sex=new_sex,
        ),
    )

    return world._replace(agents=new_agents, rng=rng_next)


# ── Retrocompatibilidade ──
def activate_agents(
    world: WorldState,
    num_agents: int,
    initial_energy: float = 100.0,
    rng: PRNGKey | None = None,
) -> WorldState:
    """
    Ativa `num_agents` slots no mundo com posições aleatórias.

    Usa mascaramento: marca is_active=True nos primeiros N slots.
    Mantido para retrocompatibilidade com testes existentes.
    """
    if rng is None:
        rng = world.rng

    rng_pos, rng_next = jax.random.split(rng)
    spatial_dim = world.agents.physics.position.shape[-1]

    # Gerar posições aleatórias para os agentes ativados
    positions = jax.random.uniform(
        rng_pos,
        shape=(num_agents, spatial_dim),
        minval=-10.0,
        maxval=10.0,
    )

    # Atualizar apenas os primeiros num_agents slots
    new_position = world.agents.physics.position.at[:num_agents].set(positions)
    new_energy = world.agents.meta.energy.at[:num_agents].set(initial_energy)
    new_is_active = world.agents.meta.is_active.at[:num_agents].set(True)
    new_mass = world.agents.meta.mass.at[:num_agents].set(1.0)
    new_max_energy = world.agents.meta.max_energy.at[:num_agents].set(200.0)

    new_agents = world.agents._replace(
        physics=world.agents.physics._replace(position=new_position),
        meta=world.agents.meta._replace(
            energy=new_energy,
            is_active=new_is_active,
            mass=new_mass,
            max_energy=new_max_energy,
        ),
    )

    return world._replace(agents=new_agents, rng=rng_next)

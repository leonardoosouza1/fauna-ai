"""
Estado do Mundo e dos Agentes — Projeto Fauna.

Dataclasses imutáveis compatíveis com JAX pytrees.
Toda atualização gera um novo estado: f(State_t) -> State_{t+1}.
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
# Estado Físico (corpo no ambiente Brax)
# ──────────────────────────────────────────────
class PhysicsState(NamedTuple):
    """Propriedades físicas de um agente no ambiente."""
    position: Array        # (2,) ou (3,) — posição no mundo
    velocity: Array        # (2,) ou (3,) — velocidade linear
    rotation: Array        # escalar ou (4,) quaternion


# ──────────────────────────────────────────────
# Metadados do Agente (economia, ciclo de vida)
# ──────────────────────────────────────────────
class AgentMeta(NamedTuple):
    """Informações metabólicas e de ciclo de vida."""
    energy: Array          # escalar — energia corrente
    age: Array             # escalar — idade em timesteps
    species_id: Array      # escalar — identificador de espécie
    is_active: Array       # escalar bool — slot ativo ou vago


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
    """Estado do ambiente (recursos, feromônios, etc.)."""
    food_grid: Array       # (grid_h, grid_w) — distribuição de recursos
    step: Array            # escalar — contador de timesteps global


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
        ),
    )


def create_world_state(
    max_agents: int,
    hidden_dim: int,
    grid_size: tuple[int, int] = (64, 64),
    spatial_dim: int = 2,
    workspace_dim: int = 16,
    rng_seed: int = 42,
) -> WorldState:
    """
    Cria estado do mundo com slots fixos para max_agents.

    Arrays têm tamanho fixo para compilação XLA.
    Slots inativos são marcados com is_active=False.
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
        ),
    )

    env = EnvState(
        food_grid=jnp.zeros(grid_size, dtype=DTYPE_STORE),
        step=jnp.array(0, dtype=jnp.int32),
    )

    return WorldState(agents=agents, env=env, rng=rng)


def activate_agents(
    world: WorldState,
    num_agents: int,
    initial_energy: float = 100.0,
    rng: PRNGKey | None = None,
) -> WorldState:
    """
    Ativa `num_agents` slots no mundo com posições aleatórias.

    Usa mascaramento: marca is_active=True nos primeiros N slots.
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

    new_agents = world.agents._replace(
        physics=world.agents.physics._replace(position=new_position),
        meta=world.agents.meta._replace(
            energy=new_energy,
            is_active=new_is_active,
        ),
    )

    return world._replace(agents=new_agents, rng=rng_next)

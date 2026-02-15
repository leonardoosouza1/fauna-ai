"""
Testes unitários para State — Projeto Fauna.

Verifica:
  - Criação de AgentState e WorldState
  - Compatibilidade com JAX pytrees (tree_map, jit)
  - Ativação de agentes e mascaramento
"""

import jax
import jax.numpy as jnp
import pytest

from src.core.state import (
    AgentState,
    BrainState,
    PhysicsState,
    AgentMeta,
    WorldState,
    create_agent_state,
    create_brain_state,
    create_world_state,
    activate_agents,
)


class TestBrainState:
    """Testa criação de BrainState."""

    def test_create(self):
        bs = create_brain_state(hidden_dim=64)
        assert bs.hidden.shape == (64,)
        assert bs.workspace.shape == (16,)

    def test_custom_workspace(self):
        bs = create_brain_state(hidden_dim=32, workspace_dim=8)
        assert bs.hidden.shape == (32,)
        assert bs.workspace.shape == (8,)


class TestAgentState:
    """Testa criação e propriedades de AgentState."""

    def test_create_default(self):
        agent = create_agent_state(hidden_dim=64)
        assert agent.brain.hidden.shape == (64,)
        assert agent.physics.position.shape == (2,)
        assert agent.meta.energy == 100.0
        assert agent.meta.is_active == True

    def test_create_3d(self):
        agent = create_agent_state(hidden_dim=32, spatial_dim=3)
        assert agent.physics.position.shape == (3,)
        assert agent.physics.velocity.shape == (3,)

    def test_is_namedtuple(self):
        agent = create_agent_state(hidden_dim=16)
        assert isinstance(agent, AgentState)
        assert isinstance(agent.brain, BrainState)
        assert isinstance(agent.physics, PhysicsState)
        assert isinstance(agent.meta, AgentMeta)


class TestWorldState:
    """Testa criação e manipulação de WorldState."""

    def test_create_world(self):
        world = create_world_state(max_agents=100, hidden_dim=64)

        # Todas as shapes devem ter max_agents como primeira dimensão
        assert world.agents.brain.hidden.shape == (100, 64)
        assert world.agents.physics.position.shape == (100, 2)
        assert world.agents.meta.energy.shape == (100,)
        assert world.agents.meta.is_active.shape == (100,)

        # Todos inativos inicialmente
        assert not jnp.any(world.agents.meta.is_active)

    def test_env_state(self):
        world = create_world_state(
            max_agents=10, hidden_dim=32, grid_size=(32, 32),
        )
        assert world.env.food_grid.shape == (32, 32)
        assert world.env.step == 0


class TestActivateAgents:
    """Testa ativação de agentes no mundo."""

    def test_activate(self):
        world = create_world_state(max_agents=100, hidden_dim=32)
        world = activate_agents(world, num_agents=10)

        # Primeiros 10 ativos
        assert jnp.sum(world.agents.meta.is_active) == 10
        assert jnp.all(world.agents.meta.is_active[:10])
        assert not jnp.any(world.agents.meta.is_active[10:])

    def test_energy_initialized(self):
        world = create_world_state(max_agents=50, hidden_dim=16)
        world = activate_agents(world, num_agents=5, initial_energy=200.0)

        assert jnp.all(world.agents.meta.energy[:5] == 200.0)

    def test_positions_randomized(self):
        world = create_world_state(max_agents=50, hidden_dim=16)
        world = activate_agents(world, num_agents=5)

        # Posições dos agentes ativos não devem ser todas zero
        positions = world.agents.physics.position[:5]
        assert not jnp.allclose(positions, 0.0)


class TestPytreeCompatibility:
    """Testa que os estados são JAX pytrees válidos."""

    def test_tree_map(self):
        agent = create_agent_state(hidden_dim=16)
        # tree_map deve funcionar sem erros
        doubled = jax.tree.map(lambda x: x * 2, agent)
        assert jnp.allclose(doubled.meta.energy, 200.0)

    def test_jit_compatible(self):
        """WorldState deve ser passável para funções @jax.jit."""
        world = create_world_state(max_agents=10, hidden_dim=16)

        @jax.jit
        def count_active(w):
            return jnp.sum(w.agents.meta.is_active)

        assert count_active(world) == 0

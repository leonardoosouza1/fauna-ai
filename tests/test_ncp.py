"""
Testes unitários para NCP e AgentBrain — Projeto Fauna.

Verifica:
  - Construção de NCPWiring com esparsidade correta
  - Forward pass do NCPCell
  - Forward pass completo do AgentBrain
  - Shapes de saída consistentes
"""

import jax
import jax.numpy as jnp
import pytest

from src.systems.brain.ncp import NCPWiring, NCPCell, build_ncp_wiring
from src.systems.brain.policy import AgentBrain


class TestNCPWiring:
    """Testa construção da topologia NCP."""

    def test_build_default(self):
        wiring = build_ncp_wiring()

        assert wiring.num_sensory == 32
        assert wiring.num_inter == 48
        assert wiring.num_command == 16
        assert wiring.num_motor == 8

    def test_mask_shapes(self):
        wiring = build_ncp_wiring(
            num_sensory=16, num_inter=24, num_command=8, num_motor=4,
        )

        assert wiring.sensory_to_inter.shape == (16, 24)
        assert wiring.inter_to_inter.shape == (24, 24)
        assert wiring.inter_to_command.shape == (24, 8)
        assert wiring.command_to_command.shape == (8, 8)
        assert wiring.command_to_motor.shape == (8, 4)

    def test_sparsity(self):
        """Máscaras devem ter ~50% de conexões com sparsity=0.5."""
        wiring = build_ncp_wiring(
            num_sensory=100, num_inter=100,
            num_command=50, num_motor=20,
            sparsity=0.5,
        )
        # Verificar que a proporção de 1s está em torno de 0.5 (±0.1)
        density = jnp.mean(wiring.sensory_to_inter)
        assert 0.3 < float(density) < 0.7

    def test_no_self_connections(self):
        """Camadas recorrentes não devem ter auto-conexões."""
        wiring = build_ncp_wiring(num_inter=32, num_command=16)

        # Diagonais devem ser zero
        inter_diag = jnp.diag(wiring.inter_to_inter)
        cmd_diag = jnp.diag(wiring.command_to_command)

        assert jnp.all(inter_diag == 0.0)
        assert jnp.all(cmd_diag == 0.0)

    def test_reproducibility(self):
        """Mesma seed deve gerar mesma topologia."""
        w1 = build_ncp_wiring(rng_seed=123)
        w2 = build_ncp_wiring(rng_seed=123)

        assert jnp.allclose(w1.sensory_to_inter, w2.sensory_to_inter)
        assert jnp.allclose(w1.inter_to_inter, w2.inter_to_inter)


class TestNCPCell:
    """Testa forward pass do NCPCell."""

    @pytest.fixture
    def ncp_setup(self):
        wiring = build_ncp_wiring(
            num_sensory=16, num_inter=24,
            num_command=8, num_motor=4,
        )
        cell = NCPCell(
            num_inter=24, num_command=8, num_motor=4, wiring=wiring,
        )
        rng = jax.random.PRNGKey(42)

        h_inter = jnp.zeros((1, 24))
        h_cmd = jnp.zeros((1, 8))
        sensory = jnp.ones((1, 16))
        dt = jnp.ones((1, 1)) * 0.02

        params = cell.init(rng, h_inter, h_cmd, sensory, dt)
        return cell, params, wiring

    def test_output_shapes(self, ncp_setup):
        cell, params, _ = ncp_setup
        h_inter = jnp.zeros((1, 24))
        h_cmd = jnp.zeros((1, 8))
        sensory = jnp.ones((1, 16))
        dt = jnp.ones((1, 1)) * 0.02

        new_inter, new_cmd, motor = cell.apply(
            params, h_inter, h_cmd, sensory, dt,
        )

        assert new_inter.shape == (1, 24)
        assert new_cmd.shape == (1, 8)
        assert motor.shape == (1, 4)

    def test_motor_bounded(self, ncp_setup):
        """Saída motora deve estar em [-1, 1] (tanh)."""
        cell, params, _ = ncp_setup
        h_inter = jax.random.normal(jax.random.PRNGKey(0), (1, 24))
        h_cmd = jax.random.normal(jax.random.PRNGKey(1), (1, 8))
        sensory = jax.random.normal(jax.random.PRNGKey(2), (1, 16)) * 10

        _, _, motor = cell.apply(params, h_inter, h_cmd, sensory, jnp.array([[1.0]]))

        assert jnp.all(motor >= -1.0)
        assert jnp.all(motor <= 1.0)

    def test_no_nan(self, ncp_setup):
        cell, params, _ = ncp_setup
        h_inter = jax.random.normal(jax.random.PRNGKey(0), (4, 24))
        h_cmd = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
        sensory = jax.random.normal(jax.random.PRNGKey(2), (4, 16))
        dt = jnp.ones((4, 1)) * 0.02

        new_inter, new_cmd, motor = cell.apply(
            params, h_inter, h_cmd, sensory, dt,
        )

        assert not jnp.any(jnp.isnan(new_inter))
        assert not jnp.any(jnp.isnan(new_cmd))
        assert not jnp.any(jnp.isnan(motor))


class TestAgentBrain:
    """Testa o pipeline cognitivo completo."""

    @pytest.fixture
    def brain_setup(self):
        brain = AgentBrain(
            obs_dim=20, action_dim=6,
            num_sensory=16, num_inter=24, num_command=8,
        )
        rng = jax.random.PRNGKey(42)

        h_inter = jnp.zeros((1, 24))
        h_cmd = jnp.zeros((1, 8))
        obs = jnp.ones((1, 20))
        dt = jnp.ones((1, 1)) * 0.02

        params = brain.init(rng, h_inter, h_cmd, obs, dt)
        return brain, params

    def test_forward_pass(self, brain_setup):
        brain, params = brain_setup
        h_inter = jnp.zeros((1, 24))
        h_cmd = jnp.zeros((1, 8))
        obs = jnp.ones((1, 20))
        dt = jnp.ones((1, 1)) * 0.02

        new_inter, new_cmd, actions = brain.apply(
            params, h_inter, h_cmd, obs, dt,
        )

        assert new_inter.shape == (1, 24)
        assert new_cmd.shape == (1, 8)
        assert actions.shape == (1, 6)

    def test_actions_bounded(self, brain_setup):
        brain, params = brain_setup
        h_inter = jax.random.normal(jax.random.PRNGKey(0), (1, 24))
        h_cmd = jax.random.normal(jax.random.PRNGKey(1), (1, 8))
        obs = jax.random.normal(jax.random.PRNGKey(2), (1, 20)) * 5.0
        dt = jnp.ones((1, 1))

        _, _, actions = brain.apply(params, h_inter, h_cmd, obs, dt)

        assert jnp.all(actions >= -1.0)
        assert jnp.all(actions <= 1.0)

    def test_initialize_hidden(self, brain_setup):
        brain, _ = brain_setup
        h_inter, h_cmd = brain.initialize_hidden(batch_size=32)

        assert h_inter.shape == (32, 24)
        assert h_cmd.shape == (32, 8)

    def test_jit_compatible(self, brain_setup):
        """AgentBrain deve ser compilável com @jax.jit."""
        brain, params = brain_setup

        @jax.jit
        def step(p, h_i, h_c, obs, dt):
            return brain.apply(p, h_i, h_c, obs, dt)

        h_inter = jnp.zeros((1, 24))
        h_cmd = jnp.zeros((1, 8))
        obs = jnp.ones((1, 20))
        dt = jnp.ones((1, 1)) * 0.02

        new_inter, new_cmd, actions = step(params, h_inter, h_cmd, obs, dt)

        assert actions.shape == (1, 6)
        assert not jnp.any(jnp.isnan(actions))

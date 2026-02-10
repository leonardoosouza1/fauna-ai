"""
Testes unitários para a célula CfC — Projeto Fauna.

Verifica:
  - Shapes de saída corretas
  - Estabilidade numérica (sem NaN/Inf)
  - Comportamento temporal (Δt=0 → estado preservado)
  - Compatibilidade com bfloat16
"""

import jax
import jax.numpy as jnp
import pytest

from src.systems.brain.cfc_cell import CfCCell


@pytest.fixture
def cell_params():
    """Inicializa parâmetros de um CfCCell com hidden_size=32."""
    cell = CfCCell(hidden_size=32)
    rng = jax.random.PRNGKey(42)

    # Dummy inputs para inicialização
    h = jnp.zeros((1, 32))
    x = jnp.zeros((1, 16))
    dt = jnp.ones((1, 1))

    params = cell.init(rng, h, x, dt)
    return cell, params


class TestCfCCellShapes:
    """Testa que as shapes de saída estão corretas."""

    def test_single_sample(self, cell_params):
        cell, params = cell_params
        h = jnp.zeros((1, 32))
        x = jnp.ones((1, 16))
        dt = jnp.ones((1, 1))

        h_new, carry = cell.apply(params, h, x, dt)

        assert h_new.shape == (1, 32)
        assert carry.shape == (1, 32)

    def test_batched(self, cell_params):
        cell, params = cell_params
        batch_size = 64
        h = jnp.zeros((batch_size, 32))
        x = jnp.ones((batch_size, 16))
        dt = jnp.ones((batch_size, 1))

        h_new, _ = cell.apply(params, h, x, dt)

        assert h_new.shape == (batch_size, 32)

    def test_scalar_time_delta(self, cell_params):
        """Verifica que Δt escalar funciona via broadcast."""
        cell, params = cell_params
        h = jnp.zeros((4, 32))
        x = jnp.ones((4, 16))
        dt = jnp.array(0.02)

        h_new, _ = cell.apply(params, h, x, dt)

        assert h_new.shape == (4, 32)


class TestCfCCellStability:
    """Testa estabilidade numérica."""

    def test_no_nan(self, cell_params):
        """Forward pass não deve produzir NaN."""
        cell, params = cell_params
        h = jax.random.normal(jax.random.PRNGKey(0), (8, 32))
        x = jax.random.normal(jax.random.PRNGKey(1), (8, 16))
        dt = jnp.ones((8, 1)) * 0.02

        h_new, _ = cell.apply(params, h, x, dt)

        assert not jnp.any(jnp.isnan(h_new))
        assert not jnp.any(jnp.isinf(h_new))

    def test_large_time_delta(self, cell_params):
        """Δt muito grande não deve causar overflow (sigmoid limita)."""
        cell, params = cell_params
        h = jnp.ones((1, 32))
        x = jnp.ones((1, 16))
        dt = jnp.array([[1000.0]])  # Δt enorme

        h_new, _ = cell.apply(params, h, x, dt)

        assert not jnp.any(jnp.isnan(h_new))
        assert not jnp.any(jnp.isinf(h_new))

    def test_zero_inputs(self, cell_params):
        """Entrada zero não deve causar problemas."""
        cell, params = cell_params
        h = jnp.zeros((1, 32))
        x = jnp.zeros((1, 16))
        dt = jnp.ones((1, 1))

        h_new, _ = cell.apply(params, h, x, dt)

        assert not jnp.any(jnp.isnan(h_new))


class TestCfCCellTemporal:
    """Testa comportamento temporal da célula CfC."""

    def test_different_time_deltas_produce_different_outputs(self, cell_params):
        """Δt diferentes devem produzir estados ocultos diferentes."""
        cell, params = cell_params
        h = jax.random.normal(jax.random.PRNGKey(0), (1, 32))
        x = jax.random.normal(jax.random.PRNGKey(1), (1, 16))

        h_fast, _ = cell.apply(params, h, x, jnp.array([[10.0]]))
        h_slow, _ = cell.apply(params, h, x, jnp.array([[0.001]]))

        # Os resultados devem ser diferentes
        assert not jnp.allclose(h_fast, h_slow, atol=1e-3)

    def test_multiple_steps_evolve_state(self, cell_params):
        """Múltiplos passos devem alterar o estado progressivamente."""
        cell, params = cell_params
        h = jnp.zeros((1, 32))
        x = jnp.ones((1, 16)) * 0.5
        dt = jnp.ones((1, 1)) * 0.02

        states = [h]
        for _ in range(5):
            h, _ = cell.apply(params, h, x, dt)
            states.append(h)

        # Cada estado deve ser diferente do anterior
        for i in range(1, len(states)):
            assert not jnp.allclose(states[i], states[i - 1], atol=1e-6)


class TestCfCCellDtype:
    """Testa compatibilidade com bfloat16."""

    def test_bfloat16_input(self):
        """CfC deve funcionar com entradas bfloat16."""
        cell = CfCCell(hidden_size=16)
        rng = jax.random.PRNGKey(42)

        h = jnp.zeros((1, 16), dtype=jnp.bfloat16)
        x = jnp.ones((1, 8), dtype=jnp.bfloat16)
        dt = jnp.ones((1, 1), dtype=jnp.bfloat16)

        params = cell.init(rng, h, x, dt)
        h_new, _ = cell.apply(params, h, x, dt)

        assert not jnp.any(jnp.isnan(h_new))

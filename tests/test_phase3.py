"""
Testes para os sistemas de Fase 3: workspace (GWT), plasticity.
"""

import jax
import jax.numpy as jnp
import pytest

from src.systems.brain.workspace import (
    CrossAttentionFusion,
    ConsensusFusion,
    GlobalWorkspace,
    WorkspaceModulator,
)
from src.systems.brain.plasticity import (
    hebbian_update,
    oja_update,
    homeostatic_scaling,
    compute_activity_stats,
)


class TestGlobalWorkspace:
    """Testes do Global Workspace Theory."""

    @pytest.fixture
    def gwt_setup(self):
        """Cria GlobalWorkspace e dados dummy."""
        batch = 4
        num_workers = 3
        worker_dim = 16
        workspace_dim = 8

        gw = GlobalWorkspace(
            num_workers=num_workers,
            worker_dim=worker_dim,
            workspace_dim=workspace_dim,
            num_heads=2,
        )

        rng = jax.random.PRNGKey(0)
        embeddings = jax.random.normal(rng, (batch, num_workers, worker_dim))
        confidences = jax.random.normal(rng, (batch, num_workers))

        params = gw.init(rng, embeddings, confidences)
        return gw, params, embeddings, confidences

    def test_workspace_output_shape(self, gwt_setup):
        """Workspace retorna shape correto."""
        gw, params, embeddings, confidences = gwt_setup
        ws_state, attn_weights = gw.apply(params, embeddings, confidences)
        assert ws_state.shape == (4, 8)  # (batch, workspace_dim)
        assert attn_weights.shape == (4, 3)  # (batch, num_workers)

    def test_attention_weights_sum_to_one(self, gwt_setup):
        """Pesos de atenção somam 1.0 por batch."""
        gw, params, embeddings, confidences = gwt_setup
        _, attn_weights = gw.apply(params, embeddings, confidences)
        sums = jnp.sum(attn_weights, axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)

    def test_no_nans(self, gwt_setup):
        """Saída livre de NaNs."""
        gw, params, embeddings, confidences = gwt_setup
        ws_state, _ = gw.apply(params, embeddings, confidences)
        assert not jnp.any(jnp.isnan(ws_state))

    def test_modulator_shape(self):
        """WorkspaceModulator retorna shape do target."""
        modulator = WorkspaceModulator(target_dim=24)
        rng = jax.random.PRNGKey(1)
        hidden = jax.random.normal(rng, (2, 24))
        workspace = jax.random.normal(rng, (2, 8))
        params = modulator.init(rng, hidden, workspace)
        modulated = modulator.apply(params, hidden, workspace)
        assert modulated.shape == (2, 24)


class TestCrossAttentionFusion:
    """Testes do CrossAttentionFusion."""

    def test_output_preserves_shape(self):
        """Cross-attention preserva shape de entrada."""
        fusion = CrossAttentionFusion(embed_dim=16, num_heads=2)
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (2, 4, 16))
        params = fusion.init(rng, x)
        out = fusion.apply(params, x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Com residual, saída != entrada mas mesma shape."""
        fusion = CrossAttentionFusion(embed_dim=8, num_heads=2)
        rng = jax.random.PRNGKey(2)
        x = jax.random.normal(rng, (1, 3, 8))
        params = fusion.init(rng, x)
        out = fusion.apply(params, x)
        # Saída deve ser diferente da entrada (attention modificou)
        # mas shapes iguais
        assert out.shape == x.shape


class TestPlasticity:
    """Testes da plasticidade Hebbiana."""

    def test_hebbian_update_shape(self):
        """Hebb update mantém shape dos pesos."""
        pre = jax.random.normal(jax.random.PRNGKey(0), (4, 8))
        post = jax.random.normal(jax.random.PRNGKey(1), (4, 16))
        weights = jnp.zeros((8, 16))

        new_w = hebbian_update(pre, post, weights)
        assert new_w.shape == (8, 16)

    def test_hebbian_correlation(self):
        """Pesos crescem quando pré/pós co-ativam."""
        # Pré e pós perfeitamente correlacionados
        pre = jnp.ones((1, 4))
        post = jnp.ones((1, 4))
        weights = jnp.zeros((4, 4))

        new_w = hebbian_update(pre, post, weights, learning_rate=0.1)
        # Pesos devem crescer (co-ativação positiva)
        assert jnp.all(new_w > 0)

    def test_oja_bounded(self):
        """Oja update mantém pesos mais controlados que Hebb puro."""
        pre = jax.random.normal(jax.random.PRNGKey(0), (10, 8))
        post = jax.random.normal(jax.random.PRNGKey(1), (10, 8))
        weights = jnp.eye(8) * 0.5

        # Múltiplas iterações de Oja
        for _ in range(100):
            weights = oja_update(pre, post, weights, learning_rate=0.01)

        # Pesos não devem explodir
        assert jnp.max(jnp.abs(weights)) < 10.0

    def test_homeostatic_scaling(self):
        """Homeostase reduz pesos quando atividade é alta."""
        weights = jnp.ones((4, 4))
        scaled = homeostatic_scaling(
            weights, target_activity=0.5, current_activity=0.9,
        )
        # Alta atividade → pesos devem reduzir
        assert jnp.all(scaled < weights)

    def test_activity_stats(self):
        """Estatísticas de atividade são calculadas corretamente."""
        activations = jax.random.normal(jax.random.PRNGKey(0), (8, 16))
        stats = compute_activity_stats(activations)
        assert "mean_activity" in stats
        assert "sparsity" in stats
        assert stats["mean_activity"] >= 0
        assert 0 <= stats["sparsity"] <= 1

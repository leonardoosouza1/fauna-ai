"""
Testes para os sistemas de Fase 4: evolution, training, checkpoint.
"""

import os
import tempfile

import jax
import jax.numpy as jnp
import pytest

from src.core.genome import Genome, create_genome, mutate_genome, crossover_genomes
from src.systems.evolution import (
    compute_fitness,
    tournament_selection,
    elite_selection,
    genome_distance,
    evolve_generation,
    log_generation_stats,
)
from src.systems.training import (
    compute_gae,
    ppo_loss,
    create_optimizer,
    compute_reward,
)
from src.adapters.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    list_checkpoints,
    get_latest_checkpoint,
)


class TestEvolution:
    """Testes do sistema de evolução."""

    def test_fitness_computation(self):
        """Fitness é calculado corretamente."""
        energy = jnp.array([100.0, 50.0, 0.0, 200.0])
        age = jnp.array([1000, 500, 100, 5000])
        active = jnp.array([True, True, False, True])

        fitness = compute_fitness(energy, age, active)
        assert fitness.shape == (4,)
        assert fitness[2] == 0.0  # Inativo tem fitness 0
        assert fitness[3] > fitness[1]  # Mais energia+idade = mais fitness

    def test_elite_selection(self):
        """Elitismo seleciona os melhores."""
        fitness = jnp.array([0.1, 0.9, 0.5, 0.8, 0.3])
        elites = elite_selection(fitness, num_elites=2)
        # Top 2: índices 1 (0.9) e 3 (0.8)
        assert 1 in elites
        assert 3 in elites

    def test_tournament_selection(self):
        """Tournament selection retorna índices válidos."""
        rng = jax.random.PRNGKey(0)
        fitness = jnp.array([0.1, 0.5, 0.9, 0.3])
        selected, _ = tournament_selection(rng, fitness, tournament_size=3, num_selections=2)
        assert selected.shape == (2,)
        assert jnp.all(selected >= 0)
        assert jnp.all(selected < 4)

    def test_genome_distance_self(self):
        """Distância de um genoma consigo mesmo é zero."""
        g = create_genome()
        dist = genome_distance(g, g)
        assert jnp.isclose(dist, 0.0, atol=1e-6)

    def test_genome_distance_different(self):
        """Genomas diferentes têm distância > 0."""
        g1 = create_genome(num_inter_neurons=48, sensor_range=10.0)
        g2 = create_genome(num_inter_neurons=24, sensor_range=30.0)
        dist = genome_distance(g1, g2)
        assert dist > 0

    def test_evolve_generation(self):
        """Ciclo completo de evolução produz nova geração."""
        rng = jax.random.PRNGKey(42)
        genomes = [create_genome() for _ in range(10)]
        fitness = jnp.linspace(0.1, 1.0, 10)

        new_genomes, _ = evolve_generation(
            rng, genomes, fitness,
            num_offspring=10, num_elites=2,
        )
        assert len(new_genomes) == 10

    def test_log_stats(self):
        """Estatísticas de geração são calculadas."""
        fitness = jnp.array([0.5, 0.8, 0.3, 0.0])
        stats = log_generation_stats(1, fitness)
        assert stats["generation"] == 1
        assert stats["population"] == 3  # 3 com fitness > 0
        assert stats["max_fitness"] == pytest.approx(0.8, abs=1e-6)


class TestTraining:
    """Testes do loop de treinamento PPO."""

    def test_gae_shapes(self):
        """GAE retorna shapes corretas."""
        T, N = 10, 5
        rewards = jax.random.normal(jax.random.PRNGKey(0), (T, N))
        values = jax.random.normal(jax.random.PRNGKey(1), (T, N))
        dones = jnp.zeros((T, N), dtype=jnp.bool_)

        advantages, returns = compute_gae(rewards, values, dones)
        assert advantages.shape == (T, N)
        assert returns.shape == (T, N)

    def test_gae_no_nans(self):
        """GAE não produz NaNs."""
        T, N = 20, 3
        rewards = jnp.ones((T, N)) * 0.1
        values = jnp.ones((T, N)) * 0.5
        dones = jnp.zeros((T, N))

        advantages, returns = compute_gae(rewards, values, dones)
        assert not jnp.any(jnp.isnan(advantages))
        assert not jnp.any(jnp.isnan(returns))

    def test_ppo_loss_components(self):
        """PPO loss retorna todas as métricas."""
        T, N = 5, 3
        log_probs_new = jnp.zeros((T, N))
        log_probs_old = jnp.zeros((T, N))
        advantages = jax.random.normal(jax.random.PRNGKey(0), (T, N))
        values_pred = jnp.ones((T, N))
        returns = jnp.ones((T, N)) * 1.5

        loss, metrics = ppo_loss(
            log_probs_new, log_probs_old,
            advantages, values_pred, returns,
        )
        assert not jnp.isnan(loss)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "mean_ratio" in metrics

    def test_optimizer_creation(self):
        """Optimizer é criado sem erros."""
        opt = create_optimizer(learning_rate=1e-3)
        assert opt is not None

    def test_reward_computation(self):
        """Recompensas são calculadas corretamente."""
        energy_before = jnp.array([100.0, 50.0, 10.0])
        energy_after = jnp.array([90.0, 60.0, 0.0])
        active = jnp.array([True, True, True])
        food = jnp.array([False, True, False])

        rewards = compute_reward(energy_before, energy_after, active, food)
        assert rewards.shape == (3,)
        # Agente 2 morreu → penalidade negativa forte
        assert rewards[2] < 0
        # Agente 1 comeu → recompensa positiva
        assert rewards[1] > rewards[0]


class TestCheckpoint:
    """Testes do adaptador de checkpoint."""

    def test_save_load_roundtrip(self):
        """Save + load preserva os dados."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = {
                "a": jnp.array([1.0, 2.0, 3.0]),
                "b": jnp.array(42),
            }
            save_checkpoint(state, tmpdir, step=0)
            loaded = load_checkpoint(tmpdir, step=0, template_state=state)

            assert jnp.allclose(loaded["a"], state["a"])
            assert jnp.allclose(loaded["b"], state["b"])

    def test_list_checkpoints(self):
        """Listagem de checkpoints funciona."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = {"x": jnp.ones(5)}
            save_checkpoint(state, tmpdir, step=0)
            save_checkpoint(state, tmpdir, step=100)
            save_checkpoint(state, tmpdir, step=50)

            steps = list_checkpoints(tmpdir)
            assert steps == [0, 50, 100]

    def test_latest_checkpoint(self):
        """Retorna o step mais recente."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = {"x": jnp.ones(3)}
            save_checkpoint(state, tmpdir, step=0)
            save_checkpoint(state, tmpdir, step=200)

            latest = get_latest_checkpoint(tmpdir)
            assert latest == 200

    def test_no_checkpoints(self):
        """Retorna None quando não há checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            latest = get_latest_checkpoint(tmpdir)
            assert latest is None

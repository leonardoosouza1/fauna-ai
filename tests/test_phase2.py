"""
Testes para os sistemas de Fase 2: physics, sensors, metabolism.
"""

import jax
import jax.numpy as jnp
import pytest

from src.systems.physics import (
    apply_forces,
    compute_distances,
    find_nearest_neighbors,
    detect_collisions,
    MAX_SPEED,
    WORLD_BOUNDS,
)
from src.systems.sensors import (
    sense_proprioception,
    sense_food_gradient,
    compute_obs_dim,
)
from src.systems.metabolism import (
    compute_energy_costs,
    update_metabolism,
    respawn_food,
    check_reproduction_eligibility,
    REPRODUCTION_THRESHOLD,
)


class TestPhysics:
    """Testes do motor de física."""

    def test_apply_forces_shapes(self):
        """Posição/velocidade mantêm shapes corretos."""
        max_agents = 20
        pos = jnp.zeros((max_agents, 2))
        vel = jnp.zeros((max_agents, 2))
        acc = jnp.ones((max_agents, 2)) * 0.5
        active = jnp.ones(max_agents, dtype=jnp.bool_)

        new_pos, new_vel = apply_forces(pos, vel, acc, active)
        assert new_pos.shape == (max_agents, 2)
        assert new_vel.shape == (max_agents, 2)

    def test_inactive_agents_stay_still(self):
        """Agentes inativos não se movem."""
        pos = jnp.ones((5, 2)) * 10.0
        vel = jnp.ones((5, 2)) * 3.0
        acc = jnp.ones((5, 2))
        active = jnp.zeros(5, dtype=jnp.bool_)

        new_pos, new_vel = apply_forces(pos, vel, acc, active)
        assert jnp.allclose(new_pos, 0.0)
        assert jnp.allclose(new_vel, 0.0)

    def test_velocity_clamped(self):
        """Velocidade não excede MAX_SPEED."""
        pos = jnp.zeros((3, 2))
        vel = jnp.ones((3, 2)) * 100.0  # Velocidade absurda
        acc = jnp.ones((3, 2)) * 100.0
        active = jnp.ones(3, dtype=jnp.bool_)

        _, new_vel = apply_forces(pos, vel, acc, active)
        speeds = jnp.linalg.norm(new_vel, axis=-1)
        assert jnp.all(speeds <= MAX_SPEED + 1e-5)

    def test_world_bounds(self):
        """Posição fica dentro dos limites."""
        pos = jnp.ones((3, 2)) * 49.9
        vel = jnp.ones((3, 2)) * MAX_SPEED
        acc = jnp.ones((3, 2))
        active = jnp.ones(3, dtype=jnp.bool_)

        new_pos, _ = apply_forces(pos, vel, acc, active)
        assert jnp.all(new_pos >= -WORLD_BOUNDS)
        assert jnp.all(new_pos <= WORLD_BOUNDS)

    def test_distances_symmetric(self):
        """Matriz de distâncias é simétrica."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (10, 2))
        active = jnp.ones(10, dtype=jnp.bool_)
        dists = compute_distances(positions, active)
        # Simétrica exceto diagonal (inf) — filtrar inf antes de comparar
        mask = ~jnp.eye(10, dtype=jnp.bool_)
        finite_mask = jnp.isfinite(dists) & mask
        assert jnp.allclose(dists[finite_mask], dists.T[finite_mask], atol=1e-6)

    def test_nearest_neighbors(self):
        """Vizinhos mais próximos retornam shapes corretas."""
        positions = jax.random.normal(jax.random.PRNGKey(1), (20, 2))
        active = jnp.ones(20, dtype=jnp.bool_)
        dists = compute_distances(positions, active)
        indices, neigh_dists = find_nearest_neighbors(dists, k=5)
        assert indices.shape == (20, 5)
        assert neigh_dists.shape == (20, 5)

    def test_collision_detection(self):
        """Detecta colisões quando agentes estão próximos."""
        # Criar posições com dois agentes muito próximos e um distante
        positions = jnp.array([
            [0.0, 0.0],
            [0.3, 0.0],
            [10.0, 10.0],
            [20.0, 20.0],
            [30.0, 30.0],
        ])
        active = jnp.ones(5, dtype=jnp.bool_)
        dists = compute_distances(positions, active)
        collisions = detect_collisions(dists, collision_radius=1.0)
        # Agentes 0 e 1 colidem (dist=0.3 < 1.0)
        assert collisions[0, 1]
        assert collisions[1, 0]
        # Agente 2 está longe de 0
        assert not collisions[0, 2]


class TestSensors:
    """Testes do sistema sensorial."""

    def test_proprioception_shape(self):
        """Propriocepção retorna vetor correto."""
        vel = jnp.array([1.0, 2.0])
        energy = jnp.array(100.0)
        age = jnp.array(500.0)

        proprio = sense_proprioception(vel, energy, age)
        # vel(2) + speed(1) + energy(1) + age(1) = 5
        assert proprio.shape == (5,)
        assert not jnp.any(jnp.isnan(proprio))

    def test_food_gradient_shape(self):
        """Gradiente de comida retorna 5 valores."""
        pos = jnp.array([0.0, 0.0])
        food = jnp.ones((64, 64))
        grad = sense_food_gradient(pos, food)
        assert grad.shape == (5,)

    def test_obs_dim_calculation(self):
        """Dimensão de observação é calculada corretamente."""
        dim = compute_obs_dim(spatial_dim=2, num_neighbors=5)
        # proprio: 2+3=5, neighbors: 5*(2+1)=15, food: 5 → total: 25
        assert dim == 25


class TestMetabolism:
    """Testes do sistema metabólico."""

    def test_energy_costs_positive(self):
        """Custos de energia são sempre positivos para agentes ativos."""
        vel = jnp.ones((10, 2)) * 2.0
        active = jnp.ones(10, dtype=jnp.bool_)
        costs = compute_energy_costs(vel, active)
        assert jnp.all(costs > 0)

    def test_inactive_no_cost(self):
        """Agentes inativos não consomem energia."""
        vel = jnp.ones((5, 2))
        active = jnp.zeros(5, dtype=jnp.bool_)
        costs = compute_energy_costs(vel, active)
        assert jnp.allclose(costs, 0.0)

    def test_death_by_starvation(self):
        """Agentes sem energia morrem."""
        energy = jnp.array([0.0, 50.0, 0.0, 100.0, 0.0])
        age = jnp.zeros(5, dtype=jnp.int32)
        vel = jnp.zeros((5, 2))
        active = jnp.ones(5, dtype=jnp.bool_)
        gained = jnp.zeros(5)

        _, _, new_active = update_metabolism(energy, age, vel, active, gained)
        # Agentes com energia 0 devem morrer
        assert not new_active[0]
        assert new_active[1]  # Sobrevive
        assert not new_active[2]
        assert new_active[3]  # Sobrevive
        assert not new_active[4]

    def test_food_respawn(self):
        """Comida é regenerada com taxa correta."""
        food = jnp.zeros((32, 32))
        rng = jax.random.PRNGKey(42)
        new_food, _ = respawn_food(food, rng, respawn_rate=1.0)
        # Com rate=1.0, toda célula deveria ganhar comida
        assert jnp.all(new_food >= 1.0)

    def test_reproduction_threshold(self):
        """Apenas agentes com energia suficiente podem reproduzir."""
        energy = jnp.array([50.0, REPRODUCTION_THRESHOLD, 200.0, 0.0])
        active = jnp.ones(4, dtype=jnp.bool_)
        can_repro = check_reproduction_eligibility(energy, active)
        assert not can_repro[0]  # Abaixo do threshold
        assert can_repro[1]     # No threshold
        assert can_repro[2]     # Acima do threshold
        assert not can_repro[3]  # Sem energia

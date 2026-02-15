"""
Estrutura do Genoma — Projeto Fauna.

O genoma codifica os hiperparâmetros estruturais e de plasticidade
de cada agente. Na Fase 4 (Evolução), estes valores serão evoluídos
via algoritmos genéticos / HyperNEAT.

O genoma NÃO contém os pesos da rede neural diretamente —
ele descreve a TOPOLOGIA e os meta-parâmetros que geram a rede.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from src.core.types import Array, PRNGKey, DTYPE_STORE


class Genome(NamedTuple):
    """
    Genoma de um agente Fauna.

    Codifica a "receita" para construir o cérebro e corpo do agente.
    Valores escalares são representados como arrays 0-D para
    compatibilidade com pytrees JAX.
    """
    # --- Topologia Neural ---
    hidden_size: Array          # Dimensão do estado oculto CfC
    num_inter_neurons: Array    # Neurônios na camada intermediária (NCP)
    num_command_neurons: Array  # Neurônios na camada de comando (NCP)

    # --- Meta-parâmetros de Plasticidade (Fase 3) ---
    plasticity_rate: Array      # η (eta) — taxa de aprendizado Hebbiano
    plasticity_decay: Array     # Fator de decaimento Oja

    # --- Parâmetros Sensoriais ---
    sensor_range: Array         # Alcance máximo dos sensores
    sensor_fov: Array           # Campo de visão (radianos)

    # --- Fitness (preenchido durante simulação) ---
    fitness: Array              # Fitness acumulado do indivíduo


def create_genome(
    hidden_size: int = 64,
    num_inter_neurons: int = 48,
    num_command_neurons: int = 16,
    plasticity_rate: float = 0.01,
    plasticity_decay: float = 0.001,
    sensor_range: float = 10.0,
    sensor_fov: float = 2.0,
) -> Genome:
    """Cria um genoma com valores padrão."""
    return Genome(
        hidden_size=jnp.array(hidden_size, dtype=jnp.int32),
        num_inter_neurons=jnp.array(num_inter_neurons, dtype=jnp.int32),
        num_command_neurons=jnp.array(num_command_neurons, dtype=jnp.int32),
        plasticity_rate=jnp.array(plasticity_rate, dtype=DTYPE_STORE),
        plasticity_decay=jnp.array(plasticity_decay, dtype=DTYPE_STORE),
        sensor_range=jnp.array(sensor_range, dtype=DTYPE_STORE),
        sensor_fov=jnp.array(sensor_fov, dtype=DTYPE_STORE),
        fitness=jnp.array(0.0, dtype=DTYPE_STORE),
    )


def mutate_genome(genome: Genome, rng: PRNGKey, mutation_rate: float = 0.1) -> Genome:
    """
    Aplica mutação gaussiana nos parâmetros contínuos do genoma.

    Parâmetros inteiros (hidden_size, num_neurons) são perturbados
    com probabilidade mutation_rate e incrementados/decrementados.
    """
    keys = jax.random.split(rng, 7)

    # Mutação de parâmetros contínuos (perturbação gaussiana)
    new_plasticity_rate = genome.plasticity_rate + (
        jax.random.normal(keys[0]) * mutation_rate * genome.plasticity_rate
    )
    new_plasticity_decay = genome.plasticity_decay + (
        jax.random.normal(keys[1]) * mutation_rate * genome.plasticity_decay
    )
    new_sensor_range = genome.sensor_range + (
        jax.random.normal(keys[2]) * mutation_rate * genome.sensor_range
    )
    new_sensor_fov = genome.sensor_fov + (
        jax.random.normal(keys[3]) * mutation_rate * 0.1
    )

    # Clamp para valores válidos
    new_plasticity_rate = jnp.clip(new_plasticity_rate, 1e-5, 0.1)
    new_plasticity_decay = jnp.clip(new_plasticity_decay, 1e-6, 0.01)
    new_sensor_range = jnp.clip(new_sensor_range, 1.0, 50.0)
    new_sensor_fov = jnp.clip(new_sensor_fov, 0.5, jnp.pi)

    # Mutação de parâmetros inteiros (±1 com probabilidade mutation_rate)
    inter_delta = jax.random.choice(
        keys[4], jnp.array([-1, 0, 1]),
        p=jnp.array([mutation_rate / 2, 1.0 - mutation_rate, mutation_rate / 2]),
    )
    command_delta = jax.random.choice(
        keys[5], jnp.array([-1, 0, 1]),
        p=jnp.array([mutation_rate / 2, 1.0 - mutation_rate, mutation_rate / 2]),
    )
    hidden_delta = jax.random.choice(
        keys[6], jnp.array([-4, 0, 4]),
        p=jnp.array([mutation_rate / 2, 1.0 - mutation_rate, mutation_rate / 2]),
    )

    new_inter = jnp.clip(genome.num_inter_neurons + inter_delta, 8, 128).astype(jnp.int32)
    new_command = jnp.clip(genome.num_command_neurons + command_delta, 4, 64).astype(jnp.int32)
    new_hidden = jnp.clip(genome.hidden_size + hidden_delta, 16, 256).astype(jnp.int32)

    return genome._replace(
        hidden_size=new_hidden,
        num_inter_neurons=new_inter,
        num_command_neurons=new_command,
        plasticity_rate=new_plasticity_rate,
        plasticity_decay=new_plasticity_decay,
        sensor_range=new_sensor_range,
        sensor_fov=new_sensor_fov,
        fitness=jnp.array(0.0, dtype=DTYPE_STORE),  # Reset fitness
    )


def crossover_genomes(
    parent_a: Genome,
    parent_b: Genome,
    rng: PRNGKey,
) -> Genome:
    """
    Crossover uniforme entre dois genomas.

    Cada gene é herdado aleatoriamente de um dos pais.
    """
    keys = jax.random.split(rng, 7)

    def _pick(key, gene_a, gene_b):
        """Seleciona gene de parent_a ou parent_b com 50% de chance."""
        return jax.lax.cond(
            jax.random.uniform(key) < 0.5,
            lambda: gene_a,
            lambda: gene_b,
        )

    return Genome(
        hidden_size=_pick(keys[0], parent_a.hidden_size, parent_b.hidden_size),
        num_inter_neurons=_pick(keys[1], parent_a.num_inter_neurons, parent_b.num_inter_neurons),
        num_command_neurons=_pick(keys[2], parent_a.num_command_neurons, parent_b.num_command_neurons),
        plasticity_rate=_pick(keys[3], parent_a.plasticity_rate, parent_b.plasticity_rate),
        plasticity_decay=_pick(keys[4], parent_a.plasticity_decay, parent_b.plasticity_decay),
        sensor_range=_pick(keys[5], parent_a.sensor_range, parent_b.sensor_range),
        sensor_fov=_pick(keys[6], parent_a.sensor_fov, parent_b.sensor_fov),
        fitness=jnp.array(0.0, dtype=DTYPE_STORE),
    )

"""
Sistema de Evolução — Projeto Fauna.

Implementa neuroevolução para otimizar topologias e meta-parâmetros
dos agentes ao longo de gerações.

Operadores:
  1. Seleção: Tournament selection baseada em fitness (tempo de vida + energia)
  2. Crossover: Troca uniforme de genes entre dois genomas
  3. Mutação: Perturbação Gaussiana com taxa adaptativa
  4. Especiação: Agrupamento por similaridade genômica (δ-threshold)

A evolução complementa a plasticidade:
  - Plasticidade adapta DENTRO da vida (fenótipo)
  - Evolução adapta ENTRE vidas (genótipo)
  - Baldwin Effect: aprendizados frequentes são assimilados geneticamente

Referência: NEAT/HyperNEAT (Stanley & Miikkulainen, 2002)
"""

import jax
import jax.numpy as jnp
from functools import partial

from src.core.types import Array, PRNGKey, DTYPE_STORE
from src.core.genome import Genome, create_genome, mutate_genome, crossover_genomes


# ──────────────────────────────────────────────
# Fitness
# ──────────────────────────────────────────────

def compute_fitness(
    energy: Array,
    age: Array,
    is_active: Array,
    food_consumed: Array | None = None,
    offspring_count: Array | None = None,
) -> Array:
    """
    Calcula o fitness de cada agente.

    Fitness = ponderação de:
      - Tempo de vida (age) — seleção por sobrevivência
      - Energia acumulada — seleção por eficiência
      - Comida consumida — seleção por capacidade de busca
      - Prole gerada — seleção direta de reprodução

    Args:
        energy: (max_agents,)
        age: (max_agents,)
        is_active: (max_agents,)
        food_consumed: (max_agents,) — opcional
        offspring_count: (max_agents,) — opcional

    Returns:
        fitness: (max_agents,) — score de fitness (0 para inativos)
    """
    # Normalizar componentes
    age_score = age.astype(DTYPE_STORE) / 10000.0  # Normalizar pela idade máxima
    energy_score = energy / 200.0  # Normalizar pela energia máxima

    fitness = age_score * 0.4 + energy_score * 0.3

    if food_consumed is not None:
        food_score = jnp.clip(food_consumed / 100.0, 0.0, 1.0)
        fitness = fitness + food_score * 0.2

    if offspring_count is not None:
        repro_score = jnp.clip(offspring_count.astype(DTYPE_STORE) / 5.0, 0.0, 1.0)
        fitness = fitness + repro_score * 0.1
    else:
        # Redistribuir peso se não tiver dados de reprodução
        fitness = fitness / 0.7  # Normalizar pela soma dos pesos usados

    # Mascarar inativos
    fitness = fitness * is_active.astype(DTYPE_STORE)

    return fitness


# ──────────────────────────────────────────────
# Seleção
# ──────────────────────────────────────────────

def tournament_selection(
    rng: PRNGKey,
    fitness: Array,
    tournament_size: int = 3,
    num_selections: int = 1,
) -> tuple[Array, PRNGKey]:
    """
    Seleção por torneio: escolhe o melhor entre k candidatos aleatórios.

    Args:
        rng: Chave PRNG
        fitness: (population_size,)
        tournament_size: Número de candidatos por torneio
        num_selections: Quantas seleções realizar

    Returns:
        selected_indices: (num_selections,) — índices dos vencedores
        new_rng: Chave PRNG atualizada
    """
    population_size = fitness.shape[0]

    def _single_tournament(rng):
        # Escolher k candidatos aleatórios
        rng, key = jax.random.split(rng)
        candidates = jax.random.randint(
            key, (tournament_size,), 0, population_size,
        )
        # Fitness dos candidatos
        candidate_fitness = fitness[candidates]
        # Retornar o melhor
        winner_idx = candidates[jnp.argmax(candidate_fitness)]
        return winner_idx, rng

    # Selecionar múltiplos vencedores
    indices = []
    for _ in range(num_selections):
        winner, rng = _single_tournament(rng)
        indices.append(winner)

    selected = jnp.array(indices, dtype=jnp.int32)
    return selected, rng


def elite_selection(
    fitness: Array,
    num_elites: int = 5,
) -> Array:
    """
    Preserva os N melhores indivíduos (elitismo).

    Args:
        fitness: (population_size,)
        num_elites: Número de elites a preservar

    Returns:
        elite_indices: (num_elites,)
    """
    return jnp.argsort(fitness)[-num_elites:]


# ──────────────────────────────────────────────
# Especiação
# ──────────────────────────────────────────────

def genome_distance(g1: Genome, g2: Genome) -> float:
    """
    Calcula distância genômica entre dois genomas.

    Combina diferenças em topologia + meta-parâmetros.

    Args:
        g1, g2: Genomas a comparar

    Returns:
        distance: Escalar de dissimilaridade
    """
    # Diferenças em topologia (discretas)
    topo_diff = (
        jnp.abs(g1.num_inter_neurons - g2.num_inter_neurons) / 48.0
        + jnp.abs(g1.num_command_neurons - g2.num_command_neurons) / 16.0
    )

    # Diferenças em meta-parâmetros (contínuas)
    meta_diff = (
        jnp.abs(g1.plasticity_rate - g2.plasticity_rate)
        + jnp.abs(g1.sensor_range - g2.sensor_range) / 20.0
        + jnp.abs(g1.plasticity_decay - g2.plasticity_decay)
    )

    return topo_diff * 0.5 + meta_diff * 0.5


def assign_species(
    genomes: list[Genome],
    species_representatives: list[Genome],
    threshold: float = 0.3,
) -> list[int]:
    """
    Atribui espécie a cada genoma baseado na distância ao representante.

    Se nenhuma espécie existente está próxima o suficiente (< threshold),
    cria uma nova espécie.

    Args:
        genomes: Lista de genomas da população
        species_representatives: Representante de cada espécie
        threshold: Distância máxima para pertencer a uma espécie

    Returns:
        species_ids: Lista de IDs de espécie para cada genoma
    """
    species_ids = []
    next_species_id = len(species_representatives)

    for genome in genomes:
        assigned = False
        for sp_id, rep in enumerate(species_representatives):
            dist = genome_distance(genome, rep)
            if dist < threshold:
                species_ids.append(sp_id)
                assigned = True
                break

        if not assigned:
            species_ids.append(next_species_id)
            species_representatives.append(genome)
            next_species_id += 1

    return species_ids


# ──────────────────────────────────────────────
# Pipeline de Evolução
# ──────────────────────────────────────────────

def evolve_generation(
    rng: PRNGKey,
    genomes: list[Genome],
    fitness_scores: Array,
    num_offspring: int,
    num_elites: int = 5,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.7,
    tournament_size: int = 3,
) -> tuple[list[Genome], PRNGKey]:
    """
    Executa uma geração completa de evolução.

    Pipeline:
      1. Elitismo: preservar os melhores genomas
      2. Seleção: tournament selection dos pais
      3. Crossover: recombinar genes dos pais
      4. Mutação: perturbar os filhos

    Args:
        rng: Chave PRNG
        genomes: Lista de genomas da geração atual
        fitness_scores: (population_size,) — fitness de cada genoma
        num_offspring: Número de filhos a gerar
        num_elites: Número de elites a preservar
        mutation_rate: Taxa de mutação global
        crossover_rate: Probabilidade de crossover vs mutação pura
        tournament_size: Tamanho do torneio de seleção

    Returns:
        new_genomes: Lista de genomas da nova geração
        new_rng: Chave PRNG atualizada
    """
    population_size = len(genomes)
    new_genomes = []

    # 1. Elitismo — preservar os melhores
    elite_idx = elite_selection(fitness_scores, num_elites)
    for idx in elite_idx:
        new_genomes.append(genomes[int(idx)])

    # 2. Gerar filhos
    remaining = num_offspring - num_elites
    for _ in range(max(0, remaining)):
        rng, key1, key2, key3, key_cx = jax.random.split(rng, 5)

        # Selecionar dois pais
        parent_indices, rng = tournament_selection(
            key1, fitness_scores, tournament_size, num_selections=2,
        )
        parent1 = genomes[int(parent_indices[0])]
        parent2 = genomes[int(parent_indices[1])]

        # Crossover ou clone
        do_crossover = jax.random.uniform(key_cx) < crossover_rate
        if do_crossover:
            child = crossover_genomes(parent1, parent2, key2)
        else:
            child = parent1

        # Mutação
        child = mutate_genome(child, key3, mutation_rate)
        new_genomes.append(child)

    return new_genomes, rng


def log_generation_stats(
    generation: int,
    fitness_scores: Array,
    num_species: int = 0,
) -> dict:
    """
    Calcula e retorna estatísticas de uma geração para monitoramento.

    Args:
        generation: Número da geração
        fitness_scores: (population_size,) — fitness da geração
        num_species: Número de espécies

    Returns:
        Dict com estatísticas da geração
    """
    active_fitness = fitness_scores[fitness_scores > 0]

    if active_fitness.shape[0] == 0:
        return {
            "generation": generation,
            "population": 0,
            "mean_fitness": 0.0,
            "max_fitness": 0.0,
            "min_fitness": 0.0,
            "std_fitness": 0.0,
            "num_species": num_species,
        }

    return {
        "generation": generation,
        "population": int(active_fitness.shape[0]),
        "mean_fitness": float(jnp.mean(active_fitness)),
        "max_fitness": float(jnp.max(active_fitness)),
        "min_fitness": float(jnp.min(active_fitness)),
        "std_fitness": float(jnp.std(active_fitness)),
        "num_species": num_species,
    }

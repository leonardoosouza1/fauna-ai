"""
Projeto Fauna — Main Simulation Loop (v2).

Loop principal que integra TODOS os subsistemas:
  1. Observação sensorial (sensors.py)
  2. Inferência neural CfC/NCP (brain/)
  3. Atuação física (physics.py)
  4. Metabolismo e energia (metabolism.py)
  5. Evolução neuro (evolution.py)
  6. Recompensas e treinamento (training.py)
  7. Dashboard web (visualization/)

Uso:
    python -m src.main
    python -m src.main --num-agents 50 --num-steps 5000 --dashboard
"""

import argparse
import logging
import time

import jax
import jax.numpy as jnp

from src.core.state import (
    create_world_state,
    activate_agents,
    WorldState,
)
from src.core.types import DTYPE_STORE, DTYPE_COMPUTE
from src.core.genome import create_genome, Genome
from src.systems.brain.policy import AgentBrain
from src.systems.brain.ncp import build_ncp_wiring
from src.systems.physics import apply_forces
from src.systems.sensors import build_all_observations, compute_obs_dim
from src.systems.metabolism import (
    update_metabolism,
    consume_food,
    respawn_food,
    check_reproduction_eligibility,
)
from src.systems.evolution import (
    compute_fitness,
    evolve_generation,
    log_generation_stats,
)
from src.systems.training import compute_reward
from src.visualization.exporter import SimulationExporter
from src.infrastructure.rocm_check import check_backend, get_device_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fauna")


# ──────────────────────────────────────────────
# Configuração
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Projeto Fauna — Simulação")
    # Simulação
    parser.add_argument("--num-agents", type=int, default=20, help="Número inicial de agentes")
    parser.add_argument("--max-agents", type=int, default=100, help="Capacidade máxima do mundo")
    parser.add_argument("--num-steps", type=int, default=2000, help="Steps de simulação")
    parser.add_argument("--dt", type=float, default=0.02, help="Passo temporal")
    parser.add_argument("--grid-size", type=int, default=64, help="Tamanho do grid de comida")
    parser.add_argument("--seed", type=int, default=42, help="Seed PRNG")
    parser.add_argument("--log-interval", type=int, default=100, help="Intervalo de log")
    # Evolução
    parser.add_argument("--generation-length", type=int, default=500, help="Steps por geração")
    parser.add_argument("--num-elites", type=int, default=3, help="Elites preservados por geração")
    parser.add_argument("--mutation-rate", type=float, default=0.15, help="Taxa de mutação")
    # Dashboard
    parser.add_argument("--dashboard", action="store_true", help="Ativar dashboard web")
    parser.add_argument("--dashboard-port", type=int, default=8765, help="Porta do dashboard")
    parser.add_argument("--export-interval", type=int, default=10, help="Intervalo de export para dashboard")
    # Checkpoint
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Diretório de checkpoints")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Inicialização
# ──────────────────────────────────────────────

def create_brain(
    obs_dim: int,
    action_dim: int,
    num_sensory: int = 16,
    num_inter: int = 24,
    num_command: int = 8,
    sparsity: float = 0.5,
    rng_seed: int = 0,
) -> tuple[AgentBrain, dict]:
    """Cria e inicializa o AgentBrain com parâmetros."""
    wiring = build_ncp_wiring(
        num_sensory=num_sensory,
        num_inter=num_inter,
        num_command=num_command,
        num_motor=action_dim,
        sparsity=sparsity,
        rng_seed=rng_seed,
    )

    brain = AgentBrain(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_sensory=num_sensory,
        num_inter=num_inter,
        num_command=num_command,
        wiring=wiring,
    )

    rng = jax.random.PRNGKey(rng_seed)
    h_inter, h_command = brain.initialize_hidden(batch_size=1)
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_dt = jnp.ones((1, 1)) * 0.02

    params = brain.init(rng, h_inter, h_command, dummy_obs, dummy_dt)

    return brain, params


def initialize_food(
    rng: jax.Array,
    grid_size: int = 64,
    initial_density: float = 0.3,
    max_food: float = 3.0,
) -> jax.Array:
    """Inicializa o grid de comida com distribuição aleatória."""
    rng, key = jax.random.split(rng)
    food_mask = jax.random.uniform(key, (grid_size, grid_size)) < initial_density
    food_values = food_mask.astype(DTYPE_STORE) * max_food
    return food_values


def create_initial_population(num_agents: int, rng_seed: int = 42) -> list[Genome]:
    """Cria população inicial de genomas."""
    genomes = []
    for i in range(num_agents):
        g = create_genome(
            num_inter_neurons=24,
            num_command_neurons=8,
            plasticity_rate=0.01,
            sensor_range=15.0,
            rng_seed=rng_seed + i,
        )
        genomes.append(g)
    return genomes


# ──────────────────────────────────────────────
# Simulation Step
# ──────────────────────────────────────────────

def simulation_step(
    world: WorldState,
    brain: AgentBrain,
    params: dict,
    h_inter: jax.Array,
    h_command: jax.Array,
    dt: float,
    num_neighbors: int = 5,
    sensor_range: float = 15.0,
) -> tuple[WorldState, jax.Array, jax.Array, dict]:
    """
    Executa um step completo da simulação.

    Pipeline:
        1. Sensores → Observações
        2. Brain → Ações
        3. Física → Posições/Velocidades
        4. Metabolismo → Energia/Vida
        5. Comida → Respawn

    Returns:
        new_world, new_h_inter, new_h_command, metrics
    """
    agents = world.agents
    positions = agents.physics.position
    velocities = agents.physics.velocity
    energies = agents.meta.energy
    ages = agents.meta.age
    is_active = agents.meta.is_active
    food_grid = world.env.food_grid
    rng = world.rng

    # 1. Observações sensoriais
    observations = build_all_observations(
        positions, velocities, energies, ages,
        is_active, food_grid, num_neighbors, sensor_range,
    )

    # 2. Inferência neural (batched)
    time_delta = jnp.ones((positions.shape[0], 1)) * dt

    new_h_inter, new_h_command, actions = brain.apply(
        params, h_inter, h_command, observations, time_delta,
    )

    # Separar ações: primeiras 2 dimensões = aceleração
    acceleration = actions[:, :2]

    # 3. Física
    new_positions, new_velocities = apply_forces(
        positions, velocities, acceleration, is_active, dt,
    )

    # 4. Alimentação
    energy_gained, new_food_grid = consume_food(
        new_positions, food_grid, is_active,
    )

    # 5. Metabolismo
    new_energies, new_ages, new_is_active = update_metabolism(
        energies, ages, new_velocities, is_active, energy_gained,
    )

    # 6. Respawn de comida
    rng, food_rng = jax.random.split(rng)
    new_food_grid, _ = respawn_food(new_food_grid, food_rng)

    # 7. Recompensas
    food_eaten = energy_gained > 0
    rewards = compute_reward(energies, new_energies, is_active, food_eaten)

    # Reconstruir WorldState
    new_world = world._replace(
        agents=agents._replace(
            physics=agents.physics._replace(
                position=new_positions,
                velocity=new_velocities,
            ),
            meta=agents.meta._replace(
                energy=new_energies,
                age=new_ages,
                is_active=new_is_active,
            ),
        ),
        env=world.env._replace(
            food_grid=new_food_grid,
            step=world.env.step + 1,
        ),
        rng=rng,
    )

    # Métricas
    num_alive = jnp.sum(new_is_active)
    can_reproduce = check_reproduction_eligibility(new_energies, new_is_active)
    metrics = {
        "alive": int(num_alive),
        "mean_energy": float(jnp.where(num_alive > 0, jnp.sum(new_energies * new_is_active) / jnp.maximum(num_alive, 1), 0.0)),
        "mean_age": float(jnp.where(num_alive > 0, jnp.sum(new_ages * new_is_active) / jnp.maximum(num_alive, 1), 0.0)),
        "food_total": float(jnp.sum(new_food_grid)),
        "food_consumed": int(jnp.sum(food_eaten)),
        "mean_reward": float(jnp.mean(rewards * is_active)),
        "can_reproduce": int(jnp.sum(can_reproduce)),
    }

    return new_world, new_h_inter, new_h_command, metrics


# ──────────────────────────────────────────────
# Evolução
# ──────────────────────────────────────────────

def run_evolution(
    rng: jax.Array,
    genomes: list[Genome],
    energies: jax.Array,
    ages: jax.Array,
    is_active: jax.Array,
    num_offspring: int,
    num_elites: int = 3,
    mutation_rate: float = 0.15,
) -> tuple[list[Genome], dict, jax.Array]:
    """
    Executa um ciclo de evolução ao final de uma geração.

    Returns:
        new_genomes, generation_stats, rng
    """
    # Calcular fitness
    fitness = compute_fitness(energies, ages, is_active)
    generation_number = len(genomes)

    # Log stats
    stats = log_generation_stats(generation_number, fitness)
    logger.info(
        f"🧬 Geração {stats['generation']} — "
        f"Pop: {stats['population']}, "
        f"Fitness max/mean: {stats['max_fitness']:.3f}/{stats['mean_fitness']:.3f}"
    )

    # Evoluir
    new_genomes, rng = evolve_generation(
        rng, genomes, fitness,
        num_offspring=num_offspring,
        num_elites=num_elites,
        mutation_rate=mutation_rate,
    )

    return new_genomes, stats, rng


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run_simulation():
    """Executa a simulação completa com evolução e dashboard."""
    args = parse_args()

    # ── Diagnóstico ──
    backend = check_backend()
    logger.info(f"JAX backend: {backend}")
    for d in get_device_info():
        logger.info(f"  Device: {d['platform']} — {d['device_kind']}")

    # ── Configuração ──
    spatial_dim = 2
    num_neighbors = 5
    sensor_range = 15.0
    obs_dim = compute_obs_dim(spatial_dim, num_neighbors)
    action_dim = 6  # 2 (aceleração XY) + 4 (interação futura)

    logger.info(f"Obs dim: {obs_dim}, Action dim: {action_dim}")
    logger.info(f"Agentes: {args.num_agents}/{args.max_agents}, Steps: {args.num_steps}")

    # ── Dashboard ──
    exporter = SimulationExporter()
    if args.dashboard:
        from src.visualization.server import start_dashboard_server
        start_dashboard_server(port=args.dashboard_port)
        logger.info(f"🌐 Dashboard: http://localhost:{args.dashboard_port}")

    # ── Criar mundo ──
    world = create_world_state(
        max_agents=args.max_agents,
        hidden_dim=24,
        grid_size=(args.grid_size, args.grid_size),
        rng_seed=args.seed,
    )

    # Inicializar comida
    rng_food = jax.random.PRNGKey(args.seed + 1)
    food = initialize_food(rng_food, args.grid_size)
    world = world._replace(
        env=world.env._replace(food_grid=food),
    )

    # Ativar agentes iniciais
    world = activate_agents(world, num_agents=args.num_agents, initial_energy=100.0)
    logger.info(f"Agentes ativados: {int(jnp.sum(world.agents.meta.is_active))}")

    # ── População de genomas ──
    genomes = create_initial_population(args.num_agents, rng_seed=args.seed)
    generation = 0
    generation_step_counter = 0

    # ── Criar cérebro ──
    logger.info("Inicializando AgentBrain (CfC/NCP)...")
    brain, params = create_brain(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_sensory=16,
        num_inter=24,
        num_command=8,
    )

    # Estado oculto para todos os agentes
    h_inter = jnp.zeros((args.max_agents, 24), dtype=DTYPE_COMPUTE)
    h_command = jnp.zeros((args.max_agents, 8), dtype=DTYPE_COMPUTE)

    # RNG para evolução
    evo_rng = jax.random.PRNGKey(args.seed + 100)

    # ── Simulation Loop ──
    logger.info(f"Iniciando simulação ({args.num_steps} steps, {args.generation_length} steps/geração)...")
    start_time = time.perf_counter()
    all_gen_stats = []

    for step in range(args.num_steps):
        # ── Step da simulação ──
        world, h_inter, h_command, metrics = simulation_step(
            world, brain, params, h_inter, h_command,
            dt=args.dt, num_neighbors=num_neighbors, sensor_range=sensor_range,
        )

        generation_step_counter += 1

        # ── Export para dashboard ──
        if step % args.export_interval == 0:
            gen_stats_for_export = all_gen_stats[-1] if all_gen_stats else None
            exporter.export_step(
                step=step,
                positions=world.agents.physics.position,
                energies=world.agents.meta.energy,
                ages=world.agents.meta.age,
                is_active=world.agents.meta.is_active,
                food_grid=world.env.food_grid,
                metrics=metrics,
                generation=generation,
                generation_stats=gen_stats_for_export,
            )

        # ── Log periódico ──
        if (step + 1) % args.log_interval == 0:
            elapsed = time.perf_counter() - start_time
            sps = (step + 1) / elapsed
            logger.info(
                f"Step {step + 1:>6d}/{args.num_steps} | "
                f"Gen: {generation:>2d} | "
                f"Alive: {metrics['alive']:>3d} | "
                f"Energy: {metrics['mean_energy']:>6.1f} | "
                f"Age: {metrics['mean_age']:>6.0f} | "
                f"Food: {metrics['food_total']:>6.0f} | "
                f"Reward: {metrics['mean_reward']:>+.3f} | "
                f"SPS: {sps:.1f}"
            )

        # ── Evolução: fim de geração ──
        if generation_step_counter >= args.generation_length and metrics["alive"] > 0:
            new_genomes, gen_stats, evo_rng = run_evolution(
                rng=evo_rng,
                genomes=genomes,
                energies=world.agents.meta.energy,
                ages=world.agents.meta.age,
                is_active=world.agents.meta.is_active,
                num_offspring=args.num_agents,
                num_elites=args.num_elites,
                mutation_rate=args.mutation_rate,
            )
            genomes = new_genomes
            generation += 1
            generation_step_counter = 0
            gen_stats["generation"] = generation
            all_gen_stats.append(gen_stats)

            # Re-ativar agentes com energia renovada para a nova geração
            world = activate_agents(world, num_agents=args.num_agents, initial_energy=100.0)
            h_inter = jnp.zeros((args.max_agents, 24), dtype=DTYPE_COMPUTE)
            h_command = jnp.zeros((args.max_agents, 8), dtype=DTYPE_COMPUTE)

        # ── Extinção total ──
        if metrics["alive"] == 0:
            logger.warning(f"⚠️ EXTINÇÃO no step {step + 1}! Re-populando...")
            # Re-popular com genomas mutados
            evo_rng, repop_rng = jax.random.split(evo_rng)
            genomes = create_initial_population(args.num_agents, rng_seed=int(repop_rng[0]))
            world = activate_agents(world, num_agents=args.num_agents, initial_energy=150.0)
            h_inter = jnp.zeros((args.max_agents, 24), dtype=DTYPE_COMPUTE)
            h_command = jnp.zeros((args.max_agents, 8), dtype=DTYPE_COMPUTE)
            generation += 1
            generation_step_counter = 0

            # Regenerar comida após extinção
            rng_food = jax.random.PRNGKey(args.seed + step)
            food = initialize_food(rng_food, args.grid_size, initial_density=0.5)
            world = world._replace(
                env=world.env._replace(food_grid=food),
            )

    # ── Finalização ──
    elapsed = time.perf_counter() - start_time
    exporter.export_final(step=args.num_steps, status="completed")

    logger.info("=" * 60)
    logger.info(f"Simulação concluída em {elapsed:.2f}s ({args.num_steps / elapsed:.1f} SPS)")
    logger.info(f"Gerações completadas: {generation}")

    alive = int(jnp.sum(world.agents.meta.is_active))
    logger.info(f"Agentes sobreviventes: {alive}/{args.num_agents}")
    if alive > 0:
        active_energies = world.agents.meta.energy * world.agents.meta.is_active
        active_ages = world.agents.meta.age * world.agents.meta.is_active
        logger.info(f"Energia média final: {float(jnp.sum(active_energies) / alive):.1f}")
        logger.info(f"Idade média final: {float(jnp.sum(active_ages) / alive):.0f} steps")

    if all_gen_stats:
        best = max(all_gen_stats, key=lambda s: s.get("max_fitness", 0))
        logger.info(f"Melhor fitness: {best.get('max_fitness', 0):.4f} (geração {best.get('generation', '?')})")

    logger.info("=" * 60)

    if args.dashboard:
        logger.info(f"Dashboard ainda ativo em http://localhost:{args.dashboard_port}")
        logger.info("Pressione Ctrl+C para encerrar.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Encerrando.")


if __name__ == "__main__":
    run_simulation()

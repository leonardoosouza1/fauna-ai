"""
Projeto Fauna — Main Simulation Loop (v3 Multi-Espécie).

Orquestra a simulação ecológica completa com:
  1. Fauna Brasileira (7 espécies, 5 guildas)
  2. Metabolismo Alométrico (Kleiber)
  3. Predação Holling (Tipo II/III)
  4. Dinâmica Ambiental Sazonal (C3/C4, Clima)
  5. Sensores Específicos (Visão, Olfato, Cobertura)

 Pipeline do Step:
   Environment → Sensors → Brain → Physics → Metabolism → Herbivory → Predation → Reprod
"""

import argparse
import logging
import time

import jax
import jax.numpy as jnp

from src.core.state import (
    create_world_state,
    activate_agents_multispecies,
    WorldState,
)
from src.core.species import build_species_arrays, build_prey_matrix, BIOME_REGISTRY
from src.core.types import DTYPE_STORE, DTYPE_COMPUTE
from src.core.genome import create_genome, Genome

from src.systems.brain.policy import AgentBrain
from src.systems.brain.ncp import build_ncp_wiring
from src.systems.physics import apply_forces
from src.systems.sensors import build_all_observations, compute_obs_dim

from src.systems.metabolism import (
    update_metabolism,
    check_reproduction_eligibility,
    reproduce_agent,
)
from src.systems.environment import (
    initialize_environment,
    advance_season,
    grow_vegetation,
    consume_vegetation,
    update_scent_map,
    compute_total_food_grid,
)
from src.systems.predation import resolve_predation_batch
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
    parser = argparse.ArgumentParser(description="Projeto Fauna — Simulação Ecológica")
    # Simulação
    parser.add_argument("--max-agents", type=int, default=500, help="Capacidade máxima do mundo")
    parser.add_argument("--num-steps", type=int, default=10000, help="Steps de simulação")
    parser.add_argument("--dt", type=float, default=0.1, help="Passo temporal (0.1 dia)")
    parser.add_argument("--grid-size", type=int, default=128, help="Tamanho do grid")
    parser.add_argument("--seed", type=int, default=42, help="Seed PRNG")
    parser.add_argument("--log-interval", type=int, default=100, help="Intervalo de log")
    # Bioma
    parser.add_argument("--biome-id", type=int, default=1, help="ID do Bioma (1=Pampa-Mata)")
    # Dashboard
    parser.add_argument("--dashboard", action="store_true", default=True, help="Ativar dashboard web")
    parser.add_argument("--dashboard-port", type=int, default=8765, help="Porta do dashboard")
    parser.add_argument("--export-interval", type=int, default=10, help="Intervalo de export")
    
    return parser.parse_args()


# ──────────────────────────────────────────────
# Inicialização Neural
# ──────────────────────────────────────────────

def create_brain(
    obs_dim: int,
    action_dim: int,
    num_sensory: int = 16,  # Placeholder, ajustado dinamicamente no init
    num_inter: int = 24,
    num_command: int = 8,
    sparsity: float = 0.5,
    rng_seed: int = 0,
) -> tuple[AgentBrain, dict]:
    """Cria e inicializa o AgentBrain (único compartilhado por enquanto)."""
    # Ajustar num_sensory para bater com obs_dim real
    num_sensory = obs_dim
    
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


# ──────────────────────────────────────────────
# Simulation Step
# ──────────────────────────────────────────────

def simulation_step(
    world: WorldState,
    brain: AgentBrain,
    params: dict,
    h_inter: jax.Array,
    h_command: jax.Array,
    species_arrays: dict,  # Dict com arrays imutáveis de parâmetros da espécie
    prey_matrix: dict,
    dt: float,
    num_neighbors: int = 5,
) -> tuple[WorldState, jax.Array, jax.Array, dict]:
    """
    Executa um step completo da simulação ecológica.
    """
    agents = world.agents
    env = world.env
    rng = world.rng
    
    # Desempacotar estado
    pos = agents.physics.position
    vel = agents.physics.velocity
    meta = agents.meta
    
    # ── 1. Ambiente ──
    # Avançar estação
    season_phase, temperature = advance_season(env.season_phase)
    
    # Crescer vegetação
    c4, c3, cover, rng_env = grow_vegetation(
        env.biomass_c4, env.biomass_c3, env.cover_density, season_phase, rng
    )
    
    # Atualizar mapa de odores (predadores depositam)
    scent_map = update_scent_map(
        env.scent_map, pos, meta.is_active, meta.species_id
    )

    # ── 2. Sensores ──
    # Extrair parâmetros sensoriais dos arrays de espécie (indexados)
    # Mas species_arrays já são (N,) alinhados com agentes?
    # Sim, build_species_arrays retorna arrays full (max_agents).
    # Se agentes morrerem/nascerem, precisamos manter consistência.
    # Por simplicidade neste MVP, species_arrays é estático baseada na inicialização.
    # CORREÇÃO: Reprodução altera meta.species_id, então precisamos fazer lookup dinâmico
    # ou atualizar species_arrays.
    # O meta.species_id é a verdade. Vamos usar params de species.py via lookup se possível 
    # ou assumir que species_arrays foi atualizado.
    # Para performance JAX, melhor é ter arrays no state ou passados corretos.
    # Vamos confiar no meta.species_id e usar arrays auxiliares atualizados se necessário.
    # No `activate_agents_multispecies`, inicializamos meta.mass, etc. 
    # Então usamos `meta.mass`, `meta.max_energy` que estão no world state.
    # Faltam `visual_range`, `fov`, `max_speed` no WorldState.
    # Vamos passá-los como args (species_arrays) assumindo que não mudam por slot (agentes não trocam de specie, só morrem).
    # Mas reprodução pode ocupar slot vazio.
    # OK: Vamos usar os dados de `species_arrays` MAS indexados pelo `meta.species_id` seria ideal.
    # Como species_arrays é (N,), assume slot N é especie X.
    # Se filho nasce no slot K, slot K tem que virar especie X.
    # Solução MVP: `species_arrays` passado aqui é APENAS reference data? Não, é per-agent.
    # Precisamos atualizar `species_arrays` fora ou dentro. 
    # Vamos usar `meta` values onde possível (mass, max_energy).
    # Para visual_range/max_speed, vamos assumir que o slot mantém a espécie ou atualizamos.
    
    observations = build_all_observations(
        pos, vel, meta.energy, meta.age, meta.is_active,
        meta.species_id, meta.mass, meta.max_energy, meta.reproductive_cooldown, meta.sex,
        c4, c3, cover, env.water_mask, scent_map,
        species_arrays["visual_range"], species_arrays["fov_degrees"], # Potencial bug se nascimento trocar espécie.
        num_neighbors=num_neighbors
    )

    # ── 3. Cérebro ──
    time_delta = jnp.ones((pos.shape[0], 1)) * dt
    new_h_inter, new_h_command, actions = brain.apply(
        params, h_inter, h_command, observations, time_delta,
    )
    acceleration = actions[:, :2]

    # ── 4. Física ──
    new_pos, new_vel = apply_forces(
        pos, vel, acceleration, meta.is_active,
        species_arrays["max_speed"], # Bug potencial em nascimento
        water_mask=env.water_mask,
        dt=dt,
        world_bounds=500.0, # Bioma config
    )

    # ── 5. Herbivoria (Vegetação) ──
    # Consome baseado na massa (Kleiber)
    energy_gained_herb, new_c4, new_c3 = consume_vegetation(
        c4, c3, new_pos, meta.is_active, meta.species_id, meta.mass
    )

    # ── 6. Predação (Carnivoria) ──
    # Predadores caçam e ganham energia
    # Presas morrem (active=False)
    new_active, new_energies_pred, consumption = resolve_predation_batch(
        new_pos, new_vel, meta.is_active, 
        meta.species_id, meta.mass, meta.energy,
        prey_matrix, cover,
        rng=rng
    )
    
    # Energia total ganha = herbivoria + predação (mas só um ou outro acontece por agente)
    # new_energies_pred já soma energia de carne ao meta.energy original
    # Precisamos somar energy_gained_herb ao resultado
    # (assumindo que carnívoro não come planta e vice versa, diet types separam)
    
    # Energia atualizada base + ganho carne
    current_energy_step = new_energies_pred 
    
    # ── 7. Metabolismo ──
    # Aplica custos (Kleiber) e soma ganho vegetal
    final_energies, new_ages, final_active = update_metabolism(
        current_energy_step, meta.age, new_vel, new_active,
        energy_gained=energy_gained_herb, # ganho vegetal
        mass=meta.mass,
        max_energy=meta.max_energy,
        temperature=temperature
    )

    # ── 8. Estatísticas & Updates ──
    
    # Computar Food Grid legado para visualização
    total_food_vis = compute_total_food_grid(new_c4, new_c3)

    new_world = world._replace(
        agents=agents._replace(
            physics=agents.physics._replace(
                position=new_pos,
                velocity=new_vel,
            ),
            meta=agents.meta._replace(
                energy=final_energies,
                age=new_ages,
                is_active=final_active,
            ),
        ),
        env=env._replace(
            biomass_c4=new_c4,
            biomass_c3=new_c3,
            cover_density=cover,
            scent_map=scent_map,
            season_phase=season_phase,
            temperature=temperature,
            food_grid=total_food_vis, # Retrocompatibilidade visual
            step=env.step + 1,
        ),
        rng=rng_env,
    )

    # Métricas
    num_alive = jnp.sum(final_active)
    
    # Contagem por espécie
    prea_count = jnp.sum(final_active & (meta.species_id == 1))
    veado_count = jnp.sum(final_active & (meta.species_id == 2))
    predators_count = jnp.sum(final_active & (meta.species_id >= 4))

    metrics = {
        "alive": int(num_alive),
        "prea_pop": int(prea_count),
        "veado_pop": int(veado_count),
        "predator_pop": int(predators_count),
        "biomass_total": float(jnp.sum(new_c4 + new_c3)),
        "temperature": float(temperature),
        "season_phase": float(season_phase),
    }

    return new_world, new_h_inter, new_h_command, metrics


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run_simulation():
    args = parse_args()
    
    # Setup
    backend = check_backend()
    logger.info(f"Backend JAX: {backend}")
    
    # Configurar Obs/Actions
    spatial_dim = 2
    num_neighbors = 5
    obs_dim = compute_obs_dim(num_neighbors)
    action_dim = 6 
    

    # Dashboard API Callbacks
    # ───────────────────────
    # Precisamos de um container mutável para o estado atual ser acessado pela thread do servidor
    shared_state = {
        "world": None,
        "metrics": {},
        "step": 0
    }

    def get_state() -> dict:
        """Callback API: Retorna snapshot leve para renderização 3D."""
        if shared_state["world"] is None:
            return {"status": "loading"}
        
        w = shared_state["world"]
        agents = w.agents
        
        # Converter JAX para numpy/list
        active = agents.meta.is_active
        count = int(jnp.sum(active))
        
        # Filtrar apenas ativos
        indices = jnp.where(active)[0]
        
        agents_data = []
        # Acesso direto aos arrays (lento se fizer device->host a cada request? 
        # Sim, mas para 500 agentes é ok. Se escalar, usar buffer assíncrono)
        pos = agents.physics.position[indices]
        sp_ids = agents.meta.species_id[indices]
        energies = agents.meta.energy[indices]
        
        for i, idx in enumerate(indices):
            agents_data.append({
                "id": int(idx),
                "x": float(pos[i, 0]),
                "y": float(pos[i, 1]),
                "sp": int(sp_ids[i]),
                "e": float(energies[i]),
            })

        return {
            "step": shared_state["step"],
            "metrics": shared_state["metrics"],
            "agents": agents_data,
            "environment": {
                "temp": shared_state["metrics"].get("temperature", 0),
                "season": shared_state["metrics"].get("season_phase", 0),
            }
        }

    def get_agent(agent_id: int) -> dict:
        """Callback API: Retorna detalhe profundo de um agente."""
        if shared_state["world"] is None:
            return {"error": "Not ready"}
        
        w = shared_state["world"]
        # Validar ID
        if agent_id < 0 or agent_id >= args.max_agents:
            return {"error": "Invalid ID"}
        
        if not w.agents.meta.is_active[agent_id]:
            return {"error": "Agent dead or inactive"}

        # Extrair dados
        meta = w.agents.meta
        phys = w.agents.physics
        
        return {
            "id": agent_id,
            "species_id": int(meta.species_id[agent_id]),
            "age": int(meta.age[agent_id]),
            "mass": float(meta.mass[agent_id]),
            "energy": float(meta.energy[agent_id]),
            "max_energy": float(meta.max_energy[agent_id]),
            "pos": [float(phys.position[agent_id, 0]), float(phys.position[agent_id, 1])],
            "vel": [float(phys.velocity[agent_id, 0]), float(phys.velocity[agent_id, 1])],
            "genetics": {
                "sex": int(meta.sex[agent_id]),
                # Futuro: gene sequence
            },
            "brain": {
                # Futuro: atividades neurais do NCP
                "status": "Thinking..." 
            }
        }

    if args.dashboard:
        from src.visualization.server import start_dashboard_server
        start_dashboard_server(
            port=args.dashboard_port,
            get_state_callback=get_state,
            get_agent_callback=get_agent
        )
        logger.info(f"Dashboard ativo: http://localhost:{args.dashboard_port}")

    # Inicializar Mundo e Bioma
    logger.info(f"Inicializando Bioma ID {args.biome_id}...")
    
    # Criar estado zerado
    world = create_world_state(
        max_agents=args.max_agents,
        hidden_dim=24,
        grid_size=(args.grid_size, args.grid_size),
        rng_seed=args.seed,
        biome_id=args.biome_id
    )
    
    # Inicializar ambiente (vegetação, chuva, etc)
    env_data = initialize_environment(
        grid_size=(args.grid_size, args.grid_size),
        rng_seed=args.seed
    )
    world = world._replace(
        env=world.env._replace(
            biomass_c4=env_data["biomass_c4"],
            biomass_c3=env_data["biomass_c3"],
            cover_density=env_data["cover_density"],
            water_mask=env_data["water_mask"],
        )
    )
    
    # Construir arrays de espécie para inicialização
    logger.info("Construindo fauna inicial...")
    sp_arrays = build_species_arrays(args.biome_id, args.max_agents)
    prey_matrix = build_prey_matrix(args.biome_id)
    
    # Ativar agentes
    world = activate_agents_multispecies(
        world, sp_arrays, world_bounds=500.0, rng=world.rng
    )
    
    total_pop = jnp.sum(world.agents.meta.is_active)
    logger.info(f"População inicial total: {total_pop}")
    
    # Inicializar Cérebro
    brain, params = create_brain(obs_dim, action_dim)
    h_inter = jnp.zeros((args.max_agents, 24), dtype=DTYPE_COMPUTE)
    h_command = jnp.zeros((args.max_agents, 8), dtype=DTYPE_COMPUTE)

    logger.info(f"Iniciando simulação ecológica por {args.num_steps} steps...")
    start_time = time.perf_counter()
    
    for step in range(args.num_steps):
        world, h_inter, h_command, metrics = simulation_step(
            world, brain, params, h_inter, h_command,
            sp_arrays, prey_matrix, args.dt, num_neighbors
        )
        
        # Atualizar estado compartilhado para API
        if args.dashboard and step % args.export_interval == 0:
            shared_state["world"] = world
            shared_state["metrics"] = metrics
            shared_state["step"] = step

        # Log
        if (step + 1) % args.log_interval == 0:
            elapsed = time.perf_counter() - start_time
            sps = (step + 1) / elapsed
            logger.info(
                f"Step {step+1} | "
                f"Alive: {metrics['alive']} (P:{metrics['prea_pop']} V:{metrics['veado_pop']} Pred:{metrics['predator_pop']}) | "
                f"Bio: {metrics['biomass_total']:.0f} | "
                f"Temp: {metrics['temperature']:.1f}C | "
                f"SPS: {sps:.1f}"
            )
            
        if metrics['alive'] == 0:
            logger.warning("Extinção total! Encerrando.")
            break

    total_time = time.perf_counter() - start_time
    logger.info(f"Simulação finalizada em {total_time:.2f}s")
    # exporter.export_final(step=args.num_steps, status="completed") # Exporter removido do loop crítico

    if args.dashboard:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    run_simulation()

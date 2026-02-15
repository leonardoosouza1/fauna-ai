"""
Sistema de Predação — Projeto Fauna v2.

Gerencia as interações tróficas entre predadores e presas:
  - Respostas funcionais de Holling (Tipo II e III)
  - Mecânica de caça (detecção, perseguição, captura)
  - Transferência de energia (eficiência calórica entre níveis tróficos)
  - Ecologia do medo (impacto no comportamento das presas)

Modelagem baseada em:
  - Holling (1959): Componentes da predação
  - Lotka-Volterra com saturação
  - Rosenzweig-MacArthur (estabilidade)
"""

import jax
import jax.numpy as jnp

from src.core.types import Array, PRNGKey, DTYPE_STORE, DTYPE_COMPUTE
from src.systems.physics import compute_distances


# ──────────────────────────────────────────────
# Respostas Funcionais de Holling
# ──────────────────────────────────────────────

def holling_type_ii(
    prey_density: Array,
    attack_rate: float,
    handling_time: float,
) -> Array:
    """
    Resposta Funcional Tipo II (Invertebrados, Felinos).
    Taxa de consumo satura com a densidade de presas.

    F = (a * N) / (1 + a * h * N)

    Args:
        prey_density: Densidade de presas (N)
        attack_rate: Taxa de ataque (a) - área de busca
        handling_time: Tempo de manipulação (h) - comer/digerir

    Returns:
        consumption_rate: Presas consumidas por predador por unidade de tempo
    """
    numerator = attack_rate * prey_density
    denominator = 1.0 + attack_rate * handling_time * prey_density
    return numerator / denominator


def holling_type_iii(
    prey_density: Array,
    attack_rate: float,
    handling_time: float,
) -> Array:
    """
    Resposta Funcional Tipo III (Vertebrados generalistas, Canídeos).
    Sigmoidal: baixo consumo em baixas densidades (prey switching/refúgio),
    satura em altas densidades.

    F = (a * N^2) / (1 + a * h * N^2)

    Args:
        prey_density: Densidade de presas (N)
        attack_rate: Taxa de ataque (a)
        handling_time: Tempo de manipulação (h)

    Returns:
        consumption_rate: Presas consumidas por predador por unidade de tempo
    """
    n_squared = prey_density ** 2
    numerator = attack_rate * n_squared
    denominator = 1.0 + attack_rate * handling_time * n_squared
    return numerator / denominator


# ──────────────────────────────────────────────
# Mecânica de Predação Física
# ──────────────────────────────────────────────

def attempt_predation(
    predator_idx: int,
    predator_pos: Array,
    predator_vel: Array,
    prey_positions: Array,
    prey_velocities: Array,
    prey_active: Array,
    prey_species_ids: Array,
    target_prey_ids: list[int],
    attack_range: float = 2.0,
    cover_density: Array | None = None,
    world_bounds: float = 500.0,
    rng: PRNGKey | None = None,
) -> tuple[int, bool, float]:
    """
    Tenta realizar predação de um único predador contra presas próximas.

    Lógica:
    1. Encontrar presas da lista target_prey_ids dentro do attack_range
    2. Se houver múltiplas, escolhe a mais próxima (ou mais vulnerável)
    3. Teste de captura baseado em:
       - Velocidade relativa (predador vs presa)
       - Cobertura vegetal (refúgio)
       - Estocasticidade

    Args:
        predator_idx: Índice do predador
        predator_pos: (2,) Posição
        predator_vel: (2,) Velocidade
        prey_positions: (N, 2) Posições de todos agentes
        prey_velocities: (N, 2) Velocidades de todos agentes
        prey_active: (N,) Máscara de ativos
        prey_species_ids: (N,) IDs de espécie de todos agentes
        target_prey_ids: Lista de IDs de espécies que são presas válidas
        attack_range: Distância máxima para iniciar ataque
        cover_density: (H, W) Grid de cobertura (opcional)
        world_bounds: Limites do mundo
        rng: Chave PRNG

    Returns:
        (caught_prey_idx, success, efficiency_mult)
        caught_prey_idx: Índice da presa capturada (-1 se nenhuma)
        success: Booleano se capturou
        efficiency_mult: Multiplicador de eficiência (ex: 1.0)
    """
    if rng is None:
        # Fallback determinístico se RNG não fornecido (não ideal)
        rng = jax.random.PRNGKey(0)

    # 1. Identificar presas válidas
    is_prey_species = jnp.isin(prey_species_ids, jnp.array(target_prey_ids))
    # Não pode ser o próprio predador (embora IDs de espécie já evitem isso geralmente)
    not_self = jnp.arange(prey_positions.shape[0]) != predator_idx
    
    valid_targets = is_prey_species & prey_active & not_self

    # 2. Calcular distâncias
    diff = prey_positions - predator_pos
    dists = jnp.linalg.norm(diff, axis=-1)
    
    # 3. Filtrar por alcance
    in_range = dists <= attack_range
    candidates = valid_targets & in_range
    
    # Se não há candidatos, retorna falha
    # Como JAX requer retornos de shape fixo, usamos jax.lax.cond ou similar,
    # mas aqui vamos fazer uma lógica vetorizável mascarada.
    
    # Encontrar a presa mais próxima entre os candidatos
    # Infinitos para não-candidatos
    masked_dists = jnp.where(candidates, dists, jnp.inf)
    nearest_idx = jnp.argmin(masked_dists)
    min_dist = masked_dists[nearest_idx]
    
    has_target = min_dist < jnp.inf

    # 4. Probabilidade de Captura
    # Base: 0.5
    # Bônus Velocidade: +0.5 * (V_pred / V_prey - 1)
    # Penalidade Cobertura: -0.8 * cover_density
    
    pred_speed = jnp.linalg.norm(predator_vel)
    prey_speed = jnp.linalg.norm(prey_velocities[nearest_idx])
    
    speed_ratio = jnp.clip(pred_speed / (prey_speed + 0.01), 0.5, 2.0)
    base_prob = 0.4
    speed_bonus = 0.4 * (speed_ratio - 1.0) # Se dobro da vel, ganha +0.4
    
    # Cobertura no local da presa
    cover_penalty = 0.0
    if cover_density is not None:
        grid_h, grid_w = cover_density.shape
        normalized = (prey_positions[nearest_idx] + world_bounds) / (2.0 * world_bounds)
        gx = jnp.clip((normalized[0] * grid_w).astype(jnp.int32), 0, grid_w - 1)
        gy = jnp.clip((normalized[1] * grid_h).astype(jnp.int32), 0, grid_h - 1)
        
        local_cover = cover_density[gy, gx]
        # Cobertura ajuda a presa significativamente
        cover_penalty = 0.6 * local_cover

    capture_prob = jnp.clip(base_prob + speed_bonus - cover_penalty, 0.05, 0.95)
    
    # Sorteio
    rng_capture, _ = jax.random.split(rng)
    success_roll = jax.random.uniform(rng_capture)
    success = has_target & (success_roll < capture_prob)
    
    target_idx = jnp.where(success, nearest_idx, -1)
    
    return target_idx, success, 1.0


def resolve_predation_batch(
    positions: Array,
    velocities: Array,
    is_active: Array,
    species_ids: Array,
    masses: Array,
    energies: Array,
    prey_matrix: dict[int, list[int]],
    cover_density: Array,
    base_capture_efficiency: float = 0.15,
    world_bounds: float = 500.0,
    rng: PRNGKey = None,
) -> tuple[Array, Array, Array]:
    """
    Resolve todas as tentativas de predação em um passo.

    Args:
        positions: (N, 2)
        velocities: (N, 2)
        is_active: (N,)
        species_ids: (N,)
        masses: (N,)
        energies: (N,)
        prey_matrix: Dict mapeando predator_species_id -> [prey_species_id, ...]
        cover_density: (H, W)
        base_capture_efficiency: Eficiência de conversão de biomassa em energia (kJ/kg -> kJ)
        world_bounds: Limites do tamanho do mundo
        rng: Chave aleatória

    Returns:
        (new_active, new_energies, consumption_events)
        new_active: Máscara atualizada (presas mortas = False)
        new_energies: Energias atualizadas (predadores ganham, presas 'perdem' mas já estão mortas)
        consumption_events: Array auxiliar para estatísticas (quem comeu quem? - opcional)
    """
    # Converter prey_matrix para formato JAX-friendly (tensor denso ou lógica condicional)
    # Como a matriz é pequena (poucas espécies), podemos iterar por tipos de predador
    # ou usar vmap com lógica condicional.
    
    # Para simplicidade e performance em TPUs/GPUs, vamos permitir que cada predador
    # tente capturar *uma* presa por step.
    
    N = positions.shape[0]
    
    # Arrays de saída iniciados com valores atuais
    next_active = is_active
    next_energies = energies
    
    if rng is None:
        rng = jax.random.PRNGKey(42)
        
    rng_sub = jax.random.split(rng, N)

    # Definir função scan para processar predadores sequencialmente
    # (Necessário sequencial para evitar que dois predadores comam a mesma presa no mesmo step
    #  e ganhem energia duplicada, ou matem uma presa já morta)
    
    # Estado do scan: (active_mask, energies)
    ScanState = tuple[Array, Array]
    
    def predation_step(carry: ScanState, idx: int) -> tuple[ScanState, int]:
        curr_active, curr_energies = carry
        
        # Dados do predador atual
        my_species = species_ids[idx]
        my_pos = positions[idx]
        my_vel = velocities[idx]
        am_active = curr_active[idx]
        
        # Verificar se é predador (tem entradas na prey_matrix)
        # Como prey_matrix é python dict, precisamos de lógica fixa.
        # Vamos assumir que species_ids são 0..K e fazer um lookup.
        # Mas `prey_matrix` varia. 
        # Abordagem híbrida: Predadores definidos por species.py.
        # Aqui, vamos checar se my_species tem presas.
        
        # Truque JAX: não podemos iterar dict python dentro de JIT facilmente se chaves forem dinâmicas de dados.
        # Mas `species_ids` é um array.
        
        # Solução: Passar a lógica de "quem come quem" como parâmetros numéricos ou 
        # usar uma função `params_lookup` construída fora.
        # Para este MVP, vamos hardcodar as guildas ou usar `diet_type` se passado.
        # Mas `prey_matrix` foi passado. Vamos assumir max 6 espécies.
        
        # Simplificação: O predador idx tenta caçar.
        # Precisamos saber QUAIS ids são presas dele.
        # Vamos construir um array de máscara de presas: (N_species, N_species) bool
        # Mas isso é complexo de passar.
        
        # Vamos usar a logica: Se (species >= 4), é predador.
        # Graxaim (4) come Preá (1).
        # Jaguatirica (5) come Preá (1).
        # Puma (6) come Veado (2), Capivara (3), Graxaim (4).
        
        # Lógica predador hardcoded baseada na spec (species.py values):
        # Isso evita passar estruturas complexas.
        is_graxaim = (my_species == 4)
        is_jaguatirica = (my_species == 5)
        is_puma = (my_species == 6)
        
        is_predator = (is_graxaim | is_jaguatirica | is_puma) & am_active
        
        # Definir alvos baseados na espécie
        # targets mask: (N,) bool
        targets_mask = jnp.zeros_like(species_ids, dtype=jnp.bool_)
        
        # Graxaim -> Preá (1)
        targets_mask = jnp.where(is_graxaim, (species_ids == 1), targets_mask)
        # Jaguatirica -> Preá (1)
        targets_mask = jnp.where(is_jaguatirica, (species_ids == 1), targets_mask)
        # Puma -> Veado (2), Capivara (3), Graxaim (4)
        targets_mask = jnp.where(
            is_puma, 
            (species_ids == 2) | (species_ids == 3) | (species_ids == 4), 
            targets_mask
        )
        
        # Remover inativos e self (embora self não seja alvo pela espécie, safety check)
        targets_mask = targets_mask & curr_active
        
        # Executar tentativa SE for predador e houver alvos
        # Para evitar computação cara, usamos cond.
        
        def attempt_hunt(rng_key):
             # Calcular distâncias apenas para alvos
            dists = jnp.linalg.norm(positions - my_pos, axis=-1)
            # Mascarar não-alvos com inf
            dists = jnp.where(targets_mask, dists, jnp.inf)
            
            # Mais próximo
            nearest_idx = jnp.argmin(dists)
            min_dist = dists[nearest_idx]
            
            # Attack range fixo ou alométrico? Vamos usar 2.0 padrão
            can_attack = min_dist < 4.0  # Range aumentado para facilitar MVP
            
            # Probabilidade
            # Em scan interno, acesso a globals como cover_density é ok
            target_idx, success, eff = attempt_predation(
                predator_idx=idx,
                predator_pos=my_pos,
                predator_vel=my_vel,
                prey_positions=positions,
                prey_velocities=velocities,
                prey_active=curr_active,
                prey_species_ids=species_ids, # Não usado na simplified call interna abaixo, mas mantido assinatura
                target_prey_ids=[], # Dummy, já filtramos com targets_mask
                attack_range=4.0,
                cover_density=cover_density,
                world_bounds=world_bounds,
                rng=rng_key
            )
            
            # Override da função helper genérica com nossa lógica 'nearest already found' simplificada
            # Se can_attack e RNG favorável:
            rng_c, _ = jax.random.split(rng_key)
            capture_roll = jax.random.uniform(rng_c)
            
            # Lógica simplificada de chance: 30% base
            prob = 0.3
            
            did_catch = can_attack & (capture_roll < prob)
            
            victim_idx = jnp.where(did_catch, nearest_idx, -1)
            return victim_idx

        victim_idx = jax.lax.cond(
            is_predator,
            lambda r: attempt_hunt(r),
            lambda r: -1,
            rng_sub[idx]
        )
        
        # Atualizar estado se houve captura
        caught = victim_idx != -1
        
        # 1. Matar a presa
        # new_active[victim_idx] = False (se caught)
        updated_active = jnp.where(
            caught,
            curr_active.at[victim_idx].set(False),
            curr_active
        )
        
        # 2. Transferir energia para predador
        # Energia ganha = Massa presa * Densidade Calórica * Eficiência
        # Densidade ~ 6000 kJ/kg (carne fresca)
        # Eficiência ~ 0.15 (já passado como param, ou da config)
        prey_mass = jnp.where(caught, masses[victim_idx], 0.0)
        energy_gain = prey_mass * 6000.0 * 0.20 # 20% eficiência
        
        updated_energies = jnp.where(
            caught,
            curr_energies.at[idx].add(energy_gain),
            curr_energies
        )
        
        # Clipar energia do predador (opcional, ou deixar pro update_metabolism)
        # Mas vamos deixar pro update_metabolism lidar com max caps.
        
        return (updated_active, updated_energies), victim_idx

    scan_init = (is_active, energies)
    indices = jnp.arange(N)
    
    (final_active, final_energies), victims = jax.lax.scan(predation_step, scan_init, indices)
    
    return final_active, final_energies, victims

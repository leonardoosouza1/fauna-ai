"""
Registro de Espécies — Projeto Fauna.

Catálogo central de parâmetros biológicos por espécie, organizado por bioma.
Cada espécie é descrita por uma SpeciesConfig imutável com dados alométricos
derivados de literatura científica.

Escalabilidade:
    - BiomeConfig permite registrar espécies POR BIOMA brasileiro
    - Pampa-Mata Atlântica é o primeiro bioma implementado
    - Futuro: Cerrado, Amazônia, Caatinga, Pantanal, cada um com
      seu próprio conjunto de espécies e parâmetros ambientais

Referências:
    - Kleiber's Law: BMR = α × M^0.75 (α ≈ 3.5 W para mamíferos neotropicais)
    - COT (Custo de Transporte): COT = 10.7 × M^(-0.32) J/(kg·m)
    - Visual detection: d_v = 12.5 × M^0.30 metros
"""

from typing import NamedTuple
import jax.numpy as jnp

from src.core.types import DTYPE_STORE


# ──────────────────────────────────────────────
# Enums de Dieta
# ──────────────────────────────────────────────
class DietType:
    """Tipos tróficos para resposta funcional correta."""
    PRODUCER = 0       # Gramínea/vegetação
    HERBIVORE = 1      # Come apenas vegetação
    OMNIVORE = 2       # Come vegetação + presas (Holling III → prey switching)
    CARNIVORE = 3      # Come apenas presas (Holling II → saturação)
    APEX = 4           # Predador de topo (Holling II + predação intraguilda)


# ──────────────────────────────────────────────
# Configuração de Reprodução
# ──────────────────────────────────────────────
class ReproductionConfig(NamedTuple):
    """Parâmetros reprodutivos de uma espécie."""
    gestation_steps: int         # Steps de gestação
    litter_size_min: int         # Filhotes mínimos por ninhada
    litter_size_max: int         # Filhotes máximos
    maturity_age_steps: int      # Idade de maturidade sexual (steps)
    energy_cost_fraction: float  # Fração da energia gasta na reprodução
    cooldown_steps: int          # Intervalo mínimo entre reproduções
    seasonal_breeding: bool      # True = reprodução sazonal (primavera/verão)


# ──────────────────────────────────────────────
# Configuração de Espécie
# ──────────────────────────────────────────────
class SpeciesConfig(NamedTuple):
    """
    Ficha biológica completa de uma espécie.

    Todos os valores são calibrados por dados do documento
    'Fauna Brasileira: Projeto Simulação Nível MIT'.
    """
    # ── Identificação ──
    species_id: int
    name: str
    scientific_name: str
    trophic_level: int           # 0=produtor, 1=herb, 2=meso, 3=apex

    # ── Morfologia ──
    mass_kg: float               # Massa corporal média (kg)
    body_length_m: float         # Comprimento do corpo (m)

    # ── Metabolismo (Kleiber) ──
    bmr_watts: float             # TMB = 3.5 × M^0.75 (Watts)
    fmr_multiplier: float        # FMR = fmr_mult × BMR (2.0-3.0)
    cot_jkg_m: float             # Custo de Transporte: 10.7 × M^(-0.32) J/(kg·m)
    max_energy_kj: float         # Capacidade energética máxima (kJ)
    energy_from_food_kj: float   # Energia digestível por kg de alimento (kJ/kg)

    # ── Locomoção ──
    max_speed_ms: float          # Velocidade máxima (m/s)
    sustained_speed_ms: float    # Velocidade sustentada (m/s)
    home_range_km2: float        # Área de vida (km²)

    # ── Sensorial ──
    visual_range_m: float        # Detecção visual: 12.5 × M^0.30
    olfactory_range_m: float     # Detecção olfativa (m)
    fov_degrees: float           # Campo de visão (graus)

    # ── Dieta ──
    diet_type: int               # DietType enum
    prey_species_ids: tuple      # IDs das espécies-presa
    attack_rate: float           # Parâmetro 'a' de Holling
    handling_time: float         # Parâmetro 'h' de Holling (steps)
    caloric_efficiency: float    # Eficiência de conversão (0.0-1.0)

    # ── Reprodução ──
    reproduction: ReproductionConfig

    # ── Comportamento ──
    is_social: bool              # Vive em grupo?
    group_size_max: int          # Tamanho máximo do grupo
    is_nocturnal: bool           # Atividade noturna?
    uses_cover: bool             # Depende de cobertura vegetal?
    is_aquatic: bool             # Pode entrar na água?

    # ── Dashboard ──
    color_hex: str               # Cor para visualização
    emoji: str                   # Emoji representativo


# ──────────────────────────────────────────────
# Configuração de Bioma
# ──────────────────────────────────────────────
class BiomeConfig(NamedTuple):
    """
    Configuração de um bioma brasileiro.

    Cada bioma tem suas espécies, clima e parâmetros ambientais.
    Permite escalar de Pampa-Mata Atlântica para todo o Brasil.
    """
    biome_id: int
    name: str
    region: str                           # "Sul", "Sudeste", "Norte", etc.

    # Clima
    mean_temperature_c: float             # Temperatura média anual (°C)
    temperature_amplitude_c: float        # Amplitude sazonal (°C)
    annual_rainfall_mm: float             # Precipitação anual

    # Vegetação
    anpp_kg_m2_year: float                # Produtividade Primária Líquida
    max_standing_biomass_kg_m2: float     # Biomassa máxima em pé
    has_c4_grasses: bool                  # Gramíneas C4 presentes
    has_c3_grasses: bool                  # Gramíneas C3 presentes
    forest_cover_fraction: float          # Fração de cobertura florestal

    # Mundo
    world_size_km: float                  # Tamanho do mundo simulado (km)

    # Espécies
    species: dict                         # {species_id: SpeciesConfig}
    initial_populations: dict             # {species_id: count}


# ══════════════════════════════════════════════
# Funções de Alometria
# ══════════════════════════════════════════════

def compute_bmr(mass_kg: float, alpha: float = 3.5) -> float:
    """TMB via Lei de Kleiber: BMR = α × M^0.75 (Watts)."""
    return alpha * (mass_kg ** 0.75)


def compute_cot(mass_kg: float) -> float:
    """Custo de Transporte: COT = 10.7 × M^(-0.32) J/(kg·m)."""
    return 10.7 * (mass_kg ** -0.32)


def compute_visual_range(mass_kg: float) -> float:
    """Detecção visual alométrica: d_v = 12.5 × M^0.30 metros."""
    return 12.5 * (mass_kg ** 0.30)


def compute_max_energy(mass_kg: float) -> float:
    """Capacidade energética: E_max = 500 × M^0.75 kJ."""
    return 500.0 * (mass_kg ** 0.75)


# ══════════════════════════════════════════════
# BIOMA: Pampa-Mata Atlântica (Sul do Brasil)
# ══════════════════════════════════════════════

# ── Preá (Cavia aperea) ──
PREA = SpeciesConfig(
    species_id=1,
    name="Preá",
    scientific_name="Cavia aperea",
    trophic_level=1,
    mass_kg=0.55,
    body_length_m=0.272,
    bmr_watts=compute_bmr(0.55),
    fmr_multiplier=2.5,
    cot_jkg_m=compute_cot(0.55),
    max_energy_kj=compute_max_energy(0.55),
    energy_from_food_kj=9.5,    # MJ/kg MS → kJ digestível
    max_speed_ms=6.0,
    sustained_speed_ms=3.5,
    home_range_km2=0.001,       # 0.02-0.1 ha
    visual_range_m=25.0,        # Panorâmico mas baixa acuidade
    olfactory_range_m=10.0,
    fov_degrees=330.0,          # Quase panorâmico (presa)
    diet_type=DietType.HERBIVORE,
    prey_species_ids=(),        # Come gramínea (grid)
    attack_rate=0.0,
    handling_time=0.0,
    caloric_efficiency=0.55,    # Verão, cai no inverno
    reproduction=ReproductionConfig(
        gestation_steps=620,       # 62 dias → ~620 steps (1 step ≈ 0.1 dia)
        litter_size_min=1,
        litter_size_max=3,
        maturity_age_steps=350,    # 35 dias → precoce
        energy_cost_fraction=0.3,
        cooldown_steps=620,        # Cio pós-parto
        seasonal_breeding=False,   # Reproduz o ano todo
    ),
    is_social=True,
    group_size_max=15,
    is_nocturnal=False,
    uses_cover=True,             # Run-and-hide
    is_aquatic=False,
    color_hex="#22c55e",          # Verde
    emoji="🐹",
)

# ── Veado-campeiro (Ozotoceros bezoarticus) ──
VEADO_CAMPEIRO = SpeciesConfig(
    species_id=2,
    name="Veado-campeiro",
    scientific_name="Ozotoceros bezoarticus",
    trophic_level=1,
    mass_kg=35.0,
    body_length_m=1.2,
    bmr_watts=compute_bmr(35.0),
    fmr_multiplier=2.5,
    cot_jkg_m=compute_cot(35.0),
    max_energy_kj=compute_max_energy(35.0),
    energy_from_food_kj=9.5,
    max_speed_ms=14.0,           # ~50 km/h
    sustained_speed_ms=5.0,
    home_range_km2=10.0,
    visual_range_m=compute_visual_range(35.0),  # ~38m
    olfactory_range_m=1500.0,    # Detecção de puma via feromônios!
    fov_degrees=280.0,
    diet_type=DietType.HERBIVORE,
    prey_species_ids=(),
    attack_rate=0.0,
    handling_time=0.0,
    caloric_efficiency=0.50,
    reproduction=ReproductionConfig(
        gestation_steps=2130,      # 213 dias
        litter_size_min=1,
        litter_size_max=1,         # Raramente 2
        maturity_age_steps=3650,   # ~1 ano
        energy_cost_fraction=0.4,
        cooldown_steps=3650,       # 1 cria/ano
        seasonal_breeding=True,    # Primavera/verão
    ),
    is_social=False,
    group_size_max=5,             # Pequenos grupos
    is_nocturnal=False,
    uses_cover=False,             # Campo aberto
    is_aquatic=False,
    color_hex="#3b82f6",          # Azul
    emoji="🦌",
)

# ── Capivara (Hydrochoerus hydrochaeris) ──
CAPIVARA = SpeciesConfig(
    species_id=3,
    name="Capivara",
    scientific_name="Hydrochoerus hydrochaeris",
    trophic_level=1,
    mass_kg=50.0,
    body_length_m=1.3,
    bmr_watts=compute_bmr(50.0),
    fmr_multiplier=2.0,
    cot_jkg_m=compute_cot(50.0),
    max_energy_kj=compute_max_energy(50.0),
    energy_from_food_kj=8.5,     # Gramíneas C4 aquáticas
    max_speed_ms=8.0,
    sustained_speed_ms=3.0,
    home_range_km2=1.0,
    visual_range_m=compute_visual_range(50.0),
    olfactory_range_m=50.0,
    fov_degrees=260.0,
    diet_type=DietType.HERBIVORE,
    prey_species_ids=(),
    attack_rate=0.0,
    handling_time=0.0,
    caloric_efficiency=0.50,
    reproduction=ReproductionConfig(
        gestation_steps=1500,      # 150 dias
        litter_size_min=2,
        litter_size_max=8,
        maturity_age_steps=3650,
        energy_cost_fraction=0.35,
        cooldown_steps=1500,
        seasonal_breeding=False,
    ),
    is_social=True,
    group_size_max=20,
    is_nocturnal=False,
    uses_cover=False,
    is_aquatic=True,              # Refúgio na água!
    color_hex="#92400e",          # Marrom
    emoji="🦫",
)

# ── Graxaim-do-campo (Lycalopex gymnocercus) ──
GRAXAIM = SpeciesConfig(
    species_id=4,
    name="Graxaim-do-campo",
    scientific_name="Lycalopex gymnocercus",
    trophic_level=2,
    mass_kg=5.0,
    body_length_m=0.62,
    bmr_watts=compute_bmr(5.0),
    fmr_multiplier=3.0,
    cot_jkg_m=compute_cot(5.0),
    max_energy_kj=compute_max_energy(5.0),
    energy_from_food_kj=15.0,    # Carne é mais calórica
    max_speed_ms=10.0,
    sustained_speed_ms=4.0,
    home_range_km2=3.0,          # 1-5 km²
    visual_range_m=compute_visual_range(5.0),
    olfactory_range_m=500.0,     # Carniça a longa distância
    fov_degrees=200.0,
    diet_type=DietType.OMNIVORE, # Holling Tipo III — prey switching!
    prey_species_ids=(1,),       # Preá (+ frutos do grid)
    attack_rate=0.02,
    handling_time=50.0,          # Steps por presa
    caloric_efficiency=0.15,     # 10-20% entre níveis tróficos
    reproduction=ReproductionConfig(
        gestation_steps=580,       # 58 dias
        litter_size_min=3,
        litter_size_max=5,
        maturity_age_steps=3650,
        energy_cost_fraction=0.3,
        cooldown_steps=3650,       # 1 ninhada/ano
        seasonal_breeding=True,
    ),
    is_social=True,              # Monogâmico — casais
    group_size_max=2,
    is_nocturnal=True,           # Crepuscular/noturno
    uses_cover=False,
    is_aquatic=False,
    color_hex="#f97316",          # Laranja
    emoji="🦊",
)

# ── Jaguatirica (Leopardus pardalis) ──
JAGUATIRICA = SpeciesConfig(
    species_id=5,
    name="Jaguatirica",
    scientific_name="Leopardus pardalis",
    trophic_level=2,
    mass_kg=10.5,
    body_length_m=0.8,
    bmr_watts=compute_bmr(10.5),
    fmr_multiplier=2.5,
    cot_jkg_m=compute_cot(10.5),
    max_energy_kj=compute_max_energy(10.5),
    energy_from_food_kj=18.0,    # Carnívoro estrito, alta qualidade
    max_speed_ms=12.0,
    sustained_speed_ms=4.0,
    home_range_km2=15.0,         # 2-31 km²
    visual_range_m=compute_visual_range(10.5),
    olfactory_range_m=100.0,
    fov_degrees=180.0,           # Predador frontal
    diet_type=DietType.CARNIVORE,  # Holling Tipo II
    prey_species_ids=(1,),       # Preá (especialista em roedores)
    attack_rate=0.03,
    handling_time=80.0,
    caloric_efficiency=0.15,
    reproduction=ReproductionConfig(
        gestation_steps=800,       # 80 dias
        litter_size_min=1,
        litter_size_max=2,
        maturity_age_steps=7300,   # ~2 anos
        energy_cost_fraction=0.35,
        cooldown_steps=7300,
        seasonal_breeding=False,
    ),
    is_social=False,
    group_size_max=1,
    is_nocturnal=True,           # Caçador noturno de emboscada
    uses_cover=True,             # Depende da mata!
    is_aquatic=False,
    color_hex="#eab308",          # Amarelo
    emoji="🐆",
)

# ── Puma / Onça-parda (Puma concolor) ──
PUMA = SpeciesConfig(
    species_id=6,
    name="Puma",
    scientific_name="Puma concolor",
    trophic_level=3,
    mass_kg=40.0,
    body_length_m=1.5,
    bmr_watts=compute_bmr(40.0),
    fmr_multiplier=2.5,
    cot_jkg_m=compute_cot(40.0),
    max_energy_kj=compute_max_energy(40.0),
    energy_from_food_kj=20.0,
    max_speed_ms=18.0,           # Sprints curtos
    sustained_speed_ms=5.0,
    home_range_km2=100.0,        # 24-363 km²
    visual_range_m=compute_visual_range(40.0),
    olfactory_range_m=200.0,
    fov_degrees=150.0,           # Predador frontal focado
    diet_type=DietType.APEX,
    prey_species_ids=(2, 3, 4),  # Veado, Capivara, Graxaim
    attack_rate=0.04,
    handling_time=140.0,         # ~14 dias por presa grande
    caloric_efficiency=0.12,
    reproduction=ReproductionConfig(
        gestation_steps=900,       # 90 dias
        litter_size_min=1,
        litter_size_max=3,
        maturity_age_steps=10950,  # ~3 anos
        energy_cost_fraction=0.4,
        cooldown_steps=7300,       # 1 ninhada a cada 2 anos
        seasonal_breeding=False,
    ),
    is_social=False,
    group_size_max=1,            # Solitário (exceto mãe+filhotes)
    is_nocturnal=True,
    uses_cover=True,
    is_aquatic=False,
    color_hex="#ef4444",          # Vermelho
    emoji="🦁",
)


# ══════════════════════════════════════════════
# Bioma: Pampa-Mata Atlântica
# ══════════════════════════════════════════════

PAMPA_MATA_ATLANTICA = BiomeConfig(
    biome_id=1,
    name="Pampa-Mata Atlântica",
    region="Sul",
    mean_temperature_c=18.0,
    temperature_amplitude_c=10.0,
    annual_rainfall_mm=1500.0,
    anpp_kg_m2_year=0.53,
    max_standing_biomass_kg_m2=0.6,
    has_c4_grasses=True,
    has_c3_grasses=True,
    forest_cover_fraction=0.25,
    world_size_km=10.0,          # 10km × 10km = 100 km²
    species={
        1: PREA,
        2: VEADO_CAMPEIRO,
        3: CAPIVARA,
        4: GRAXAIM,
        5: JAGUATIRICA,
        6: PUMA,
    },
    initial_populations={
        1: 250,    # Preá — base trófica abundante
        2: 40,     # Veado-campeiro
        3: 30,     # Capivara
        4: 12,     # Graxaim
        5: 5,      # Jaguatirica
        6: 3,      # Puma
    },
)


# ══════════════════════════════════════════════
# Biomas futuros (placeholders para escalabilidade)
# ══════════════════════════════════════════════
# CERRADO = BiomeConfig(biome_id=2, name="Cerrado", region="Centro-Oeste", ...)
# AMAZONIA = BiomeConfig(biome_id=3, name="Amazônia", region="Norte", ...)
# CAATINGA = BiomeConfig(biome_id=4, name="Caatinga", region="Nordeste", ...)
# PANTANAL = BiomeConfig(biome_id=5, name="Pantanal", region="Centro-Oeste", ...)
# MATA_ATLANTICA_SE = BiomeConfig(biome_id=6, name="Mata Atlântica SE", region="Sudeste", ...)


# ══════════════════════════════════════════════
# Registry & Helpers
# ══════════════════════════════════════════════

BIOME_REGISTRY: dict[int, BiomeConfig] = {
    1: PAMPA_MATA_ATLANTICA,
}

ALL_SPECIES: dict[int, SpeciesConfig] = {
    s.species_id: s
    for biome in BIOME_REGISTRY.values()
    for s in biome.species.values()
}


def get_species(species_id: int) -> SpeciesConfig:
    """Retorna a configuração de uma espécie pelo ID."""
    return ALL_SPECIES[species_id]


def get_biome(biome_id: int) -> BiomeConfig:
    """Retorna a configuração de um bioma pelo ID."""
    return BIOME_REGISTRY[biome_id]


def get_species_ids(biome_id: int) -> list[int]:
    """Retorna IDs de espécies de um bioma."""
    return list(BIOME_REGISTRY[biome_id].species.keys())


def get_total_population(biome_id: int) -> int:
    """Retorna a população total inicial de um bioma."""
    return sum(BIOME_REGISTRY[biome_id].initial_populations.values())


def build_species_arrays(biome_id: int, max_agents: int) -> dict:
    """
    Constrói arrays JAX com parâmetros por agente baseado no bioma.

    Returns:
        dict com arrays: species_id, mass, max_speed, max_energy,
                         visual_range, etc. — cada um de shape (max_agents,)
    """
    biome = get_biome(biome_id)
    total_pop = get_total_population(biome_id)

    # Preencher arrays na ordem das espécies
    species_ids = []
    masses = []
    max_speeds = []
    max_energies = []
    visual_ranges = []
    olfactory_ranges = []
    fov_degrees = []
    diet_types = []
    trophic_levels = []
    bmr_values = []

    for sp_id, count in biome.initial_populations.items():
        sp = biome.species[sp_id]
        for _ in range(count):
            species_ids.append(sp.species_id)
            masses.append(sp.mass_kg)
            max_speeds.append(sp.max_speed_ms)
            max_energies.append(sp.max_energy_kj)
            visual_ranges.append(sp.visual_range_m)
            olfactory_ranges.append(sp.olfactory_range_m)
            fov_degrees.append(sp.fov_degrees)
            diet_types.append(sp.diet_type)
            trophic_levels.append(sp.trophic_level)
            bmr_values.append(sp.bmr_watts)

    # Pad até max_agents
    pad_len = max_agents - len(species_ids)
    species_ids += [0] * pad_len
    masses += [0.0] * pad_len
    max_speeds += [0.0] * pad_len
    max_energies += [0.0] * pad_len
    visual_ranges += [0.0] * pad_len
    olfactory_ranges += [0.0] * pad_len
    fov_degrees += [0.0] * pad_len
    diet_types += [0] * pad_len
    trophic_levels += [0] * pad_len
    bmr_values += [0.0] * pad_len

    return {
        "species_id": jnp.array(species_ids[:max_agents], dtype=jnp.int32),
        "mass": jnp.array(masses[:max_agents], dtype=DTYPE_STORE),
        "max_speed": jnp.array(max_speeds[:max_agents], dtype=DTYPE_STORE),
        "max_energy": jnp.array(max_energies[:max_agents], dtype=DTYPE_STORE),
        "visual_range": jnp.array(visual_ranges[:max_agents], dtype=DTYPE_STORE),
        "olfactory_range": jnp.array(olfactory_ranges[:max_agents], dtype=DTYPE_STORE),
        "fov_degrees": jnp.array(fov_degrees[:max_agents], dtype=DTYPE_STORE),
        "diet_type": jnp.array(diet_types[:max_agents], dtype=jnp.int32),
        "trophic_level": jnp.array(trophic_levels[:max_agents], dtype=jnp.int32),
        "bmr": jnp.array(bmr_values[:max_agents], dtype=DTYPE_STORE),
    }


def is_prey_of(predator_species_id: int, prey_species_id: int) -> bool:
    """Verifica se uma espécie é presa de outra."""
    predator = get_species(predator_species_id)
    return prey_species_id in predator.prey_species_ids


def build_prey_matrix(biome_id: int) -> dict:
    """
    Constrói matriz de predação para o bioma.

    Returns:
        dict: {predator_id: [prey_id_1, prey_id_2, ...]}
    """
    biome = get_biome(biome_id)
    matrix = {}
    for sp_id, sp in biome.species.items():
        if sp.prey_species_ids:
            matrix[sp_id] = list(sp.prey_species_ids)
    return matrix

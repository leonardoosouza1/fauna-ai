"""
Tipo aliases e constantes para o universo Fauna.

Centraliza definições de tipo JAX para evitar acoplamento
direto com jax.numpy em camadas superiores.
"""

from typing import Any, Dict, NamedTuple

import jax
import jax.numpy as jnp

# ──────────────────────────────────────────────
# Aliases de Tipo JAX
# ──────────────────────────────────────────────
Array = jax.Array
PRNGKey = jax.Array
Scalar = float | jnp.ndarray

# Pytree genérico (dicionários aninhados, NamedTuples, etc.)
PyTree = Any

# Identificador numérico de agente (int32 para economia de memória)
AgentID = jnp.int32

# ──────────────────────────────────────────────
# Constantes de Precisão
# ──────────────────────────────────────────────
# bfloat16 nativo na RDNA2 — reduz consumo de VRAM pela metade
DTYPE_COMPUTE = jnp.bfloat16
DTYPE_STORE = jnp.float32  # Para checkpoints e operações que exigem precisão

# ──────────────────────────────────────────────
# Constantes do Wavefront RDNA2
# ──────────────────────────────────────────────
WAVE_SIZE = 32  # Threads por wavefront na RDNA2
ALIGNMENT_MULTIPLE = 64  # Múltiplo ideal para paralelismo (2 * WAVE_SIZE)

# ──────────────────────────────────────────────
# Tipo para configurações (Hydra OmegaConf)
# ──────────────────────────────────────────────
Config = Dict[str, Any]

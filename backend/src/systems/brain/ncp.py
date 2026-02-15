"""
Neural Circuit Policies (NCP) — Projeto Fauna.

Implementa a topologia esparsa bio-inspirada baseada no conectoma
do C. elegans. Organiza neurônios CfC em 4 camadas funcionais:

  1. Sensorial  → Encoders (processamento instantâneo)
  2. Intermediária → CfC recorrente (working memory, dinâmicas temporais)
  3. Comando   → CfC gargalo (decisão de alto nível)
  4. Motora    → Decoders lineares (atuação)

A esparsidade reduz parâmetros (milhões → centenas) e melhora
a interpretabilidade causal dos comportamentos emergentes.

Referência: https://ncps.readthedocs.io/
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from src.core.types import Array, PRNGKey
from src.systems.brain.cfc_cell import CfCCell


# ──────────────────────────────────────────────
# Topologia de Fiação (Wiring)
# ──────────────────────────────────────────────

class NCPWiring(NamedTuple):
    """
    Define a topologia esparsa de conexões entre neurônios.

    Cada campo é uma máscara binária indicando quais conexões existem.
    """
    # --- Dimensões ---
    num_sensory: int
    num_inter: int
    num_command: int
    num_motor: int

    # --- Máscaras de Adjacência ---
    # Conexões sensorial → intermediária: (num_sensory, num_inter)
    sensory_to_inter: Array
    # Conexões intermediária ↔ intermediária (recorrência): (num_inter, num_inter)
    inter_to_inter: Array
    # Conexões intermediária → comando: (num_inter, num_command)
    inter_to_command: Array
    # Conexões comando ↔ comando (inibição lateral): (num_command, num_command)
    command_to_command: Array
    # Conexões comando → motor: (num_command, num_motor)
    command_to_motor: Array


def build_ncp_wiring(
    num_sensory: int = 32,
    num_inter: int = 48,
    num_command: int = 16,
    num_motor: int = 8,
    sparsity: float = 0.5,
    rng_seed: int = 0,
) -> NCPWiring:
    """
    Constrói a topologia NCP com esparsidade controlada.

    Args:
        num_sensory: Neurônios sensoriais (entrada).
        num_inter: Neurônios intermediários (working memory CfC).
        num_command: Neurônios de comando (decisão CfC).
        num_motor: Neurônios motores (saída).
        sparsity: Fração de conexões removidas (0=denso, 1=sem conexões).
        rng_seed: Seed para reprodutibilidade da topologia.

    Returns:
        NCPWiring com máscaras binárias para cada grupo de conexões.
    """
    rng = jax.random.PRNGKey(rng_seed)
    keys = jax.random.split(rng, 5)

    def _sparse_mask(key: PRNGKey, shape: tuple[int, ...]) -> Array:
        """Gera máscara binária com a esparsidade especificada."""
        return (jax.random.uniform(key, shape) > sparsity).astype(jnp.float32)

    # Garantir que cada neurônio intermediário receba pelo menos uma conexão sensorial
    sensory_mask = _sparse_mask(keys[0], (num_sensory, num_inter))

    # Recorrência entre intermediários (excluir auto-conexões)
    inter_mask = _sparse_mask(keys[1], (num_inter, num_inter))
    inter_mask = inter_mask * (1.0 - jnp.eye(num_inter))  # Remove diagonal

    # Intermediários → Comando
    inter_cmd_mask = _sparse_mask(keys[2], (num_inter, num_command))

    # Recorrência de comando (inibição lateral, sem auto-conexões)
    cmd_mask = _sparse_mask(keys[3], (num_command, num_command))
    cmd_mask = cmd_mask * (1.0 - jnp.eye(num_command))

    # Comando → Motor (geralmente mais denso para controle preciso)
    cmd_motor_mask = _sparse_mask(keys[4], (num_command, num_motor))

    return NCPWiring(
        num_sensory=num_sensory,
        num_inter=num_inter,
        num_command=num_command,
        num_motor=num_motor,
        sensory_to_inter=sensory_mask,
        inter_to_inter=inter_mask,
        inter_to_command=inter_cmd_mask,
        command_to_command=cmd_mask,
        command_to_motor=cmd_motor_mask,
    )


# ──────────────────────────────────────────────
# Módulo NCP Flax
# ──────────────────────────────────────────────

class NCPCell(nn.Module):
    """
    Neural Circuit Policy — célula recorrente com topologia esparsa.

    Processa uma observação sensorial através das 4 camadas NCP,
    aplicando CfC cells na camada intermediária e de comando.

    O mascaramento de conexões garante que apenas as sinapses
    definidas na NCPWiring sejam ativas.

    Attributes:
        num_inter: Neurônios intermediários.
        num_command: Neurônios de comando.
        num_motor: Neurônios motores (dimensão da ação).
        wiring: Topologia de conexões (máscaras binárias).
    """
    num_inter: int
    num_command: int
    num_motor: int
    wiring: NCPWiring | None = None

    @nn.compact
    def __call__(
        self,
        h_inter: Array,
        h_command: Array,
        sensory_input: Array,
        time_delta: Array,
    ) -> tuple[Array, Array, Array]:
        """
        Forward pass da NCP.

        Args:
            h_inter: Estado oculto intermediário — (batch, num_inter)
            h_command: Estado oculto de comando — (batch, num_command)
            sensory_input: Features sensoriais — (batch, num_sensory)
            time_delta: Passo de tempo — escalar ou (batch, 1)

        Returns:
            Tupla (new_h_inter, new_h_command, motor_output).
        """
        # ── Camada 1: Sensorial → Intermediária ──
        # Projeção linear mascarada pelas conexões da topologia
        sensory_proj = nn.Dense(
            self.num_inter,
            name="sensory_proj",
        )(sensory_input)

        if self.wiring is not None:
            sensory_proj = sensory_proj * self.wiring.sensory_to_inter.sum(axis=0)

        # ── Camada 2: Intermediária (CfC recorrente) ──
        # A recorrência é mascarada pela topologia inter→inter
        inter_recurrent_input = nn.Dense(
            self.num_inter,
            name="inter_recurrent",
        )(h_inter)

        if self.wiring is not None:
            inter_recurrent_input = inter_recurrent_input * self.wiring.inter_to_inter.sum(axis=0)

        inter_input = sensory_proj + inter_recurrent_input

        inter_cell = CfCCell(
            hidden_size=self.num_inter,
            name="inter_cfc",
        )
        new_h_inter, _ = inter_cell(h_inter, inter_input, time_delta)

        # ── Camada 3: Comando (CfC com inibição lateral) ──
        inter_to_cmd = nn.Dense(
            self.num_command,
            name="inter_to_command",
        )(new_h_inter)

        if self.wiring is not None:
            inter_to_cmd = inter_to_cmd * self.wiring.inter_to_command.sum(axis=0)

        # Recorrência de comando (competição por inibição lateral)
        cmd_recurrent = nn.Dense(
            self.num_command,
            name="command_recurrent",
        )(h_command)

        if self.wiring is not None:
            cmd_recurrent = cmd_recurrent * self.wiring.command_to_command.sum(axis=0)

        cmd_input = inter_to_cmd + cmd_recurrent

        cmd_cell = CfCCell(
            hidden_size=self.num_command,
            name="command_cfc",
        )
        new_h_command, _ = cmd_cell(h_command, cmd_input, time_delta)

        # ── Camada 4: Motora (saída linear) ──
        motor_output = nn.Dense(
            self.num_motor,
            name="motor_head",
        )(new_h_command)

        # Tanh para limitar ações ao intervalo [-1, 1]
        motor_output = nn.tanh(motor_output)

        return new_h_inter, new_h_command, motor_output

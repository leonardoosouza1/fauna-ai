"""
AgentBrain — Política Neural Completa (IARA v10).

Integra o pipeline cognitivo completo:
  Observação Bruta → Encoder Sensorial → NCP (Inter+Command CfC) → Ação

Este módulo é o ponto de entrada para a cognição de um agente.
Ele encapsula a NCP e fornece a interface que o loop de simulação utiliza.

Futuro (Fases 3-4):
  - Global Workspace Theory (GWT) para atenção seletiva
  - Mixture of Experts (MoE) com confidence gating
  - Plasticidade Hebbiana online
"""

import jax.numpy as jnp
import flax.linen as nn

from src.core.types import Array
from src.systems.brain.ncp import NCPCell, NCPWiring


class SensoryEncoder(nn.Module):
    """
    Encoder sensorial: converte observações brutas em features latentes.

    Camada mais externa da NCP — sem recorrência temporal.
    Usa MLP para projetar observações de dimensão variável
    para o espaço de entrada da camada intermediária.

    Attributes:
        output_dim: Dimensão do vetor latente sensorial.
    """
    output_dim: int

    @nn.compact
    def __call__(self, obs: Array) -> Array:
        x = nn.Dense(self.output_dim * 2, name="enc_hidden")(obs)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim, name="enc_out")(x)
        return nn.tanh(x)


class AgentBrain(nn.Module):
    """
    Cérebro completo de um agente Fauna.

    Pipeline:
        obs → SensoryEncoder → NCPCell → action_logits

    O estado oculto (h_inter, h_command) é gerenciado externamente
    pelo loop de simulação (paradigma funcional do JAX: estado explícito).

    Attributes:
        obs_dim: Dimensão da observação bruta.
        action_dim: Dimensão da saída de ação.
        num_sensory: Neurônios sensoriais (codificados pelo encoder).
        num_inter: Neurônios intermediários CfC.
        num_command: Neurônios de comando CfC.
        wiring: Topologia NCP opcional.
    """
    obs_dim: int
    action_dim: int
    num_sensory: int = 32
    num_inter: int = 48
    num_command: int = 16
    wiring: NCPWiring | None = None

    @nn.compact
    def __call__(
        self,
        h_inter: Array,
        h_command: Array,
        obs: Array,
        time_delta: Array,
    ) -> tuple[Array, Array, Array]:
        """
        Forward pass do cérebro.

        Args:
            h_inter: Estado oculto intermediário — (batch, num_inter)
            h_command: Estado oculto de comando — (batch, num_command)
            obs: Observação bruta do ambiente — (batch, obs_dim)
            time_delta: Δt físico — escalar ou (batch, 1)

        Returns:
            Tupla (new_h_inter, new_h_command, actions).
            - actions: no intervalo [-1, 1] via tanh.
        """
        # ── Encoder Sensorial ──
        sensory_features = SensoryEncoder(
            output_dim=self.num_sensory,
            name="sensory_encoder",
        )(obs)

        # ── NCP (Inter + Command + Motor) ──
        ncp = NCPCell(
            num_inter=self.num_inter,
            num_command=self.num_command,
            num_motor=self.action_dim,
            wiring=self.wiring,
            name="ncp",
        )

        new_h_inter, new_h_command, actions = ncp(
            h_inter, h_command, sensory_features, time_delta,
        )

        return new_h_inter, new_h_command, actions

    def initialize_hidden(self, batch_size: int = 1) -> tuple[Array, Array]:
        """Cria estados ocultos zerados para inicialização."""
        h_inter = jnp.zeros((batch_size, self.num_inter))
        h_command = jnp.zeros((batch_size, self.num_command))
        return h_inter, h_command

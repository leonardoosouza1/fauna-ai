"""
Closed-form Continuous-time (CfC) Cell — Projeto Fauna.

Implementação da célula neural CfC em Flax Linen, baseada no paper
"Closed-form Continuous-time Neural Networks" (Hasani et al., 2022).

A CfC resolve a equação diferencial LTC de forma fechada:
    h_new = σ(time_gate · Δt) ⊙ ff1 + (1 - σ(time_gate · Δt)) ⊙ ff2

Vantagens para a RX 6750 XT:
  - Inferência em O(1) vs O(K) de ODE solvers
  - Kernel fusível pelo XLA (backbone → heads → gating em uma passada)
  - Parâmetros reduzidos → cabe no Infinity Cache de 96MB

Referência: https://arxiv.org/abs/2106.13898
"""

from typing import Callable

import jax.numpy as jnp
import flax.linen as nn

from src.core.types import Array


class CfCCell(nn.Module):
    """
    Célula neural Closed-form Continuous-time (CfC).

    Implementa o mecanismo de gating temporal que permite ao neurônio
    adaptar sua constante de tempo dinamicamente — "acelerando" para
    estímulos fortes e "desacelerando" para integração lenta.

    Attributes:
        hidden_size: Dimensão do estado oculto.
        backbone_features: Dimensão do backbone compartilhado.
                          Se None, usa hidden_size.
        backbone_activation: Função de ativação do backbone.
    """
    hidden_size: int
    backbone_features: int | None = None
    backbone_activation: Callable = nn.tanh

    @nn.compact
    def __call__(
        self,
        h: Array,
        input_signal: Array,
        time_delta: Array,
    ) -> tuple[Array, Array]:
        """
        Aplica a atualização CfC de forma fechada.

        Args:
            h: Estado oculto anterior — shape (batch, hidden_size)
            input_signal: Entrada sensorial — shape (batch, input_dim)
            time_delta: Passo de tempo físico — shape (batch, 1) ou escalar

        Returns:
            Tupla (h_new, h_new) para compatibilidade com nn.scan.
        """
        backbone_dim = self.backbone_features or self.hidden_size

        # ── Concatenação estado + entrada ──
        input_state = jnp.concatenate([h, input_signal], axis=-1)

        # ── Backbone compartilhado (extração de features) ──
        backbone = nn.Dense(
            backbone_dim,
            name="backbone",
            kernel_init=nn.initializers.lecun_normal(),
        )(input_state)
        backbone = self.backbone_activation(backbone)

        # ── Cabeças (heads) da fórmula CfC ──

        # ff1: componente de estado/decaimento
        ff1 = nn.Dense(
            self.hidden_size,
            name="ff1",
            kernel_init=nn.initializers.lecun_normal(),
        )(backbone)

        # ff2: componente de entrada/equilíbrio
        ff2 = nn.Dense(
            self.hidden_size,
            name="ff2",
            kernel_init=nn.initializers.lecun_normal(),
        )(backbone)

        # time_gate: controla a constante de tempo líquida
        time_gate_raw = nn.Dense(
            self.hidden_size,
            name="time_gate",
            kernel_init=nn.initializers.lecun_normal(),
        )(backbone)

        # ── Mecanismo de Gating Temporal ──
        # σ(time_gate · Δt) modula a mistura entre ff1 e ff2
        # Quando Δt é grande ou o gate é alto → transição rápida
        # Quando Δt é pequeno ou o gate é baixo → preserva memória
        time_delta = jnp.broadcast_to(
            time_delta,
            time_gate_raw.shape,
        )
        gate = nn.sigmoid(time_gate_raw * time_delta)

        # ── Atualização de Estado (fórmula CfC fechada) ──
        # h_new = gate ⊙ ff1 + (1 - gate) ⊙ ff2
        h_new = gate * ff1 + (1.0 - gate) * ff2

        return h_new, h_new

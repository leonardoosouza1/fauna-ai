"""
Plasticidade Hebbiana — Projeto Fauna.

Implementa mecanismos de plasticidade sináptica inspirados na regra de Hebb:
    "Neurons that fire together, wire together."

Na IARA v10, a plasticidade opera em 3 escalas:
  1. STDP local: Ajuste de pesos sinápticos baseado em co-ativação
  2. Modulação de learning rate: Meta-aprendizado via genoma
  3. Homeostase: Normalização para evitar runaway excitation

A plasticidade permite aprendizado DURANTE a vida do agente,
complementando a evolução que opera ENTRE gerações.
Essa dupla adaptação possibilita o Efeito Baldwin:
comportamentos aprendidos se tornam instintivos ao longo das gerações.

Referência: Hasani et al. (2022) — CfC + plasticity meta-parameters
"""

import jax
import jax.numpy as jnp

from src.core.types import Array, DTYPE_COMPUTE


def hebbian_update(
    pre_activation: Array,
    post_activation: Array,
    weights: Array,
    learning_rate: float = 0.001,
    decay: float = 0.999,
) -> Array:
    """
    Atualiza pesos sinápticos pela regra de Hebb com decaimento.

    Fórmula:
        ΔW = η * (pre ⊗ post) - λ * W
        W_new = W + ΔW

    Onde:
        - pre ⊗ post: produto externo (co-ativação)
        - λ: fator de decaimento (previne crescimento ilimitado)

    Args:
        pre_activation: (batch, pre_dim) — ativação pré-sináptica
        post_activation: (batch, post_dim) — ativação pós-sináptica
        weights: (pre_dim, post_dim) — pesos sinápticos atuais
        learning_rate: Taxa de aprendizado (η)
        decay: Fator de decaimento (λ)

    Returns:
        new_weights: (pre_dim, post_dim) — pesos atualizados
    """
    # Produto externo médio sobre o batch (co-ativação)
    # (batch, pre, 1) * (batch, 1, post) → (batch, pre, post) → mean → (pre, post)
    hebbian_term = jnp.mean(
        pre_activation[:, :, None] * post_activation[:, None, :],
        axis=0,
    )

    # Atualização com decaimento
    delta_w = learning_rate * hebbian_term - (1.0 - decay) * weights
    new_weights = weights + delta_w

    return new_weights


def oja_update(
    pre_activation: Array,
    post_activation: Array,
    weights: Array,
    learning_rate: float = 0.001,
) -> Array:
    """
    Regra de Oja — Hebb normalizado (PCA online).

    Previne crescimento ilimitado dos pesos automaticamente:
        ΔW = η * post * (pre - W^T * post)

    Mais estável que Hebb puro para redes recorrentes.

    Args:
        pre_activation: (batch, pre_dim)
        post_activation: (batch, post_dim)
        weights: (pre_dim, post_dim)
        learning_rate: Taxa de aprendizado

    Returns:
        new_weights: (pre_dim, post_dim)
    """
    # Reconstrução: W^T * post → (batch, pre_dim)
    reconstruction = post_activation @ weights.T

    # Erro de reconstrução: pre - reconstruction
    error = pre_activation - reconstruction

    # Produto externo do erro com post
    delta_w = learning_rate * jnp.mean(
        error[:, :, None] * post_activation[:, None, :],
        axis=0,
    )

    return weights + delta_w


def stdp_update(
    pre_trace: Array,
    post_trace: Array,
    weights: Array,
    a_plus: float = 0.01,
    a_minus: float = 0.012,
    tau_plus: float = 20.0,
    tau_minus: float = 20.0,
    dt: float = 1.0,
) -> tuple[Array, Array, Array]:
    """
    Spike-Timing-Dependent Plasticity (STDP) simplificado.

    Atualiza traces de atividade e pesos baseado na temporalidade
    relativa de ativação pré e pós sináptica.

    A janela temporal assimétrica (a_minus > a_plus) cria
    uma pressão natural para esparsificação das conexões.

    Args:
        pre_trace: (pre_dim,) — trace de atividade pré-sináptica
        post_trace: (post_dim,) — trace de atividade pós-sináptica
        weights: (pre_dim, post_dim)
        a_plus: Amplitude de potenciação (LTP)
        a_minus: Amplitude de depressão (LTD)
        tau_plus/tau_minus: Constantes de tempo dos traces
        dt: Passo temporal

    Returns:
        new_weights, new_pre_trace, new_post_trace
    """
    # Decaimento exponencial dos traces
    decay_plus = jnp.exp(-dt / tau_plus)
    decay_minus = jnp.exp(-dt / tau_minus)

    new_pre_trace = pre_trace * decay_plus
    new_post_trace = post_trace * decay_minus

    # LTP: pré antes de pós → fortalece
    ltp = a_plus * new_pre_trace[:, None] * new_post_trace[None, :]

    # LTD: pós antes de pré → enfraquece
    ltd = -a_minus * new_post_trace[None, :] * new_pre_trace[:, None]

    new_weights = weights + ltp + ltd

    return new_weights, new_pre_trace, new_post_trace


def homeostatic_scaling(
    weights: Array,
    target_activity: float = 0.5,
    current_activity: float = 0.5,
    scaling_rate: float = 0.001,
) -> Array:
    """
    Escalonamento homeostático para manter atividade estável.

    Se a atividade média está muito alta → reduz pesos
    Se está muito baixa → aumenta pesos

    Previne runaway excitation e silent death de neurônios.

    Args:
        weights: Pesos sinápticos a escalar
        target_activity: Nível de atividade desejado
        current_activity: Atividade observada
        scaling_rate: Velocidade de ajuste

    Returns:
        scaled_weights: Pesos ajustados
    """
    scaling_factor = 1.0 + scaling_rate * (target_activity - current_activity)
    return weights * scaling_factor


def apply_plasticity_mask(
    weight_update: Array,
    plasticity_mask: Array,
    plasticity_rate: float = 1.0,
) -> Array:
    """
    Aplica máscara de plasticidade do genoma à atualização de pesos.

    Cada sinapse tem uma taxa de plasticidade definida geneticamente.
    Isso permite que a evolução controle QUAIS conexões são plásticas
    e com qual intensidade — implementando o meta-aprendizado.

    Args:
        weight_update: (pre_dim, post_dim) — ΔW proposto
        plasticity_mask: (pre_dim, post_dim) — máscara [0, 1] do genoma
        plasticity_rate: Escalar global de plasticidade

    Returns:
        masked_update: ΔW com plasticidade aplicada
    """
    return weight_update * plasticity_mask * plasticity_rate


def compute_activity_stats(activations: Array) -> dict:
    """
    Calcula estatísticas de atividade neural para monitoramento.

    Args:
        activations: (batch, neurons) — ativações de uma camada

    Returns:
        Dict com mean, std, sparsity, max_activation
    """
    mean_activity = jnp.mean(jnp.abs(activations))
    std_activity = jnp.std(activations)
    sparsity = jnp.mean(jnp.abs(activations) < 0.1)
    max_activation = jnp.max(jnp.abs(activations))

    return {
        "mean_activity": mean_activity,
        "std_activity": std_activity,
        "sparsity": sparsity,
        "max_activation": max_activation,
    }

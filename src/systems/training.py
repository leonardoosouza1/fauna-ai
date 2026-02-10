"""
Loop de Treinamento PPO — Projeto Fauna.

Implementa Proximal Policy Optimization para treinar as políticas
dos agentes (AgentBrain) usando as experiências do ambiente.

O PPO é usado em vez de evolução pura quando queremos treinar
agentes rapidamente dentro de uma geração (online learning).

Características:
  - Advantage estimation com GAE (Generalized Advantage Estimation)
  - Clipping de ratio para atualizações estáveis
  - Entropia da política como regularizador (incentiva exploração)
  - Compatível com jax.jit e vmap

Nota: O PPO pode operar em conjunto com a evolução:
  - PPO: otimiza params DENTRO de uma vida
  - Evolução: otimiza genoma/topologia ENTRE vidas
"""

import jax
import jax.numpy as jnp
import optax

from src.core.types import Array, PRNGKey, DTYPE_COMPUTE


# ──────────────────────────────────────────────
# Estruturas de Dados
# ──────────────────────────────────────────────

class RolloutBuffer:
    """
    Armazena transições para batch training.

    Estrutura flat compatível com JAX pytrees.
    """
    def __init__(
        self,
        max_steps: int,
        obs_dim: int,
        action_dim: int,
        max_agents: int,
    ):
        self.observations = jnp.zeros((max_steps, max_agents, obs_dim))
        self.actions = jnp.zeros((max_steps, max_agents, action_dim))
        self.rewards = jnp.zeros((max_steps, max_agents))
        self.values = jnp.zeros((max_steps, max_agents))
        self.log_probs = jnp.zeros((max_steps, max_agents))
        self.dones = jnp.zeros((max_steps, max_agents), dtype=jnp.bool_)
        self.step = 0
        self.max_steps = max_steps

    def add(
        self,
        obs: Array,
        action: Array,
        reward: Array,
        value: Array,
        log_prob: Array,
        done: Array,
    ) -> "RolloutBuffer":
        """Adiciona uma transição ao buffer."""
        idx = self.step % self.max_steps
        self.observations = self.observations.at[idx].set(obs)
        self.actions = self.actions.at[idx].set(action)
        self.rewards = self.rewards.at[idx].set(reward)
        self.values = self.values.at[idx].set(value)
        self.log_probs = self.log_probs.at[idx].set(log_prob)
        self.dones = self.dones.at[idx].set(done)
        self.step += 1
        return self


# ──────────────────────────────────────────────
# GAE (Generalized Advantage Estimation)
# ──────────────────────────────────────────────

def compute_gae(
    rewards: Array,
    values: Array,
    dones: Array,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[Array, Array]:
    """
    Calcula vantagens usando GAE (Schulman et al., 2015).

    Args:
        rewards: (T, N) — recompensas por step por agente
        values: (T, N) — estimativas de valor do critic
        dones: (T, N) — flags de episódio terminado
        gamma: Fator de desconto temporal
        gae_lambda: Fator de suavização GAE (1.0 = Monte Carlo, 0.0 = TD(0))

    Returns:
        advantages: (T, N)
        returns: (T, N) — targets para a value function
    """
    T, N = rewards.shape
    advantages = jnp.zeros_like(rewards)
    last_gae = jnp.zeros(N)

    # Compute backwards (GAE requer iteração reversa)
    def _scan_step(carry, t):
        last_gae = carry
        # TD error
        next_value = jnp.where(t < T - 1, values[t + 1], 0.0)
        not_done = 1.0 - dones[t].astype(jnp.float32)
        delta = rewards[t] + gamma * next_value * not_done - values[t]

        # GAE
        new_gae = delta + gamma * gae_lambda * not_done * last_gae
        return new_gae, new_gae

    # Scan reverso
    reversed_indices = jnp.arange(T - 1, -1, -1)
    _, reversed_advantages = jax.lax.scan(_scan_step, last_gae, reversed_indices)
    advantages = reversed_advantages[::-1]  # Reverter de volta

    # Returns = advantages + values
    returns = advantages + values

    return advantages, returns


# ──────────────────────────────────────────────
# PPO Loss
# ──────────────────────────────────────────────

def ppo_loss(
    log_probs_new: Array,
    log_probs_old: Array,
    advantages: Array,
    values_pred: Array,
    returns: Array,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    entropy: Array | None = None,
) -> tuple[Array, dict]:
    """
    Calcula a loss PPO-Clip.

    Componentes:
      1. Policy loss: clipped ratio * advantage
      2. Value loss: MSE entre value prediction e returns
      3. Entropy bonus: incentiva exploração

    Args:
        log_probs_new: (T, N) — log prob das ações sob política nova
        log_probs_old: (T, N) — log prob das ações sob política antiga
        advantages: (T, N) — GAE advantages
        values_pred: (T, N) — predição da value function
        returns: (T, N) — targets de retorno
        clip_epsilon: Epsilon do clipping PPO
        value_coef: Peso da value loss
        entropy_coef: Peso do bônus de entropia
        entropy: (T, N) — entropia da política (opcional)

    Returns:
        total_loss: Escalar
        metrics: Dict com componentes da loss
    """
    # Normalizar advantages
    adv_mean = jnp.mean(advantages)
    adv_std = jnp.std(advantages) + 1e-8
    advantages_norm = (advantages - adv_mean) / adv_std

    # Ratio de probabilidade
    ratio = jnp.exp(log_probs_new - log_probs_old)

    # Clipped surrogate
    surr1 = ratio * advantages_norm
    surr2 = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_norm
    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

    # Value loss
    value_loss = jnp.mean((values_pred - returns) ** 2)

    # Entropy bonus
    entropy_loss = 0.0
    if entropy is not None:
        entropy_loss = -jnp.mean(entropy)

    # Total
    total_loss = (
        policy_loss
        + value_coef * value_loss
        + entropy_coef * entropy_loss
    )

    metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": -entropy_loss if entropy is not None else 0.0,
        "mean_ratio": jnp.mean(ratio),
        "clip_fraction": jnp.mean(jnp.abs(ratio - 1.0) > clip_epsilon),
    }

    return total_loss, metrics


# ──────────────────────────────────────────────
# Optimizer
# ──────────────────────────────────────────────

def create_optimizer(
    learning_rate: float = 3e-4,
    max_grad_norm: float = 0.5,
) -> optax.GradientTransformation:
    """
    Cria optimizer Adam com gradient clipping.

    Args:
        learning_rate: Taxa de aprendizado
        max_grad_norm: Norma máxima do gradiente

    Returns:
        optax.GradientTransformation
    """
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate),
    )


def compute_reward(
    energy_before: Array,
    energy_after: Array,
    is_active: Array,
    food_consumed: Array,
) -> Array:
    """
    Calcula recompensa por agent por step.

    Reward shaping:
      + Comida consumida: incentiva busca de recursos
      + Energia mantida: incentiva eficiência
      - Morte: penalidade forte

    Args:
        energy_before: (max_agents,) — energia antes do step
        energy_after: (max_agents,) — energia depois do step
        is_active: (max_agents,) — status de vida
        food_consumed: (max_agents,) — booleano de alimentação

    Returns:
        rewards: (max_agents,)
    """
    # Recompensa pela sobrevivência
    survival_reward = is_active.astype(jnp.float32) * 0.01

    # Recompensa por comida
    food_reward = food_consumed.astype(jnp.float32) * 1.0

    # Penalidade por perda de energia
    energy_delta = (energy_after - energy_before) / 100.0

    # Penalidade por morte
    death_penalty = jnp.where(
        (energy_before > 0) & (energy_after <= 0),
        -5.0,
        0.0,
    )

    reward = survival_reward + food_reward + energy_delta + death_penalty
    return reward * is_active.astype(jnp.float32)

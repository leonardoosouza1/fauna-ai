"""
Global Workspace Theory (GWT) — Projeto Fauna.

Implementa o mecanismo de atenção seletiva inspirado na teoria do
espaço de trabalho global de Bernard Baars (1988).

Na IARA v10, o GWT funciona como o "teatro da consciência":
  - Múltiplos módulos especialistas (workers) competem por atenção
  - O vencedor "transmite" seu sinal para todos os outros módulos
  - Isso cria um gargalo computacional que força priorização

Arquitetura:
  1. WorkerEmbeddings → Cross-Attention → Consensus
  2. Consensus → Confidence Gating → Broadcast
  3. Broadcast → Modulation de h_inter/h_command

Referência saci-trader: CrossAttentionFusion + ConsensusFusion
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from src.core.types import Array, DTYPE_COMPUTE


class CrossAttentionFusion(nn.Module):
    """
    Fusão por cross-attention entre módulos especialistas.

    Cada embedding de worker atende a todos os outros workers
    para determinar a importância relativa, criando um "voto"
    ponderado pelo consenso do grupo.

    Attributes:
        embed_dim: Dimensão dos embeddings de entrada.
        num_heads: Número de cabeças de atenção.
        dropout_rate: Taxa de dropout para regularização.
    """
    embed_dim: int
    num_heads: int = 4
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self, x: Array, deterministic: bool = True,
    ) -> Array:
        """
        Args:
            x: Worker embeddings — (batch, num_workers, embed_dim)
            deterministic: Se True, desativa dropout.

        Returns:
            Context-aware embeddings — (batch, num_workers, embed_dim)
        """
        # Multi-head self-attention
        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
            name="mha",
        )(x, x)

        # Residual + LayerNorm
        x = nn.LayerNorm(name="ln")(x + attended)

        return x


class ConsensusFusion(nn.Module):
    """
    Funde embeddings de workers + confidence scores em um vetor de contexto.

    Combina:
      1. Attended embeddings (o que cada worker "pensa")
      2. Confidence weights (quão certo cada worker está)

    O resultado é o vetor do Global Workspace — o "pensamento consciente"
    que é transmitido para todos os subsistemas.

    Attributes:
        num_workers: Número de módulos especialistas.
        worker_dim: Dimensão do embedding de cada worker.
        output_dim: Dimensão do vetor de contexto de saída.
    """
    num_workers: int
    worker_dim: int
    output_dim: int

    @nn.compact
    def __call__(
        self,
        embeddings: Array,
        confidence_scores: Array,
    ) -> Array:
        """
        Args:
            embeddings: (batch, num_workers, worker_dim)
            confidence_scores: (batch, num_workers) — logits de confiança

        Returns:
            context: (batch, output_dim)
        """
        # Attention weights baseados na confiança
        attention_weights = nn.softmax(confidence_scores, axis=-1)  # (batch, num_workers)

        # Weighted sum dos embeddings
        # (batch, num_workers, 1) * (batch, num_workers, worker_dim) → sum → (batch, worker_dim)
        weighted = attention_weights[:, :, None] * embeddings
        pooled = jnp.sum(weighted, axis=1)  # (batch, worker_dim)

        # Projeção para dimensão de saída
        context = nn.Dense(
            self.output_dim,
            name="context_proj",
            kernel_init=nn.initializers.lecun_normal(),
        )(pooled)
        context = nn.tanh(context)

        return context


class GlobalWorkspace(nn.Module):
    """
    Global Workspace — integra attention fusion e consensus.

    Pipeline:
        Worker Embeddings → CrossAttention → ConsensusFusion → Broadcast Vector

    O vetor de broadcast modula o processamento nas camadas
    intermediária e de comando da NCP.

    Attributes:
        num_workers: Número de especialistas.
        worker_dim: Dimensão do embedding por worker.
        workspace_dim: Dimensão do espaço de trabalho global.
        num_heads: Cabeças de atenção do cross-attention.
    """
    num_workers: int
    worker_dim: int
    workspace_dim: int = 16
    num_heads: int = 4

    @nn.compact
    def __call__(
        self,
        worker_embeddings: Array,
        worker_confidences: Array,
        deterministic: bool = True,
    ) -> tuple[Array, Array]:
        """
        Args:
            worker_embeddings: (batch, num_workers, worker_dim)
            worker_confidences: (batch, num_workers)
            deterministic: Desativa dropout durante inferência.

        Returns:
            workspace_state: (batch, workspace_dim) — vetor de broadcast
            attention_weights: (batch, num_workers) — para interpretabilidade
        """
        # 1. Cross-Attention Fusion
        attended = CrossAttentionFusion(
            embed_dim=self.worker_dim,
            num_heads=self.num_heads,
            name="cross_attn",
        )(worker_embeddings, deterministic=deterministic)

        # 2. Consensus Fusion (com confidence gating)
        workspace_state = ConsensusFusion(
            num_workers=self.num_workers,
            worker_dim=self.worker_dim,
            output_dim=self.workspace_dim,
            name="consensus",
        )(attended, worker_confidences)

        # 3. Retornar attention weights para interpretabilidade
        attention_weights = nn.softmax(worker_confidences, axis=-1)

        return workspace_state, attention_weights


class WorkspaceModulator(nn.Module):
    """
    Modula o processamento neural usando o sinal do Global Workspace.

    Aplica o vetor de workspace como bias multiplicativo/aditivo
    nos estados ocultos da NCP, similar ao mecanismo de
    modulação LFM→LNN no saci-trader (TrinityLiquidBrain).

    Attributes:
        target_dim: Dimensão do alvo a modular (num_inter ou num_command).
    """
    target_dim: int

    @nn.compact
    def __call__(
        self,
        hidden_state: Array,
        workspace_state: Array,
    ) -> Array:
        """
        Args:
            hidden_state: (batch, target_dim) — estado CfC a modular
            workspace_state: (batch, workspace_dim) — sinal GWT

        Returns:
            modulated: (batch, target_dim) — estado modulado
        """
        # Projetar workspace para a dimensão do alvo
        gate = nn.Dense(
            self.target_dim,
            name="gate_proj",
            kernel_init=nn.initializers.lecun_normal(),
        )(workspace_state)
        gate = nn.sigmoid(gate)  # Gate multiplicativo [0, 1]

        bias = nn.Dense(
            self.target_dim,
            name="bias_proj",
            kernel_init=nn.initializers.lecun_normal(),
        )(workspace_state)

        # Modulação: gate controla quanto o workspace influencia
        modulated = hidden_state * gate + bias

        return modulated

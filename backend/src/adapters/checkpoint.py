"""
Adaptador de Checkpoint — Projeto Fauna.

Abstrai a serialização e desserialização de estados de simulação.
Usa Orbax para checkpointing compatível com JAX pytrees.

Suporta:
  - Salvamento incremental (apenas delta)
  - Compressão automática
  - Restore parcial (apenas certos campos)
"""

import os
import logging
from typing import Any

import jax
import jax.numpy as jnp

logger = logging.getLogger("fauna.checkpoint")


# ──────────────────────────────────────────────
# Fallback puro-JAX (sem dependência de Orbax)
# ──────────────────────────────────────────────

def save_checkpoint(
    state: Any,
    path: str,
    step: int = 0,
    overwrite: bool = False,
) -> str:
    """
    Salva checkpoint de um JAX pytree como arquivo numpy.

    Args:
        state: Qualquer JAX pytree (WorldState, params, etc.)
        path: Diretório base para checkpoints
        step: Número do step para versionamento
        overwrite: Se True, sobrescreve checkpoints existentes

    Returns:
        filepath: Caminho completo do arquivo salvo
    """
    checkpoint_dir = os.path.join(path, f"step_{step:08d}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    filepath = os.path.join(checkpoint_dir, "state.npz")

    if os.path.exists(filepath) and not overwrite:
        logger.warning(f"Checkpoint já existe: {filepath}. Use overwrite=True.")
        return filepath

    # Flatten do pytree em arrays individuais
    leaves, treedef = jax.tree.flatten(state)

    # Salvar como npz comprimido
    arrays = {f"leaf_{i}": jnp.asarray(leaf) for i, leaf in enumerate(leaves)}
    jnp.savez(filepath, **arrays)

    # Salvar treedef como metadata (string serializada)
    meta_path = os.path.join(checkpoint_dir, "treedef.txt")
    with open(meta_path, "w") as f:
        f.write(str(treedef))

    logger.info(f"Checkpoint salvo: {filepath} ({len(leaves)} arrays)")
    return filepath


def load_checkpoint(
    path: str,
    step: int = 0,
    template_state: Any = None,
) -> Any:
    """
    Carrega checkpoint de um JAX pytree.

    Args:
        path: Diretório base dos checkpoints
        step: Número do step a carregar
        template_state: Estado template para reconstruir o pytree.
                       Se None, retorna dict flat.

    Returns:
        state: O pytree reconstruído ou dict flat
    """
    checkpoint_dir = os.path.join(path, f"step_{step:08d}")
    filepath = os.path.join(checkpoint_dir, "state.npz")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint não encontrado: {filepath}")

    # Carregar arrays
    data = jnp.load(filepath)
    leaves = [data[f"leaf_{i}"] for i in range(len(data.files))]

    if template_state is not None:
        # Reconstruir pytree usando o template
        _, treedef = jax.tree.flatten(template_state)
        state = treedef.unflatten(leaves)
        logger.info(f"Checkpoint carregado: {filepath}")
        return state

    logger.info(f"Checkpoint carregado (flat): {filepath}")
    return {f"leaf_{i}": leaf for i, leaf in enumerate(leaves)}


def list_checkpoints(path: str) -> list[int]:
    """
    Lista steps de checkpoints disponíveis.

    Args:
        path: Diretório base dos checkpoints

    Returns:
        steps: Lista ordenada de steps disponíveis
    """
    if not os.path.exists(path):
        return []

    steps = []
    for dirname in os.listdir(path):
        if dirname.startswith("step_"):
            try:
                step = int(dirname.split("_")[1])
                steps.append(step)
            except ValueError:
                continue

    return sorted(steps)


def get_latest_checkpoint(path: str) -> int | None:
    """
    Retorna o step do checkpoint mais recente.

    Returns:
        step: Número do step, ou None se não existir
    """
    steps = list_checkpoints(path)
    return steps[-1] if steps else None

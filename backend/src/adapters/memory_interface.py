"""
Interface de Memória Associativa — Projeto Fauna.

Define o Protocol (contrato) para sistemas de memória vetorial.
O código de simulação depende apenas desta interface, nunca da
implementação concreta (FAISS), seguindo a Clean Architecture.

Implementação concreta: src/infrastructure/faiss_memory.py (Fase 4)
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class MemoryStore(Protocol):
    """
    Contrato para armazenamento de memória vetorial.

    Reside na CPU (RAM) para economizar VRAM da GPU.
    Comunicação via jax.pure_callback no loop de simulação.
    """

    def add(self, agent_id: int, embedding: np.ndarray) -> None:
        """
        Armazena um vetor de memória para um agente.

        Args:
            agent_id: Identificador do agente.
            embedding: Vetor de embedding — shape (dim,).
        """
        ...

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        agent_id: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Busca os k vizinhos mais próximos.

        Args:
            query: Vetor de busca — shape (dim,).
            k: Número de vizinhos a retornar.
            agent_id: Se especificado, busca apenas memórias deste agente.

        Returns:
            Tupla (distances, indices) — shapes (k,) e (k,).
        """
        ...

    def size(self) -> int:
        """Retorna o número total de vetores armazenados."""
        ...

    def clear(self, agent_id: int | None = None) -> None:
        """
        Remove memórias. Se agent_id=None, limpa tudo.

        Args:
            agent_id: Se especificado, remove apenas memórias deste agente.
        """
        ...

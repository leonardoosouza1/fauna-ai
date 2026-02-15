"""
Data Exporter — Projeto Fauna.

Converte estado da simulação em JSON para o dashboard web.
Escreve periodicamente o estado em um arquivo que o dashboard lê via polling.
"""

import json
import os
import time
import logging
from typing import Any

import jax.numpy as jnp

logger = logging.getLogger("fauna.exporter")

# Diretório padrão para dados exportados
DEFAULT_DATA_DIR = "simulation_data"


class SimulationExporter:
    """
    Exporta dados da simulação para consumo pelo dashboard web.

    Mantém histórico de métricas e exporta snapshots do mundo.
    """

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.history: list[dict] = []
        self.generation_history: list[dict] = []
        self.start_time = time.time()

    def export_step(
        self,
        step: int,
        positions: Any,
        energies: Any,
        ages: Any,
        is_active: Any,
        species_ids: Any,  # [NEW]
        masses: Any,       # [NEW]
        food_grid: Any,
        metrics: dict,
        generation: int = 0,
        generation_stats: dict | None = None,
        world_bounds: float = 500.0,
    ) -> None:
        """Exporta o estado de um step para JSON."""
        # Converter JAX arrays para listas Python
        active_mask = jnp.array(is_active, dtype=bool)
        num_active = int(jnp.sum(active_mask))

        # Dados dos agentes ativos
        agents_data = []
        pos_np = jnp.array(positions)
        energy_np = jnp.array(energies)
        age_np = jnp.array(ages)
        species_np = jnp.array(species_ids)
        mass_np = jnp.array(masses)

        for i in range(len(active_mask)):
            if active_mask[i]:
                agents_data.append({
                    "id": int(i),
                    "x": float(pos_np[i, 0]),
                    "y": float(pos_np[i, 1]),
                    "energy": float(energy_np[i]),
                    "age": int(age_np[i]),
                    "sp": int(species_np[i]),  # species_id
                    "m": float(mass_np[i]),    # mass
                })

        # Food grid (downsample para performance)
        food_np = jnp.array(food_grid)
        grid_h, grid_w = food_np.shape
        # Downsample se muito grande
        if grid_h > 32:
            step_size = grid_h // 32
            food_np = food_np[::step_size, ::step_size]
        food_list = food_np.tolist()

        # Métricas do step
        step_metrics = {
            "step": step,
            "timestamp": time.time() - self.start_time,
            "alive": num_active,
            "mean_energy": metrics.get("mean_energy", 0.0),
            "mean_age": metrics.get("mean_age", 0.0),
            "food_total": metrics.get("food_total", 0.0),
            "food_consumed": metrics.get("food_consumed", 0),
            "generation": generation,
        }
        self.history.append(step_metrics)

        if generation_stats:
            self.generation_history.append(generation_stats)

        # Construir snapshot completo
        snapshot = {
            "step": step,
            "generation": generation,
            "elapsed": time.time() - self.start_time,
            "agents": agents_data,
            "food_grid": food_list,
            "metrics": step_metrics,
            "history": self.history[-200:],  # Últimos 200 pontos
            "generation_history": self.generation_history,
            "world_bounds": world_bounds,
            "status": "running",
        }

        # Escrever atomicamente (write + rename)
        tmp_path = os.path.join(self.data_dir, "state.json.tmp")
        final_path = os.path.join(self.data_dir, "state.json")

        try:
            with open(tmp_path, "w") as f:
                json.dump(snapshot, f)
            os.replace(tmp_path, final_path)
        except Exception as e:
            logger.warning(f"Falha ao exportar: {e}")

    def export_final(self, step: int, status: str = "completed") -> None:
        """Marca a simulação como finalizada."""
        final_path = os.path.join(self.data_dir, "state.json")
        try:
            with open(final_path, "r") as f:
                data = json.load(f)
            data["status"] = status
            data["final_step"] = step
            with open(final_path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

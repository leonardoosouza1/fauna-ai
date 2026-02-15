"""
Validação do Ambiente ROCm/JAX — Projeto Fauna.

Script de diagnóstico que verifica:
  1. Backend JAX ativo (CPU vs GPU)
  2. Dispositivos disponíveis
  3. Benchmark de matmul (performance)
  4. Estimativa de VRAM disponível

Uso:
    python -m src.infrastructure.rocm_check
"""

import sys
import time

import jax
import jax.numpy as jnp


def check_backend() -> str:
    """Verifica qual backend o JAX está utilizando."""
    devices = jax.devices()
    backend = devices[0].platform if devices else "unknown"
    return backend


def get_device_info() -> list[dict]:
    """Retorna informações sobre todos os dispositivos disponíveis."""
    return [
        {
            "id": d.id,
            "platform": d.platform,
            "device_kind": d.device_kind,
        }
        for d in jax.devices()
    ]


def benchmark_matmul(size: int = 1024, num_warmup: int = 3, num_runs: int = 10) -> dict:
    """
    Executa benchmark de multiplicação de matrizes.

    Args:
        size: Dimensão da matriz quadrada (size × size).
        num_warmup: Iterações de aquecimento (compilação JIT).
        num_runs: Iterações de medição.

    Returns:
        Dicionário com tempo médio, TFLOPs e status.
    """
    rng = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(rng)

    # Criar matrizes no device
    a = jax.random.normal(k1, (size, size), dtype=jnp.float32)
    b = jax.random.normal(k2, (size, size), dtype=jnp.float32)

    # Função JIT-compilada
    @jax.jit
    def matmul(x, y):
        return x @ y

    # Warmup (compilação JIT)
    for _ in range(num_warmup):
        result = matmul(a, b)
        result.block_until_ready()

    # Medição
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = matmul(a, b)
        result.block_until_ready()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)

    # Calcular TFLOPs (2 * N^3 operações para matmul)
    flops = 2 * (size ** 3)
    tflops = (flops / avg_time) / 1e12

    return {
        "matrix_size": size,
        "avg_time_ms": avg_time * 1000,
        "tflops": tflops,
        "num_runs": num_runs,
    }


def estimate_vram_capacity() -> dict | None:
    """
    Estima VRAM disponível alocando blocos crescentes.

    Retorna None se rodando em CPU.
    """
    if jax.devices()[0].platform != "gpu":
        return None

    allocated_mb = 0
    block_size_mb = 256
    blocks = []

    try:
        while True:
            block = jnp.zeros((block_size_mb * 1024 * 1024 // 4,), dtype=jnp.float32)
            block.block_until_ready()
            blocks.append(block)
            allocated_mb += block_size_mb
    except (RuntimeError, jax.errors.JaxRuntimeError):
        pass
    finally:
        del blocks

    return {"estimated_vram_mb": allocated_mb}


def run_diagnostics() -> None:
    """Executa todos os diagnósticos e imprime relatório."""
    print("=" * 60)
    print("  Projeto Fauna — Diagnóstico de Hardware")
    print("=" * 60)

    # Backend
    backend = check_backend()
    status = "✓ GPU (ROCm)" if backend == "gpu" else "⚠ CPU (sem aceleração)"
    print(f"\n  Backend JAX: {status}")

    # Dispositivos
    devices = get_device_info()
    print(f"\n  Dispositivos ({len(devices)}):")
    for d in devices:
        print(f"    [{d['id']}] {d['platform']} — {d['device_kind']}")

    # Benchmark
    print("\n  Benchmark matmul 1024×1024:")
    bench = benchmark_matmul(1024)
    print(f"    Tempo médio: {bench['avg_time_ms']:.2f} ms")
    print(f"    Performance: {bench['tflops']:.4f} TFLOPs")

    # VRAM
    if backend == "gpu":
        print("\n  Estimativa de VRAM:")
        vram = estimate_vram_capacity()
        if vram:
            print(f"    VRAM alocável: ~{vram['estimated_vram_mb']} MB")
    else:
        print("\n  ⚠ VRAM: N/A (rodando em CPU)")
        print("    Para GPU AMD, configure ROCm com:")
        print("    export HSA_OVERRIDE_GFX_VERSION=10.3.0")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_diagnostics()

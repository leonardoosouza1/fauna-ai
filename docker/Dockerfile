# ===== Projeto Fauna — Ambiente JAX/ROCm para RX 6750 XT =====
FROM rocm/jax:latest

LABEL maintainer="Engenharia Fauna"
LABEL description="Ambiente JAX/ROCm otimizado para RDNA2 (RX 6750 XT)"

# --- Variáveis de Ambiente Críticas para RDNA2 ---
# Força compatibilidade com kernels Navi 21 (gfx1030) na Navi 22 (gfx1031)
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
# Limita alocação de VRAM pelo XLA (85%) para evitar travamentos do display
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
# Otimizações de compilação XLA
ENV XLA_FLAGS="--xla_gpu_autotune_level=2"

# --- Dependências de Sistema ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    htop \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# --- Dependências Python ---
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

WORKDIR /app

CMD ["bash"]

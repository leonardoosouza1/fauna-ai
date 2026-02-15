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
# Permitir instalação de pacotes Python no sistema (Ubuntu 24.04+)
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# --- Dependências de Sistema ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    htop \
    libgl1 \
    libglx-mesa0 \
    lld \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -f render

# Corrigir caminhos do ROCm que estão quebrados na imagem base (apontam para 7.2 em vez de 7.2.0)
ENV ROCM_PATH=/opt/rocm
ENV LLVM_PATH=/opt/rocm/llvm

# --- Dependências Python ---
COPY backend/requirements.txt /tmp/requirements.txt
# Snapshot das versões instaladas (JAX 0.8.0, etc.) para evitar upgrade indesejado
RUN pip freeze > /tmp/constraints.txt && \
    pip install --no-cache-dir -r /tmp/requirements.txt -c /tmp/constraints.txt && \
    rm /tmp/requirements.txt /tmp/constraints.txt

WORKDIR /app

CMD ["bash"]

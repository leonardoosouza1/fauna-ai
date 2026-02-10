# 🧬 Projeto Fauna — Laboratório de Evolução Neuro-Líquida

Simulação de agentes autônomos com **Redes Neurais Líquidas (CfC/NCP)** em JAX/ROCm, otimizado para hardware AMD de consumo.

## 🎯 Visão Geral

O Projeto Fauna é um ecossistema digital onde agentes autônomos nascem, aprendem, evoluem e morrem em um ambiente físico contínuo. A cognição é baseada em **Closed-form Continuous-time (CfC) Networks** organizadas em topologia **Neural Circuit Policy (NCP)** — inspirada no conectoma do *C. elegans*.

### Arquitetura Cognitiva (IARA v10)

| Camada | Tipo | Função |
|--------|------|--------|
| Sensorial | MLP Encoder | Transdução de estímulos brutos |
| Intermediária | CfC (recorrente) | Working memory, dinâmicas temporais |
| Comando | CfC (inibição lateral) | Decisão de alto nível |
| Motora | Linear + tanh | Atuação física [-1, 1] |

## 📁 Estrutura

```
src/
├── core/           # Domínio — tipos e estados imutáveis (JAX pytrees)
├── systems/brain/  # Casos de Uso — CfC cell, NCP, AgentBrain
├── adapters/       # Interfaces — Protocol para memória vetorial
└── infrastructure/ # Infra — ROCm check, I/O
config/             # Hydra — hardware profiles, hiperparâmetros
docker/             # Docker — ROCm/RDNA2 container
tests/              # pytest — unitários
```

## 🚀 Quick Start

```bash
# Instalar dependências (CPU)
pip install -e ".[dev]"

# Rodar testes
python -m pytest tests/ -v

# Demo de simulação
python -m src.main

# Validação de hardware (GPU AMD)
python -m src.infrastructure.rocm_check
```

### Docker (GPU AMD ROCm)

```bash
cd docker
docker compose up -d
docker compose exec fauna bash
python -m src.infrastructure.rocm_check
```

## 🔧 Hardware Alvo

- **GPU**: AMD Radeon RX 6750 XT (12GB VRAM, 96MB Infinity Cache, 40 CUs)
- **CPU**: AMD Ryzen 5 5600 (6C/12T)
- **RAM**: 16GB DDR4
- **Stack**: JAX → XLA → ROCm 6.x → RDNA2

## 📋 Roadmap

- [x] **Fase 1**: Fundação (tipos, CfC cell, NCP, testes)
- [ ] **Fase 2**: MVP Física (Brax, sensores, metabolismo)
- [ ] **Fase 3**: IARA v10 (GWT, MoE, plasticidade Hebbiana, PPO)
- [ ] **Fase 4**: Escala (vmap multi-agente, FAISS, HyperNEAT, Baldwin)
- [ ] **Fase 5**: Otimização (Lyapunov, Transfer Entropy, visualizador)


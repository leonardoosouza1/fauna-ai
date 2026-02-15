/**
 * Projeto Fauna — Dashboard JavaScript
 * Renderização do mundo, gráficos, e polling de dados.
 */

// ============================================
// Constants
// ============================================
const POLL_INTERVAL = 500; // ms
const API_URL = '/api/state';

// ============================================
// State
// ============================================
let state = null;
let prevStep = -1;
let charts = {};
let trails = [];
let logEntries = [];

// ============================================
// DOM Elements
// ============================================
const canvas = document.getElementById('world-canvas');
const ctx = canvas.getContext('2d');

const elStatus = document.getElementById('status-badge');
const elStatusText = document.getElementById('status-text');
const elTimer = document.getElementById('timer');
const elStep = document.getElementById('step-counter');
const elAlive = document.getElementById('stat-alive');
const elEnergy = document.getElementById('stat-energy');
const elAge = document.getElementById('stat-age');
const elFood = document.getElementById('stat-food');
const elGeneration = document.getElementById('stat-generation');
const elFitness = document.getElementById('stat-fitness');
const elLog = document.getElementById('log-container');

const showFood = document.getElementById('show-food');
const showTrails = document.getElementById('show-trails');

document.getElementById('clear-log').addEventListener('click', () => {
    logEntries = [];
    elLog.innerHTML = '';
});

// ============================================
// Chart Setup
// ============================================
const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    plugins: {
        legend: { display: false },
    },
    scales: {
        x: {
            display: true,
            grid: { color: 'rgba(75, 85, 99, 0.15)' },
            ticks: { color: '#6b7280', font: { size: 10, family: 'JetBrains Mono' }, maxTicksLimit: 8 },
        },
        y: {
            display: true,
            grid: { color: 'rgba(75, 85, 99, 0.15)' },
            ticks: { color: '#6b7280', font: { size: 10, family: 'JetBrains Mono' }, maxTicksLimit: 5 },
            beginAtZero: true,
        },
    },
    elements: {
        point: { radius: 0 },
        line: { tension: 0.3, borderWidth: 2 },
    },
};

function initCharts() {
    charts.population = new Chart(
        document.getElementById('population-chart'),
        {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Agentes Vivos',
                    data: [],
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    fill: true,
                }],
            },
            options: { ...chartDefaults },
        },
    );

    charts.energy = new Chart(
        document.getElementById('energy-chart'),
        {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Energia Média',
                    data: [],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    fill: true,
                }],
            },
            options: { ...chartDefaults },
        },
    );

    charts.fitness = new Chart(
        document.getElementById('fitness-chart'),
        {
            type: 'bar',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Fitness Máximo',
                        data: [],
                        backgroundColor: 'rgba(168, 85, 247, 0.6)',
                        borderColor: '#a855f7',
                        borderWidth: 1,
                    },
                    {
                        label: 'Fitness Médio',
                        data: [],
                        backgroundColor: 'rgba(99, 102, 241, 0.4)',
                        borderColor: '#6366f1',
                        borderWidth: 1,
                    },
                ],
            },
            options: {
                ...chartDefaults,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: { color: '#9ca3af', font: { size: 10 } },
                    },
                },
            },
        },
    );
}

// ============================================
// World Rendering
// ============================================
function resizeCanvas() {
    const container = canvas.parentElement;
    const size = Math.min(container.clientWidth - 16, 800);
    canvas.width = size;
    canvas.height = size;
}

function worldToCanvas(x, y, bounds) {
    const scale = canvas.width / (bounds * 2);
    return {
        cx: (x + bounds) * scale,
        cy: (y + bounds) * scale,
    };
}

// ============================================
// Species Config
// ============================================
const SPECIES_COLORS = {
    1: '#10b981', // Preá (Green)
    2: '#3b82f6', // Veado (Blue)
    3: '#78350f', // Capivara (Brown)
    4: '#f97316', // Graxaim (Orange)
    5: '#eab308', // Jaguatirica (Yellow)
    6: '#ef4444', // Puma (Red)
};

function getAgentColor(agent) {
    // Se sp (species_id) existe, usa cor da espécie
    if (agent.sp && SPECIES_COLORS[agent.sp]) {
        return SPECIES_COLORS[agent.sp];
    }
    // Fallback para energia se não tiver sp (retrocompatibilidade)
    return energyToColorLegacy(agent.energy);
}

function energyToColorLegacy(energy) {
  if (energy > 120) return "#10b981";
  if (energy > 80) return "#22d3ee";
  if (energy > 40) return "#f59e0b";
  if (energy > 15) return "#f97316";
  return "#f43f5e";
}

function getAgentRadius(agent) {
  // Se tiver massa (m), escala com log da massa
  if (agent.m) {
    // Base 3px + log(m)*2. Ex: 0.5kg->~2px, 50kg->~10px
    return Math.max(3, 3 + Math.log(agent.m + 1) * 2);
  }
  return Math.max(3, Math.min(8, agent.energy / 20));
}

function renderWorld() {
  if (!state) return;

  // Ajusta bounds para 500m se new exporter, ou 50 legado
  const bounds = state.world_bounds || 500.0;

  // Auto-scaling se bounds mudou muito
  // (Opcional, mas worldToCanvas já usa bounds)

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Background
  const grad = ctx.createRadialGradient(
    canvas.width / 2,
    canvas.height / 2,
    0,
    canvas.width / 2,
    canvas.height / 2,
    canvas.width / 2,
  );
  grad.addColorStop(0, "#0c1120");
  grad.addColorStop(1, "#050810");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Grid lines (ajustado para visual clean em mapa grande)
  ctx.strokeStyle = "rgba(75, 85, 99, 0.1)";
  ctx.lineWidth = 0.5;
  const gridStep = canvas.width / 10;
  for (let i = 0; i <= 10; i++) {
    const pos = i * gridStep;
    ctx.beginPath();
    ctx.moveTo(pos, 0);
    ctx.lineTo(pos, canvas.height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, pos);
    ctx.lineTo(canvas.width, pos);
    ctx.stroke();
  }

  // Food grid / Vegetation
  // Se tiver dados de biomassa, desenhar (food_grid vira proxy visual)
  if (showFood.checked && state.food_grid) {
    const grid = state.food_grid;
    const rows = grid.length;
    const cols = grid[0] ? grid[0].length : 0;
    const cellW = canvas.width / cols;
    const cellH = canvas.height / rows;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = grid[r][c];
        if (val > 0.1) {
          // Verde para vegetação
          const alpha = Math.min(0.4, val / 5.0);
          ctx.fillStyle = `rgba(16, 185, 129, ${alpha})`;
          ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
        }
      }
    }
  }

  // Trails
  if (showTrails.checked && trails.length > 1) {
    ctx.globalAlpha = 0.3;
    for (const trail of trails) {
      if (trail.length < 2) continue;
      ctx.strokeStyle = "rgba(99, 102, 241, 0.3)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      const first = worldToCanvas(trail[0].x, trail[0].y, bounds);
      ctx.moveTo(first.cx, first.cy);
      for (let i = 1; i < trail.length; i++) {
        const p = worldToCanvas(trail[i].x, trail[i].y, bounds);
        ctx.lineTo(p.cx, p.cy);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1.0;
  }

  // Agents
  if (state.agents) {
    for (const agent of state.agents) {
      const { cx, cy } = worldToCanvas(agent.x, agent.y, bounds);
      const color = getAgentColor(agent);
      const radius = getAgentRadius(agent);

      // Glow indicator de energia/vida
      const energyAlpha = Math.min(0.5, agent.energy / 100.0);

      ctx.beginPath();
      ctx.arc(cx, cy, radius + 2, 0, Math.PI * 2);
      ctx.fillStyle = color.replace("#", "rgba(") + hexToRgb(color) + `, 0.2)`;
      // Hack simples para cor, melhor usar func dedicada se hex
      // Usando shadow blur para glow real
      ctx.shadowBlur = 10;
      ctx.shadowColor = color;
      ctx.fillStyle = color;
      ctx.fill();
      ctx.shadowBlur = 0;

      // Body
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }
  }

  // World border
  ctx.strokeStyle = "rgba(99, 102, 241, 0.3)";
  ctx.lineWidth = 2;
  ctx.strokeRect(1, 1, canvas.width - 2, canvas.height - 2);
}

// Helper para converter hex para "rrr, ggg, bbb" (aproximado)
function hexToRgb(hex) {
    // Remove #
    hex = hex.replace('#', '');
    const r = parseInt(hex.substring(0,2), 16);
    const g = parseInt(hex.substring(2,4), 16);
    const b = parseInt(hex.substring(4,6), 16);
    return `${r}, ${g}, ${b}`;
}

// ============================================
// Update UI
// ============================================
function updateStats() {
  if (!state || !state.metrics) return;
  const m = state.metrics;

  elStep.textContent = `Step ${m.step}`;
  // Show total alive + breakdown
  let popText = m.alive;
  if (m.prea_pop !== undefined) {
    popText += ` (P:${m.prea_pop} V:${m.veado_pop} Pred:${m.predator_pop})`;
  }
  elAlive.textContent = popText;
  elEnergy.textContent = (m.mean_energy || 0).toFixed(1);
  elAge.textContent = Math.round(m.mean_age || 0);
  elFood.textContent = Math.round(m.food_total || 0);
  elGeneration.textContent = m.generation || 0;

  // Fitness from generation history
  if (state.generation_history && state.generation_history.length > 0) {
    const latest =
      state.generation_history[state.generation_history.length - 1];
    elFitness.textContent = (latest.max_fitness || 0).toFixed(2);
  }

  // Timer
  if (state.elapsed != null) {
    const mins = Math.floor(state.elapsed / 60);
    const secs = Math.floor(state.elapsed % 60);
    elTimer.textContent = `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }
}

function updateStatus(status) {
    elStatus.className = 'status-badge';
    if (status === 'running') {
        elStatusText.textContent = 'Simulando';
    } else if (status === 'completed') {
        elStatusText.textContent = 'Finalizada';
        elStatus.classList.add('completed');
    } else if (status === 'waiting') {
        elStatusText.textContent = 'Aguardando...';
        elStatus.classList.add('offline');
    } else {
        elStatusText.textContent = status;
    }
}

function updateCharts() {
    if (!state || !state.history) return;

    const history = state.history;
    const labels = history.map(h => h.step);
    const alive = history.map(h => h.alive);
    const energy = history.map(h => h.mean_energy || 0);

    charts.population.data.labels = labels;
    charts.population.data.datasets[0].data = alive;
    charts.population.update();

    charts.energy.data.labels = labels;
    charts.energy.data.datasets[0].data = energy;
    charts.energy.update();

    // Fitness chart from generation history
    if (state.generation_history && state.generation_history.length > 0) {
        const genLabels = state.generation_history.map(g => `G${g.generation}`);
        const maxFit = state.generation_history.map(g => g.max_fitness || 0);
        const meanFit = state.generation_history.map(g => g.mean_fitness || 0);

        charts.fitness.data.labels = genLabels;
        charts.fitness.data.datasets[0].data = maxFit;
        charts.fitness.data.datasets[1].data = meanFit;
        charts.fitness.update();
    }
}

function addLog(message, type = 'info') {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    const time = state && state.elapsed != null
        ? `[${Math.floor(state.elapsed)}s] `
        : '';
    entry.textContent = `${time}${message}`;
    elLog.appendChild(entry);
    elLog.scrollTop = elLog.scrollHeight;
    logEntries.push({ message, type });

    // Limitar a 100 entradas
    while (elLog.children.length > 100) {
        elLog.removeChild(elLog.firstChild);
    }
}

// ============================================
// Trail Tracking
// ============================================
function updateTrails() {
    if (!state || !state.agents) return;

    // Manter últimas 30 posições por agente
    const maxTrailLen = 30;
    const agentMap = new Map();

    for (const agent of state.agents) {
        agentMap.set(agent.id, agent);
    }

    // Ajustar tamanho do array de trails
    while (trails.length < 200) trails.push([]);

    for (const [id, agent] of agentMap) {
        if (id < trails.length) {
            trails[id].push({ x: agent.x, y: agent.y });
            if (trails[id].length > maxTrailLen) {
                trails[id].shift();
            }
        }
    }
}

// ============================================
// Data Polling
// ============================================
async function fetchState() {
    try {
        const res = await fetch(API_URL);
        const data = await res.json();

        if (data.status === 'waiting') {
            updateStatus('waiting');
            return;
        }

        state = data;
        updateStatus(data.status || 'running');

        if (data.metrics && data.metrics.step !== prevStep) {
            const step = data.metrics.step;

            // Log de eventos significativos
            if (prevStep === -1) {
                addLog(`Simulação iniciada — ${data.agents?.length || 0} agentes`, 'success');
            }

            // Detectar mudanças de geração
            if (data.generation_history && data.generation_history.length > 0) {
                const latestGen = data.generation_history[data.generation_history.length - 1];
                if (latestGen.generation > (state._lastLoggedGen || 0)) {
                    addLog(
                        `🧬 Geração ${latestGen.generation} — ` +
                        `Pop: ${latestGen.population}, ` +
                        `Fitness: ${(latestGen.max_fitness || 0).toFixed(3)}`,
                        'generation',
                    );
                    state._lastLoggedGen = latestGen.generation;
                }
            }

            // Log periódico
            if (step % 200 === 0 && step > 0) {
                const m = data.metrics;
                addLog(
                    `Step ${step} — Alive: ${m.alive}, ` +
                    `Energy: ${(m.mean_energy || 0).toFixed(1)}, ` +
                    `Age: ${Math.round(m.mean_age || 0)}`,
                    'info',
                );
            }

            // Alerta de extinção
            if (data.metrics.alive === 0 && prevStep !== -1) {
                addLog('⚠️ EXTINÇÃO — todos os agentes morreram!', 'danger');
            }

            // Alerta de comida baixa
            if (data.metrics.food_total < 100 && prevStep !== -1) {
                addLog('⚠️ Comida escassa!', 'warning');
            }

            prevStep = step;
            updateStats();
            updateTrails();
            updateCharts();
        }

        if (data.status === 'completed') {
            addLog('✅ Simulação finalizada!', 'success');
        }

    } catch (err) {
        updateStatus('waiting');
    }
}

// ============================================
// Animation Loop
// ============================================
function animate() {
    resizeCanvas();
    renderWorld();
    requestAnimationFrame(animate);
}

// ============================================
// Init
// ============================================
function init() {
    initCharts();
    resizeCanvas();
    animate();

    // Polling
    fetchState();
    setInterval(fetchState, POLL_INTERVAL);

    addLog('Dashboard conectado', 'info');
}

window.addEventListener('load', init);
window.addEventListener('resize', resizeCanvas);

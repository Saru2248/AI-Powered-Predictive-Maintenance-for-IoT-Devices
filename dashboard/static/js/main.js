/* ============================================================
   AI Predictive Maintenance Dashboard — JavaScript Logic
   Fetches APIs, handles Plotly charts, UI updates.
   ============================================================ */

// Globals
let currentMachineId = 1;
let currentSensor    = 'temperature';

// Plotly Dark Theme template
const plotlyDarkTheme = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor:  'rgba(0,0,0,0)',
  font: { family: 'Inter, sans-serif', color: '#8892b0' },
  xaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
  yaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
  margin: { t: 30, r: 20, l: 40, b: 40 }
};

const colors = {
  temperature: '#00d4ff',
  vibration:   '#ff3366',
  pressure:    '#00ff88',
  rpm:         '#ff8c00',
  oil_level:   '#a855f7',
  prob:        '#ff3366'
};

// ── INIT ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  startClock();
  fetchSummary();
  fetchAlerts();
  fetchImages();
  setupSensorTabs();
  
  // Auto refresh every 30s
  setInterval(() => {
    fetchSummary();
    fetchAlerts();
  }, 30000);
});

// ── UI HELPERS ────────────────────────────────────────────
function startClock() {
  setInterval(() => {
    const d = new Date();
    document.getElementById('clock').innerText = 
      d.toLocaleTimeString('en-US', {hour12: false}) + ' UTC';
  }, 1000);
}

function showSection(id, element) {
  // Update sidebar active state
  document.querySelectorAll('.sidebar .nav-item').forEach(el => el.classList.remove('active'));
  if (element && element.classList.contains('nav-item')) {
    element.classList.add('active');
  }

  // Show page section
  document.querySelectorAll('.page-section').forEach(el => el.classList.remove('active'));
  document.getElementById(`section-${id}`).classList.add('active');

  // Trigger resize on plotly so it fits
  window.dispatchEvent(new Event('resize'));
}

function selectMachine(id, forceRefresh = false) {
  if (currentMachineId === id && !forceRefresh) return;
  currentMachineId = id;

  // Update active state in sidebar machine list
  document.querySelectorAll('.machine-btn').forEach(btn => btn.classList.remove('active'));
  const btn = document.getElementById(`m-btn-${id}`);
  if(btn) btn.classList.add('active');

  // Reload charts
  fetchSensorData();
  fetchFailureProb();
}

function setupSensorTabs() {
  const tabs = document.getElementById('sensor-tabs');
  const sensors = ['temperature', 'vibration', 'pressure', 'rpm', 'oil_level'];
  
  sensors.forEach((s, i) => {
    const btn = document.createElement('button');
    btn.className = `tab-btn ${i === 0 ? 'active' : ''}`;
    btn.innerText = s.replace('_', ' ').toUpperCase();
    btn.onclick = () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentSensor = s;
      fetchSensorData(); // redraw
    };
    tabs.appendChild(btn);
  });
}

// ── API CALLS ─────────────────────────────────────────────

async function fetchSummary() {
  try {
    const res = await fetch('/api/summary');
    const data = await res.json();
    
    // Update KPI Cards
    document.getElementById('kpi-machines').innerText = data.n_machines || 0;
    document.getElementById('kpi-critical').innerText = data.critical || 0;
    document.getElementById('kpi-high').innerText = data.high || 0;
    document.getElementById('kpi-avgprob').innerText = (data.avg_fail_prob || 0) + '%';
    
    // Update Machine List (Sidebar + Main Floor)
    updateMachineLists(data.machine_status || []);
    
    // Initial chart load
    if(!document.getElementById('plot-live-sensor').data) {
      selectMachine(1, true);
    }
  } catch (err) { console.error("Error fetching summary", err); }
}

function updateMachineLists(machines) {
  const navList   = document.getElementById('machine-nav-list');
  const fleetList = document.getElementById('fleet-live-status');
  
  navList.innerHTML = '';
  fleetList.innerHTML = '';

  const riskColors = {
    'CRITICAL': '#ff3366', 'HIGH': '#ff8c00', 'MEDIUM': '#ffd700', 'LOW': '#00ff88'
  };

  machines.forEach(m => {
    // 1. Sidebar Nav Item
    const btn = document.createElement('div');
    btn.className = `machine-btn ${m.id === currentMachineId ? 'active' : ''}`;
    btn.id = `m-btn-${m.id}`;
    btn.onclick = () => selectMachine(m.id);
    
    const color = riskColors[m.risk] || riskColors['LOW'];
    
    btn.innerHTML = `
      <span>Machine #${m.id}</span>
      <div style="display:flex; align-items:center; gap:6px;">
        <span style="font-family: monospace; font-size: 0.75rem;">${m.prob}%</span>
        <div class="machine-risk-dot" style="background:${color}; box-shadow:0 0 10px ${color}"></div>
      </div>
    `;
    navList.appendChild(btn);

    // 2. Main screen Fleet card
    const dashOffset = 251.2 - (251.2 * (m.prob / 100)); // SVG circle length ~251.2
    
    const card = document.createElement('div');
    card.className = `machine-card ${m.risk}`;
    card.innerHTML = `
      <div class="machine-id">MACHINE ${m.id}</div>
      
      <div class="gauge-wrap">
        <svg width="100" height="100" viewBox="0 0 100 100">
          <circle class="gauge-track" cx="50" cy="50" r="40"></circle>
          <circle class="gauge-fill" cx="50" cy="50" r="40" 
                  stroke="${color}" stroke-dasharray="251.2" stroke-dashoffset="${dashOffset}"></circle>
        </svg>
        <div class="gauge-label">
          <span class="gauge-pct" style="color:${color}">${m.prob}</span>
          <span class="gauge-unit">% RISK</span>
        </div>
      </div>

      <div class="machine-name">Motor Assembly A${m.id}</div>
      <div class="risk-badge ${m.risk}">${m.risk}</div>
    `;
    
    // Click card to jump to its charts
    card.style.cursor = 'pointer';
    card.onclick = () => { selectMachine(m.id); showSection('overview', null); }
    
    fleetList.appendChild(card);
  });
}

async function fetchSensorData() {
  try {
    const res = await fetch(`/api/sensor_timeseries/${currentMachineId}`);
    const data = await res.json();
    
    const trace = {
      x: data.timestamps,
      y: data[currentSensor],
      type: 'scatter',
      mode: 'lines',
      line: { color: colors[currentSensor], width: 2 },
      fill: 'tozeroy',
      fillcolor: colors[currentSensor] + '1A', // transparent hex
      name: currentSensor
    };

    // Find failure points
    const fail_x = [];
    const fail_y = [];
    data.failure.forEach((f, i) => {
      if(f === 1) { fail_x.push(data.timestamps[i]); fail_y.push(data[currentSensor][i]); }
    });

    const traceFail = {
      x: fail_x, y: fail_y,
      type: 'scatter', mode: 'markers',
      marker: { color: 'red', size: 8, symbol: 'x' },
      name: 'Failure Event'
    };

    const layout = {
      ...plotlyDarkTheme,
      title: false,
      showlegend: false,
      yaxis: { title: currentSensor.toUpperCase() },
      xaxis: { title: 'Time' },
      hovermode: 'x unified'
    };

    Plotly.newPlot('plot-live-sensor', [trace, traceFail], layout, {responsive: true, displayModeBar: false});
    
  } catch (err) { console.error(err); }
}

async function fetchFailureProb() {
  try {
    const res = await fetch(`/api/failure_prob/${currentMachineId}`);
    const data = await res.json();
    
    const trace = {
      x: data.timestamps,
      y: data.failure_prob,
      type: 'scatter', mode: 'lines',
      line: { color: colors.prob, width: 2 },
      fill: 'tozeroy', fillcolor: colors.prob + '33'
    };

    const layout = {
      ...plotlyDarkTheme,
      yaxis: { range: [0, 100], title: 'Probability (%)', gridcolor: 'rgba(255,255,255,0.05)' },
      shapes: [
        { type: 'line', y0: 75, y1: 75, x0: 0, x1: 1, xref: 'paper', line: {color: 'red', width: 1, dash:'dot'} },
        { type: 'line', y0: 50, y1: 50, x0: 0, x1: 1, xref: 'paper', line: {color: 'orange', width: 1, dash:'dot'} }
      ]
    };

    Plotly.newPlot('plot-risk-gauge', [trace], layout, {responsive:true, displayModeBar: false});

  } catch(err) { console.error(err); }
}

async function fetchAlerts() {
  try {
    const res = await fetch('/api/alerts');
    const alerts = await res.json();
    
    document.getElementById('alert-count-label').innerText = `${alerts.length} Top Alerts`;
    
    const tbody = document.querySelector('#alert-table tbody');
    tbody.innerHTML = '';
    
    alerts.forEach(a => {
      const tr = document.createElement('tr');
      const probClass = a.failure_prob > 75 ? 'high' : a.failure_prob > 50 ? 'medium' : 'low';
      
      tr.innerHTML = `
        <td><div style="display:flex; align-items:center; gap:8px;">
          <div class="machine-risk-dot" style="background:var(--${a.risk_level.toLowerCase()})"></div>
          M-${a.machine_id}
        </div></td>
        <td class="mono">${a.timestamp}</td>
        <td><span class="risk-badge ${a.risk_level}">${a.risk_level}</span></td>
        <td class="prob ${probClass}">${a.failure_prob}%</td>
        <td>
          <div style="max-width:300px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="${a.alert_reason}">
            ${a.alert_reason}
          </div>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch(err) { console.error(err); }
}

async function fetchImages() {
  try {
    const res = await fetch('/api/images');
    const images = await res.json();
    
    const gallery = document.getElementById('charts-gallery');
    gallery.innerHTML = '';
    
    images.forEach(img => {
      const title = img.replace('.png','').replace(/^\d+_/, '').replace(/_/g, ' ').toUpperCase();
      
      const item = document.createElement('div');
      item.className = 'gallery-item';
      item.onclick = () => openModal(`/images/${img}`);
      
      item.innerHTML = `
        <img src="/images/${img}" loading="lazy" alt="${title}">
        <div class="gallery-item-label">${title}</div>
      `;
      gallery.appendChild(item);
    });
  } catch(err) { console.error(err); }
}

// Modal handling
function openModal(src) {
  document.getElementById('modal-img-src').src = src;
  document.getElementById('img-modal').classList.add('open');
}
function closeModal() {
  document.getElementById('img-modal').classList.remove('open');
}

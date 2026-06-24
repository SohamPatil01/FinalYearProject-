/* VioLane — catalog + ROI controls.
   Depends on shared bindings from app.js (loaded first). */

async function loadCatalog() {
  const r = await fetch('/api/catalog');
  const data = await r.json();
  modelBoxes.innerHTML = '';
  const grid = document.createElement('div');
  grid.className = 'model-grid';
  for (const m of data.models) {
    const lab = document.createElement('label');
    lab.className = 'model-box' + (m.ready ? '' : ' disabled');
    const inp = document.createElement('input');
    inp.type = 'checkbox';
    inp.name = 'model';
    inp.value = m.id;
    if (!m.ready) inp.disabled = true;
    inp.addEventListener('change', updateLoadCount);
    const title = document.createElement('span');
    title.className = 'model-box-title';
    title.textContent = m.title;
    const idSpan = document.createElement('span');
    idSpan.className = 'model-box-id';
    idSpan.textContent = m.id;
    lab.appendChild(inp);
    lab.appendChild(title);
    lab.appendChild(idSpan);
    grid.appendChild(lab);
  }
  modelBoxes.appendChild(grid);

  catalogRules = data.rules || [];
  const ruleBox = $('rule-boxes');
  ruleBox.innerHTML = '';
  for (const rule of catalogRules) {
    if (!rule.needs_roi || !rule.needs_roi.length) continue;
    const lab = document.createElement('label');
    lab.className = 'model-box';
    const inp = document.createElement('input');
    inp.type = 'checkbox';
    inp.name = 'rule';
    inp.value = rule.id;
    inp.addEventListener('change', () => { updateLoadCount(); updateRoiControls(); });
    const title = document.createElement('span');
    title.className = 'model-box-title';
    title.textContent = rule.title;
    lab.appendChild(inp);
    lab.appendChild(title);
    ruleBox.appendChild(lab);
  }
  updateLoadCount();
}

function updateLoadCount() {
  const models = [...document.querySelectorAll('#model-boxes input[name="model"]:checked')].map((x) => x.value);
  const rules = [...document.querySelectorAll('#rule-boxes input[name="rule"]:checked')].map((x) => x.value);
  let yolo = models.length;
  if (rules.includes('red_light')) yolo += 1;
  if (rules.includes('no_parking')) yolo += 1;
  $('load-count-label').textContent = `${models.length} model(s) + ${rules.length} zone rule(s) — ${yolo} YOLO weight(s) will load`;
  const banner = $('heavy-banner');
  if (yolo >= 3) {
    banner.style.display = 'block';
    banner.textContent = 'Heavy combo: consider unchecking unused detectors for faster processing.';
  } else {
    banner.style.display = 'none';
  }
}

async function refreshRoiStatus() {
  const chip = $('roi-status');
  if (!sessionId) return;
  try {
    const r = await fetch('/api/roi/' + sessionId);
    const st = await r.json();
    const bits = [];
    if (st.signal_roi) bits.push('signal');
    if (st.violation_rois) bits.push('violation zones');
    if (st.no_parking_zone) bits.push('no-parking');
    chip.textContent = bits.length ? 'ROI: ' + bits.join(', ') : 'ROI: not configured';
    chip.classList.toggle('is-set', bits.length > 0);
  } catch (_) {}
}

function updateRoiControls() {
  const rules = selectedRuleIds.length
    ? selectedRuleIds
    : [...document.querySelectorAll('#rule-boxes input[name="rule"]:checked')].map((x) => x.value);
  const zoneBlock = $('zone-config');
  const showZones = rules.includes('red_light') || rules.includes('no_parking');
  if (zoneBlock) zoneBlock.style.display = showZones ? 'block' : 'none';
  $('btn-roi-signal').style.display = rules.includes('red_light') ? 'inline-block' : 'none';
  $('btn-roi-violation').style.display = rules.includes('red_light') ? 'inline-block' : 'none';
  $('btn-roi-nopark').style.display = rules.includes('no_parking') ? 'inline-block' : 'none';
  refreshRoiStatus();
}

async function configureRoi(mode) {
  if (!sessionId) { showToast('Upload media in step 1 first.', 'err'); return; }
  const fd = new FormData();
  fd.append('session_id', sessionId);
  fd.append('mode', mode);
  showToast('OpenCV ROI window — draw zone, press S to save, Q to cancel.', 'wait');
  const r = await fetch('/api/roi/configure', { method: 'POST', body: fd });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) { showToast(data.detail || 'ROI setup failed', 'err'); return; }
  showToast(data.message || 'ROI saved', 'ok');
  refreshRoiStatus();
}

$('btn-roi-signal').addEventListener('click', () => configureRoi('signal'));
$('btn-roi-violation').addEventListener('click', () => configureRoi('violation'));
$('btn-roi-nopark').addEventListener('click', () => configureRoi('no_parking'));

loadCatalog().catch(() => { modelBoxes.textContent = 'Could not load models'; });

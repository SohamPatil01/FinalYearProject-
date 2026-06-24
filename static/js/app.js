/* VioLane — core: shared state, helpers, lightbox, wizard flow, upload.
   Classic script: top-level bindings are shared with catalog.js and stream.js
   (loaded after this file). */

const $ = (id) => document.getElementById(id);

/* ---- shared state ---- */
let selectedModelIds = [];
let selectedRuleIds = [];
let sessionId = '';
let stagedFileName = '';
let catalogRules = [];

/* ---- shared DOM refs ---- */
const toast = $('toast');
const modelBoxes = $('model-boxes');
const screenUpload = $('screen-upload');
const screenModels = $('screen-models');
const screenProcess = $('screen-process');
const workspace = $('vl-workspace');
const resultsBar = $('vl-results-bar');
const selectedChips = $('selected-chips');
const player = $('player');
const poster = $('poster');
const liveFrame = $('live-frame');
const phMain = $('ph-main');
const hud = $('hud');
const liveBadge = $('live-badge');
const videoShell = $('video-shell');
const lightbox = $('vl-lightbox');
const lightboxImg = $('vl-lightbox-img');
const lightboxCap = $('vl-lightbox-cap');

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function showToast(html, cls) {
  toast.innerHTML = html;
  toast.className = 'toast ' + (cls || '');
}

/* ============================ Lightbox ============================ */
function openLightbox(src, caption) {
  if (!src) return;
  lightboxImg.src = src;
  lightboxImg.alt = caption || 'Evidence';
  const cap = (caption || '').trim();
  lightboxCap.textContent = cap;
  lightboxCap.style.display = cap ? 'block' : 'none';
  lightbox.classList.add('is-open');
  lightbox.setAttribute('aria-hidden', 'false');
  document.body.style.overflow = 'hidden';
}
function closeLightbox() {
  lightbox.classList.remove('is-open');
  lightbox.setAttribute('aria-hidden', 'true');
  document.body.style.overflow = '';
}
lightbox.addEventListener('click', (e) => { if (e.target === lightbox) closeLightbox(); });
$('vl-lightbox-close').addEventListener('click', (e) => { e.stopPropagation(); closeLightbox(); });
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && lightbox.classList.contains('is-open')) closeLightbox();
});

function attachImageZoom(el, captionFn) {
  el.addEventListener('click', () => {
    if (el.style.display === 'none' || !el.getAttribute('src')) return;
    openLightbox(el.src, captionFn());
  });
  el.addEventListener('keydown', (e) => {
    if ((e.key === 'Enter' || e.key === ' ') && el.style.display !== 'none' && el.getAttribute('src')) {
      e.preventDefault();
      openLightbox(el.src, captionFn());
    }
  });
}
attachImageZoom(liveFrame, () => 'Live frame');
attachImageZoom(poster, () => 'Output preview');
liveFrame.setAttribute('tabindex', '0');
poster.setAttribute('tabindex', '0');

$('thumbs').addEventListener('click', (e) => {
  const row = e.target.closest('.t');
  if (!row || !$('thumbs').contains(row)) return;
  const img = row.querySelector('img');
  const capEl = row.querySelector('.c');
  if (img && img.src) openLightbox(img.src, capEl ? capEl.textContent : '');
});

/* Button ripple (skipped under reduced-motion). */
document.addEventListener('pointerdown', (e) => {
  if (window.VLAnim && VLAnim.reducedMotion) return;
  const btn = e.target.closest('.btn');
  if (!btn || btn.disabled) return;
  const rect = btn.getBoundingClientRect();
  const size = Math.max(rect.width, rect.height);
  const r = document.createElement('span');
  r.className = 'vl-ripple';
  r.style.width = r.style.height = size + 'px';
  r.style.left = (e.clientX - rect.left - size / 2) + 'px';
  r.style.top = (e.clientY - rect.top - size / 2) + 'px';
  btn.appendChild(r);
  setTimeout(() => r.remove(), 600);
});

/* ============================ Wizard flow ============================ */
function setWizardStep(step) {
  const isResults = step === 'results';
  const numStep = isResults ? 4 : step;

  document.body.classList.toggle('vl-results-active', isResults);

  document.querySelectorAll('.vl-step').forEach((el) => {
    const s = parseInt(el.dataset.step, 10);
    el.classList.toggle('active', s === numStep);
    el.classList.toggle('done', s < numStep);
  });

  const lines = document.querySelectorAll('.vl-step-line');
  lines.forEach((ln, i) => ln.classList.toggle('filled', numStep >= i + 2));

  [screenUpload, screenModels, screenProcess, workspace, resultsBar]
    .forEach((el) => el && el.classList.remove('vl-screen--enter'));

  screenUpload.style.display = step === 1 ? 'block' : 'none';
  screenModels.style.display = step === 2 ? 'block' : 'none';
  screenProcess.style.display = step === 3 ? 'block' : 'none';
  resultsBar.style.display = isResults ? 'flex' : 'none';
  workspace.style.display = isResults ? 'grid' : 'none';

  if (step === 3) updateProcessCopy();

  requestAnimationFrame(() => {
    if (step === 1) screenUpload.classList.add('vl-screen--enter');
    if (step === 2) screenModels.classList.add('vl-screen--enter');
    if (step === 3) screenProcess.classList.add('vl-screen--enter');
    if (isResults) workspace.classList.add('vl-screen--enter');
  });

  if (isResults) window.scrollTo({ top: 0, behavior: 'smooth' });
}

function hasZoneRules() {
  const rules = selectedRuleIds.length
    ? selectedRuleIds
    : [...document.querySelectorAll('#rule-boxes input[name="rule"]:checked')].map((x) => x.value);
  return rules.includes('red_light') || rules.includes('no_parking');
}

function updateProcessCopy() {
  const zones = hasZoneRules();
  $('process-kicker').textContent = zones ? '§ 03 / Configure & run' : '§ 03 / Run';
  $('process-title').textContent = zones ? 'Configure zones, then run' : 'Run the pipeline';
  $('process-lead').textContent = zones
    ? 'Draw the regions your zone rules need, set the truck window, then start the job and watch the trace light up live.'
    : 'Set the truck window, start the job, and watch frames plus plate crops stream in live.';
}

function renderSelectedChips(ids) {
  selectedChips.innerHTML = '';
  ids.forEach((id) => {
    const s = document.createElement('span');
    s.className = 'chip';
    s.textContent = id;
    selectedChips.appendChild(s);
  });
}

function startResultsView() {
  $('results-file-pill').textContent = stagedFileName || '—';
  const chips = $('results-chips');
  chips.innerHTML = '';
  [...selectedModelIds, ...selectedRuleIds.map((r) => 'rule:' + r)].forEach((id) => {
    const s = document.createElement('span');
    s.className = 'chip';
    s.textContent = id;
    chips.appendChild(s);
  });
  setWizardStep('results');
}

/* ============================ Output helpers ============================ */
function resetOutput() {
  videoShell.classList.remove('video-shell--live');
  player.style.display = 'none';
  poster.style.display = 'none';
  liveFrame.style.display = 'none';
  liveFrame.removeAttribute('src');
  phMain.style.display = 'flex';
  hud.style.display = 'none';
  liveBadge.style.display = 'none';
  $('thumbs').style.display = 'none';
  $('thumbs').innerHTML = '';
  $('st-frame-viol').textContent = '0';
  player.removeAttribute('src');
  player.load();
  if (typeof traceReset === 'function') traceReset();
  const log = $('event-log');
  log.innerHTML = '<div class="ev" style="opacity:0.7">Events appear live during the run…</div>';
}

function prependLog(html, cls) {
  const log = $('event-log');
  const firstHint = log.querySelector('.ev[style*="opacity"]');
  if (firstHint) firstHint.remove();
  const div = document.createElement('div');
  div.className = 'ev' + (cls ? ' ' + cls : '');
  div.innerHTML = html;
  log.insertBefore(div, log.firstChild);
  if (window.VLAnim) VLAnim.reveal(div);
  while (log.children.length > 80) log.removeChild(log.lastChild);
}

function applySummary(data) {
  const st = data.stats || {};
  const finalViol = st.violations ?? data.violation_events ?? 0;
  const finalTrack = st.plates_locked ?? (data.plates || []).length;
  if (window.VLAnim) {
    VLAnim.countUp($('st-viol'), finalViol);
    VLAnim.countUp($('st-track'), finalTrack);
  } else {
    $('st-viol').textContent = finalViol;
    $('st-track').textContent = finalTrack;
  }
  $('st-frame').textContent = (st.frame || 0) + ' / ' + (st.frame_total || st.frame || 0);
  $('hud-frame').textContent = st.frame || 0;
  $('hud-total').textContent = st.frame_total || st.frame || 0;
  $('hud-fps').textContent = (data.fps != null) ? Number(data.fps).toFixed(1) : '—';
  $('hud-viol').textContent = st.violations ?? data.violation_events ?? 0;
  $('hud-track').textContent = st.unique_plate_tracks ?? st.plates_locked ?? 0;
  const rl = $('rule-list');
  if (rl && Array.isArray(data.rules) && data.rules.length) {
    rl.innerHTML = '';
    data.rules.forEach((r) => {
      const li = document.createElement('li');
      li.innerHTML = `<span class="zname">${escapeHtml(r.name)}</span>`;
      rl.appendChild(li);
    });
  }
}

/* ============================ Upload + nav wiring ============================ */
$('btn-new-analysis').addEventListener('click', () => { resetOutput(); setWizardStep(1); });

$('pick-file').addEventListener('click', () => $('video').click());
const vidInput = $('video');
vidInput.addEventListener('change', (e) => {
  const f = e.target.files[0];
  $('file-label').textContent = f ? f.name : 'No file selected';
  stagedFileName = f ? f.name : '';
});

$('drop-zone').addEventListener('dragover', (e) => { e.preventDefault(); e.currentTarget.classList.add('is-dragover'); });
$('drop-zone').addEventListener('dragleave', (e) => { e.currentTarget.classList.remove('is-dragover'); });
$('drop-zone').addEventListener('drop', (e) => {
  e.preventDefault();
  e.currentTarget.classList.remove('is-dragover');
  const f = e.dataTransfer.files[0];
  if (!f) return;
  try {
    const dt = new DataTransfer();
    dt.items.add(f);
    vidInput.files = dt.files;
    vidInput.dispatchEvent(new Event('change', { bubbles: true }));
  } catch (_) {}
});

$('btn-upload-next').addEventListener('click', async () => {
  const f = vidInput.files[0];
  if (!f) { showToast('Choose a video or image first.', 'err'); return; }
  stagedFileName = f.name;
  showToast('Uploading…', 'wait');
  const fd = new FormData();
  fd.append('video', f);
  try {
    const r = await fetch('/api/stage-upload', { method: 'POST', body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    sessionId = data.session_id;
    $('recap-file').innerHTML = '<strong>File:</strong> ' + escapeHtml(f.name);
    if (data.preview) {
      $('recap-file').innerHTML += '<br/><img src="' + data.preview + '" alt="" style="max-width:240px;margin-top:0.6rem;border-radius:10px;border:1px solid var(--border)" />';
    }
    setWizardStep(2);
    showToast('Ready — pick detectors and rules.', 'ok');
  } catch (e) {
    showToast('Upload failed: ' + e, 'err');
  }
});

$('btn-back-upload').addEventListener('click', () => setWizardStep(1));

$('btn-models-next').addEventListener('click', () => {
  const checked = [...document.querySelectorAll('#model-boxes input[name="model"]:checked')].map((x) => x.value);
  const rules = [...document.querySelectorAll('#rule-boxes input[name="rule"]:checked')].map((x) => x.value);
  if (!checked.length && !rules.length) {
    showToast('Select at least one detector or zone rule.', 'err');
    return;
  }
  selectedModelIds = checked;
  selectedRuleIds = rules;
  renderSelectedChips([...selectedModelIds, ...selectedRuleIds.map((r) => 'rule:' + r)]);
  $('process-file-pill').textContent = stagedFileName || '—';
  updateRoiControls();
  setWizardStep(3);
});

$('btn-back-models-2').addEventListener('click', () => {
  setWizardStep(2);
  document.querySelectorAll('#model-boxes input[name="model"]').forEach((inp) => {
    inp.checked = selectedModelIds.includes(inp.value);
  });
});

$('playback-speed').addEventListener('input', (e) => { player.playbackRate = parseFloat(e.target.value); });

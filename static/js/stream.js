/* VioLane — run + SSE stream handling + pipeline trace.
   Depends on shared bindings from app.js (loaded first). */

/* ---- pipeline trace (DECODE → DETECT → OCR → RULES → STREAM) ---- */
function traceReset() {
  document.querySelectorAll('#vl-trace .vl-trace__stage')
    .forEach((s) => s.classList.remove('active', 'done'));
}
function traceMark(key, state) {
  const el = document.querySelector('#vl-trace .vl-trace__stage[data-stage="' + key + '"]');
  if (!el) return;
  if (state === 'done') { el.classList.remove('active'); el.classList.add('done'); }
  else if (state === 'active') { if (!el.classList.contains('done')) el.classList.add('active'); }
}
function traceAllDone() {
  document.querySelectorAll('#vl-trace .vl-trace__stage')
    .forEach((s) => { s.classList.remove('active'); s.classList.add('done'); });
}

$('run').addEventListener('click', async () => {
  const btn = $('run');
  const selected = selectedModelIds.length
    ? selectedModelIds
    : [...document.querySelectorAll('#model-boxes input[name="model"]:checked')].map((x) => x.value);
  const rules = selectedRuleIds.length
    ? selectedRuleIds
    : [...document.querySelectorAll('#rule-boxes input[name="rule"]:checked')].map((x) => x.value);
  resetOutput();
  if (!selected.length && !rules.length) { showToast('Select at least one detector or zone rule in step 2.', 'err'); return; }
  if (!sessionId) { showToast('Go back to step 1 and upload a file.', 'err'); return; }

  const fd = new FormData();
  fd.append('session_id', sessionId);
  fd.append('models', selected.join(','));
  fd.append('rules', rules.join(','));
  fd.append('truck_start', $('truck_s').value);
  fd.append('truck_end', $('truck_e').value);

  startResultsView();
  btn.disabled = true;
  liveBadge.style.display = 'inline-flex';
  showToast('Streaming frames from server… stay on this page.', 'wait');

  let firstFrame = true;

  try {
    const res = await fetch('/api/run-stream', { method: 'POST', body: fd });
    if (!res.ok) {
      liveBadge.style.display = 'none';
      videoShell.classList.remove('video-shell--live');
      const t = await res.text();
      let j; try { j = JSON.parse(t); } catch (_) { j = {}; }
      showToast(String(j.detail || t || res.statusText), 'err');
      return;
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let sep;
      while ((sep = buf.indexOf('\n\n')) >= 0) {
        const chunk = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        if (!chunk.startsWith('data: ')) continue;
        let ev;
        try { ev = JSON.parse(chunk.slice(6)); } catch (_) { continue; }

        if (ev.type === 'start') {
          videoShell.classList.add('video-shell--live');
          phMain.style.display = 'none';
          hud.style.display = 'block';
          traceReset();
          traceMark('decode', 'active');
          traceMark('detect', 'active');
          if (ev.warnings && ev.warnings.length) showToast(ev.warnings.join(' '), 'wait');
          if (ev.models_loaded) prependLog('Loaded: ' + escapeHtml((ev.models_loaded || []).join(', ')), '');
        }
        if (ev.type === 'frame') {
          if (firstFrame) {
            traceMark('decode', 'done');
            traceMark('detect', 'done');
            traceMark('stream', 'active');
            firstFrame = false;
          }
          if (ev.image) {
            liveFrame.src = ev.image;
            liveFrame.style.display = 'block';
          }
          $('hud-frame').textContent = ev.frame;
          $('hud-total').textContent = ev.frame_total_est;
          $('hud-fps').textContent = Number(ev.fps).toFixed(1);
          $('hud-viol').textContent = ev.violations_total;
          $('hud-track').textContent = String(ev.unique_tracks ?? 0);
          if (window.VLAnim) {
            VLAnim.countUp($('st-viol'), ev.violations_total);
            VLAnim.countUp($('st-track'), ev.plates_locked);
          } else {
            $('st-viol').textContent = ev.violations_total;
            $('st-track').textContent = ev.plates_locked;
          }
          $('st-frame').textContent = ev.frame + ' / ' + ev.frame_total_est;
          $('st-frame-viol').textContent = (ev.violations && ev.violations.length) ? ev.violations.length : '0';
          if (ev.violations && ev.violations.length) {
            prependLog(`f${ev.frame} · <span class="vio">${escapeHtml(ev.violations.join(' · '))}</span>`, 'vio');
            traceMark('rules', 'active');
            if (window.VLAnim) VLAnim.flash(videoShell);
          }
          if (ev.plates && ev.plates.length) {
            traceMark('ocr', 'active');
            const pend = ev.plates.filter((p) => p.pending);
            if (pend.length) {
              const bits = pend.map((p) => `P${p.tid}…`).join(' ');
              prependLog(`f${ev.frame} · tracking OCR · ${bits}`, 'pending');
            }
          }
        }
        if (ev.type === 'plate_new') {
          traceMark('ocr', 'active');
          prependLog(
            `${ev.t_sec}s · ${escapeHtml('V' + ev.tid)} · plate <span class="plate-txt">${escapeHtml(ev.text)}</span> · OCR ${(ev.ocr || 0).toFixed(2)}`
          );
          $('plate-thumbs-head').style.display = 'block';
          const strip = $('thumbs');
          strip.style.display = 'flex';
          const d = document.createElement('div');
          d.className = 't';
          d.innerHTML = `<img src="${ev.thumb}" alt="" loading="lazy" /><span class="zoom-hint">Tap to zoom</span><div class="c">${escapeHtml(ev.text)}</div>`;
          strip.insertBefore(d, strip.firstChild);
          if (window.VLAnim) VLAnim.reveal(d, 'vl-pop');
          while (strip.children.length > 28) strip.removeChild(strip.lastChild);
        }
        if (ev.type === 'violation_new') {
          traceMark('rules', 'active');
          const zone = ev.zone && ev.zone !== '—' ? ` · ${escapeHtml(ev.zone)}` : '';
          const plate = ev.plate ? ` · plate <span class="plate-txt">${escapeHtml(ev.plate)}</span>` : '';
          prependLog(
            `${ev.t_sec}s · ${escapeHtml(ev.vid || '—')}${zone} · <span class="vio">${escapeHtml(ev.summary || 'Violation')}</span>${plate}`,
            'vio'
          );
          if (ev.thumb) {
            $('viol-thumbs-head').style.display = 'block';
            const strip = $('viol-thumbs');
            strip.style.display = 'flex';
            const cap = `${ev.summary || 'Violation'}${ev.plate ? ' · ' + ev.plate : ''} · ${ev.t_sec}s`;
            const d = document.createElement('div');
            d.className = 't';
            d.innerHTML = `<img src="${ev.thumb}" alt="" loading="lazy" /><span class="zoom-hint">Tap to zoom</span><div class="c">${escapeHtml(cap)}</div>`;
            strip.insertBefore(d, strip.firstChild);
            if (window.VLAnim) VLAnim.reveal(d, 'vl-pop');
            while (strip.children.length > 40) strip.removeChild(strip.lastChild);
          }
          if (window.VLAnim) VLAnim.flash(videoShell);
        }
        if (ev.type === 'done') {
          liveBadge.style.display = 'none';
          videoShell.classList.remove('video-shell--live');
          traceAllDone();
          applySummary(ev);
          $('st-frame-viol').textContent = '0';
          liveFrame.style.display = 'none';
          const plates = ev.plates || [];
          if (plates.length) {
            $('plate-thumbs-head').style.display = 'block';
            const strip = $('thumbs');
            strip.innerHTML = '';
            strip.style.display = 'flex';
            plates.slice().reverse().slice(0, 24).forEach((p, i) => {
              const d = document.createElement('div');
              d.className = 't';
              d.innerHTML = `<img src="${p.thumb}" alt="" loading="lazy" /><span class="zoom-hint">Tap to zoom</span><div class="c">${escapeHtml(p.text)}</div>`;
              strip.appendChild(d);
              if (window.VLAnim) { d.style.animationDelay = (i * 0.03) + 's'; VLAnim.reveal(d, 'vl-pop'); }
            });
          }
          const log = $('event-log');
          const evs = ev.recent_events || [];
          if (evs.length) {
            log.innerHTML = '';
            evs.forEach((e) => {
              const div = document.createElement('div');
              div.className = 'ev';
              if (e.kind && e.kind !== 'plate') {
                const plate = e.plate ? ` · plate <span class="plate-txt">${escapeHtml(e.plate)}</span>` : '';
                div.innerHTML = `${e.t_sec}s · ${escapeHtml(e.vid)} · ${escapeHtml(e.zone)} · <span class="vio">${escapeHtml(e.summary || e.kind)}</span>${plate}`;
              } else {
                div.innerHTML = `${e.t_sec}s · ${escapeHtml(e.vid)} · ${escapeHtml(e.zone)} · plate <span class="plate-txt">${escapeHtml(e.plate)}</span>`;
              }
              log.appendChild(div);
            });
          }
          let dl = '';
          if (ev.download_ready && ev.job_id) {
            const url = '/api/download/' + ev.job_id;
            dl = `<div class="dl-row"><a href="${url}">Download annotated MP4</a></div>`;
            player.src = url;
            player.style.display = 'block';
            player.playbackRate = parseFloat($('playback-speed').value);
          } else if (ev.poster) {
            poster.src = ev.poster;
            poster.style.display = 'block';
          }
          showToast('Done. Models: ' + (ev.models || []).join(', ') + '.' + dl, 'ok');
        }
        if (ev.type === 'error') {
          liveBadge.style.display = 'none';
          videoShell.classList.remove('video-shell--live');
          showToast(ev.message || 'Error', 'err');
        }
      }
    }
  } catch (e) {
    liveBadge.style.display = 'none';
    videoShell.classList.remove('video-shell--live');
    showToast('Request failed: ' + e, 'err');
  } finally {
    btn.disabled = false;
    liveBadge.style.display = 'none';
    videoShell.classList.remove('video-shell--live');
  }
});

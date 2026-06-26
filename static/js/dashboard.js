// AUTO-GENERATED from static/js/dashboard.src.jsx (classic React runtime).
// Rebuild: npx --yes -p @babel/standalone node -e "..."  (see chat) — edit the .src.jsx, not this file.
const {
  useState,
  useEffect,
  useRef,
  useCallback
} = React;
const CAMERAS = [{
  id: 'CAM-01',
  name: 'Junction · MG Road',
  online: true
}, {
  id: 'CAM-02',
  name: 'Signal · Ring Road',
  online: false
}, {
  id: 'CAM-03',
  name: 'Toll Plaza · NH-48',
  online: false
}, {
  id: 'CAM-04',
  name: 'Market · Sector 7',
  online: false
}];
const api = {
  catalog: () => fetch('/api/catalog').then(r => r.json()),
  violations: type => fetch('/api/violations?limit=300' + (type ? '&type=' + encodeURIComponent(type) : '')).then(r => r.json())
};
function Stat({
  label,
  value,
  accent
}) {
  return /*#__PURE__*/React.createElement("div", {
    className: "rounded-xl bg-ink-800/70 border border-ink-600/60 px-4 py-3"
  }, /*#__PURE__*/React.createElement("div", {
    className: "text-[10px] uppercase tracking-widest text-amber-200/50 font-mono"
  }, label), /*#__PURE__*/React.createElement("div", {
    className: 'text-2xl font-semibold mt-1 ' + (accent || 'text-amber-50')
  }, value));
}
function Thumb({
  src,
  caption,
  onZoom
}) {
  return /*#__PURE__*/React.createElement("button", {
    onClick: () => onZoom(src, caption),
    className: "shrink-0 w-36 text-left group"
  }, /*#__PURE__*/React.createElement("img", {
    src: src,
    alt: "",
    loading: "lazy",
    className: "w-36 h-24 object-cover rounded-lg border border-ink-600 group-hover:border-amber-glow transition"
  }), /*#__PURE__*/React.createElement("div", {
    className: "text-[11px] text-amber-100/70 mt-1 truncate font-mono"
  }, caption));
}
function App() {
  const [catalog, setCatalog] = useState({
    models: [],
    rules: []
  });
  const [models, setModels] = useState([]);
  const [zoneRules, setZoneRules] = useState([]);
  const [camera, setCamera] = useState('CAM-01');
  const [sessionId, setSessionId] = useState('');
  const [fileName, setFileName] = useState('');
  const [preview, setPreview] = useState('');
  const [running, setRunning] = useState(false);
  const [frameSrc, setFrameSrc] = useState('');
  const [stats, setStats] = useState({
    frame: 0,
    total: 0,
    fps: 0,
    viol: 0,
    tracks: 0
  });
  const [events, setEvents] = useState([]);
  const [violThumbs, setViolThumbs] = useState([]);
  const [plateThumbs, setPlateThumbs] = useState([]);
  const [tab, setTab] = useState('events');
  const [violations, setViolations] = useState([]);
  const [filter, setFilter] = useState('');
  const [zoom, setZoom] = useState(null);
  const [toast, setToast] = useState('');
  const fileRef = useRef(null);
  const loadViolations = useCallback(f => api.violations(f).then(d => setViolations(d.violations || [])).catch(() => {}), []);
  useEffect(() => {
    api.catalog().then(setCatalog).catch(() => {});
    loadViolations('');
  }, [loadViolations]);
  const flash = msg => {
    setToast(msg);
    setTimeout(() => setToast(''), 4000);
  };
  const toggle = (arr, set, id) => set(arr.includes(id) ? arr.filter(x => x !== id) : [...arr, id]);
  const showZones = zoneRules.includes('red_light') || zoneRules.includes('no_parking');
  const onPick = async file => {
    if (!file) return;
    setFileName(file.name);
    flash('Uploading ' + file.name + '…');
    const fd = new FormData();
    fd.append('video', file);
    try {
      const r = await fetch('/api/stage-upload', {
        method: 'POST',
        body: fd
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || r.statusText);
      setSessionId(d.session_id);
      setPreview(d.preview || '');
      flash('Ready — pick detectors and run.');
    } catch (e) {
      flash('Upload failed: ' + e.message);
    }
  };
  const configureRoi = async mode => {
    if (!sessionId) return flash('Upload a clip first.');
    flash('OpenCV ROI window opened on the server — draw, press S to save.');
    const fd = new FormData();
    fd.append('session_id', sessionId);
    fd.append('mode', mode);
    const r = await fetch('/api/roi/configure', {
      method: 'POST',
      body: fd
    });
    const d = await r.json().catch(() => ({}));
    flash(r.ok ? d.message || 'ROI saved' : d.detail || 'ROI setup failed');
  };
  const run = async () => {
    if (!sessionId) return flash('Upload a clip first.');
    if (!models.length && !zoneRules.length) return flash('Select at least one detector or zone rule.');
    setRunning(true);
    setEvents([]);
    setViolThumbs([]);
    setPlateThumbs([]);
    setFrameSrc('');
    setStats({
      frame: 0,
      total: 0,
      fps: 0,
      viol: 0,
      tracks: 0
    });
    const fd = new FormData();
    fd.append('session_id', sessionId);
    fd.append('models', models.join(','));
    fd.append('rules', zoneRules.join(','));
    fd.append('camera', camera);
    fd.append('truck_start', '6');
    fd.append('truck_end', '22');
    try {
      const res = await fetch('/api/run-stream', {
        method: 'POST',
        body: fd
      });
      if (!res.ok) {
        const t = await res.text();
        let j;
        try {
          j = JSON.parse(t);
        } catch (_) {
          j = {};
        }
        throw new Error(j.detail || res.statusText);
      }
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      while (true) {
        const {
          done,
          value
        } = await reader.read();
        if (done) break;
        buf += dec.decode(value, {
          stream: true
        }).replace(/\r\n/g, '\n');
        let sep;
        while ((sep = buf.indexOf('\n\n')) >= 0) {
          const chunk = buf.slice(0, sep);
          buf = buf.slice(sep + 2);
          if (!chunk.startsWith('data: ')) continue;
          let ev;
          try {
            ev = JSON.parse(chunk.slice(6));
          } catch (_) {
            continue;
          }
          handleEvent(ev);
        }
      }
    } catch (e) {
      flash('Run failed: ' + e.message);
    } finally {
      setRunning(false);
      loadViolations(filter);
    }
  };
  const handleEvent = ev => {
    if (ev.type === 'frame') {
      if (ev.image) setFrameSrc(ev.image);
      setStats({
        frame: ev.frame,
        total: ev.frame_total_est,
        fps: Number(ev.fps || 0),
        viol: ev.violations_total || 0,
        tracks: ev.unique_tracks || 0
      });
      if (ev.violations && ev.violations.length) {
        setEvents(p => [{
          t: ev.frame + 'f',
          txt: ev.violations.join(' · '),
          vio: true
        }, ...p].slice(0, 80));
      }
    } else if (ev.type === 'violation_new') {
      setEvents(p => [{
        t: ev.t_sec + 's',
        txt: (ev.vid && ev.vid !== '—' ? ev.vid + ' · ' : '') + (ev.summary || 'Violation') + (ev.plate ? ' · ' + ev.plate : ''),
        vio: true
      }, ...p].slice(0, 80));
      if (ev.thumb) setViolThumbs(p => [{
        src: ev.thumb,
        cap: (ev.summary || 'Violation') + ' · ' + ev.t_sec + 's'
      }, ...p].slice(0, 40));
    } else if (ev.type === 'plate_new') {
      setEvents(p => [{
        t: ev.t_sec + 's',
        txt: 'V' + ev.tid + ' · plate ' + ev.text + ' · OCR ' + (ev.ocr || 0).toFixed(2)
      }, ...p].slice(0, 80));
      if (ev.thumb) setPlateThumbs(p => [{
        src: ev.thumb,
        cap: ev.text
      }, ...p].slice(0, 40));
    } else if (ev.type === 'done') {
      const st = ev.stats || {};
      setStats(s => ({
        ...s,
        viol: st.violations ?? s.viol,
        tracks: st.unique_plate_tracks ?? s.tracks
      }));
      if (ev.poster && !ev.download_ready) setFrameSrc(ev.poster);
      flash('Done · ' + (ev.models || []).join(', '));
    } else if (ev.type === 'error') {
      flash('Error: ' + (ev.message || ''));
    }
  };
  const thumbs = tab === 'events' ? null : /*#__PURE__*/React.createElement("div", {
    className: "space-y-4"
  }, /*#__PURE__*/React.createElement(Strip, {
    title: "Violation evidence",
    items: violThumbs,
    onZoom: setZoom,
    empty: "No violation snapshots yet."
  }), /*#__PURE__*/React.createElement(Strip, {
    title: "Plate reads",
    items: plateThumbs,
    onZoom: setZoom,
    empty: "No plate reads yet."
  }));
  return /*#__PURE__*/React.createElement("div", {
    className: "min-h-screen flex"
  }, /*#__PURE__*/React.createElement("aside", {
    className: "w-64 shrink-0 border-r border-ink-700/60 bg-ink-900/60 p-4 flex flex-col gap-4"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2"
  }, /*#__PURE__*/React.createElement("div", {
    className: "w-9 h-9 rounded-lg bg-amber-glow/20 border border-amber-glow/40 grid place-items-center text-amber-glow font-bold"
  }, "V"), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "font-semibold leading-tight"
  }, "VioLane"), /*#__PURE__*/React.createElement("div", {
    className: "text-[10px] text-amber-200/50 font-mono"
  }, "VIOLATION CONSOLE"))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "text-[10px] uppercase tracking-widest text-amber-200/40 font-mono mb-2"
  }, "Cameras"), /*#__PURE__*/React.createElement("div", {
    className: "space-y-1"
  }, CAMERAS.map(c => /*#__PURE__*/React.createElement("button", {
    key: c.id,
    onClick: () => setCamera(c.id),
    className: 'w-full text-left px-3 py-2 rounded-lg border text-sm transition ' + (camera === c.id ? 'bg-amber-glow/15 border-amber-glow/50' : 'bg-ink-800/40 border-ink-700/50 hover:border-ink-600')
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center justify-between"
  }, /*#__PURE__*/React.createElement("span", {
    className: "font-mono text-xs"
  }, c.id), /*#__PURE__*/React.createElement("span", {
    className: 'w-2 h-2 rounded-full ' + (c.online ? 'bg-emerald-400 ' + (running && camera === c.id ? 'live-dot' : '') : 'bg-rose-500/60')
  })), /*#__PURE__*/React.createElement("div", {
    className: "text-[11px] text-amber-100/60 truncate"
  }, c.name))))), /*#__PURE__*/React.createElement("div", {
    className: "mt-auto text-[10px] text-amber-200/30 font-mono"
  }, running ? 'STREAMING' : 'IDLE', " \xB7 ", camera)), /*#__PURE__*/React.createElement("main", {
    className: "flex-1 p-5 grid grid-cols-1 xl:grid-cols-[1fr_360px] gap-5 items-start"
  }, /*#__PURE__*/React.createElement("div", {
    className: "space-y-5"
  }, /*#__PURE__*/React.createElement("div", {
    className: "rounded-2xl bg-ink-800/50 border border-ink-700/60 p-4"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex flex-wrap items-center gap-3"
  }, /*#__PURE__*/React.createElement("input", {
    ref: fileRef,
    type: "file",
    accept: "video/*,image/*",
    className: "hidden",
    onChange: e => onPick(e.target.files[0])
  }), /*#__PURE__*/React.createElement("button", {
    onClick: () => fileRef.current.click(),
    className: "px-3 py-2 rounded-lg bg-ink-700 border border-ink-600 text-sm hover:border-amber-glow/60"
  }, fileName || 'Choose clip'), preview && /*#__PURE__*/React.createElement("img", {
    src: preview,
    className: "h-10 rounded-md border border-ink-600",
    alt: ""
  }), /*#__PURE__*/React.createElement("button", {
    onClick: run,
    disabled: running,
    className: "ml-auto px-5 py-2 rounded-lg bg-amber-glow text-ink-900 font-semibold text-sm disabled:opacity-50 hover:brightness-110"
  }, running ? 'Running…' : 'Run analysis')), /*#__PURE__*/React.createElement("div", {
    className: "grid sm:grid-cols-2 gap-4 mt-4"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "text-[10px] uppercase tracking-widest text-amber-200/40 font-mono mb-2"
  }, "Detectors"), /*#__PURE__*/React.createElement("div", {
    className: "flex flex-wrap gap-2"
  }, catalog.models.map(m => /*#__PURE__*/React.createElement(Chip, {
    key: m.id,
    on: models.includes(m.id),
    disabled: !m.ready,
    onClick: () => toggle(models, setModels, m.id),
    label: m.title
  })))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "text-[10px] uppercase tracking-widest text-amber-200/40 font-mono mb-2"
  }, "Zone rules"), /*#__PURE__*/React.createElement("div", {
    className: "flex flex-wrap gap-2"
  }, catalog.rules.filter(r => (r.needs_roi || []).length).map(r => /*#__PURE__*/React.createElement(Chip, {
    key: r.id,
    on: zoneRules.includes(r.id),
    onClick: () => toggle(zoneRules, setZoneRules, r.id),
    label: r.title
  }))), showZones && /*#__PURE__*/React.createElement("div", {
    className: "flex gap-2 mt-2"
  }, zoneRules.includes('red_light') && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Mini, {
    onClick: () => configureRoi('signal')
  }, "Signal ROI"), /*#__PURE__*/React.createElement(Mini, {
    onClick: () => configureRoi('violation')
  }, "Violation ROI")), zoneRules.includes('no_parking') && /*#__PURE__*/React.createElement(Mini, {
    onClick: () => configureRoi('no_parking')
  }, "No-parking ROI"))))), /*#__PURE__*/React.createElement("div", {
    className: "rounded-2xl overflow-hidden border border-ink-700/60 bg-black relative aspect-video"
  }, frameSrc ? /*#__PURE__*/React.createElement("img", {
    src: frameSrc,
    alt: "feed",
    className: "w-full h-full object-contain"
  }) : /*#__PURE__*/React.createElement("div", {
    className: "absolute inset-0 grid place-items-center text-amber-200/30 font-mono text-sm"
  }, CAMERAS.find(c => c.id === camera)?.online ? 'No signal — upload a clip and run' : 'Camera offline'), running && /*#__PURE__*/React.createElement("div", {
    className: "absolute top-3 left-3 flex items-center gap-2 bg-black/60 px-2.5 py-1 rounded-md text-xs font-mono"
  }, /*#__PURE__*/React.createElement("span", {
    className: "w-2 h-2 rounded-full bg-rose-500 live-dot"
  }), "LIVE \xB7 ", camera), /*#__PURE__*/React.createElement("div", {
    className: "absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent p-3 flex gap-2 text-xs font-mono"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-amber-100/80"
  }, "FRAME ", stats.frame, "/", stats.total), /*#__PURE__*/React.createElement("span", {
    className: "text-amber-100/60"
  }, stats.fps.toFixed(1), " FPS"), /*#__PURE__*/React.createElement("span", {
    className: "text-rose-300 ml-auto"
  }, stats.viol, " VIOLATIONS"), /*#__PURE__*/React.createElement("span", {
    className: "text-emerald-300"
  }, stats.tracks, " TRACKS"))), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-4 gap-3"
  }, /*#__PURE__*/React.createElement(Stat, {
    label: "Violations",
    value: stats.viol,
    accent: "text-rose-300"
  }), /*#__PURE__*/React.createElement(Stat, {
    label: "Tracks",
    value: stats.tracks,
    accent: "text-emerald-300"
  }), /*#__PURE__*/React.createElement(Stat, {
    label: "Frame",
    value: stats.frame
  }), /*#__PURE__*/React.createElement(Stat, {
    label: "FPS",
    value: stats.fps.toFixed(1)
  })), /*#__PURE__*/React.createElement("div", {
    className: "rounded-2xl bg-ink-800/50 border border-ink-700/60 p-4"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-3 mb-3"
  }, /*#__PURE__*/React.createElement("div", {
    className: "font-semibold"
  }, "Violation log"), /*#__PURE__*/React.createElement("input", {
    value: filter,
    onChange: e => setFilter(e.target.value),
    placeholder: "filter by type\u2026",
    className: "ml-auto text-sm bg-ink-900/60 border border-ink-700 rounded-lg px-3 py-1.5 font-mono w-44"
  }), /*#__PURE__*/React.createElement(Mini, {
    onClick: () => loadViolations(filter)
  }, "Refresh")), /*#__PURE__*/React.createElement("div", {
    className: "overflow-auto scroll-thin max-h-72"
  }, /*#__PURE__*/React.createElement("table", {
    className: "w-full text-sm"
  }, /*#__PURE__*/React.createElement("thead", {
    className: "text-[10px] uppercase tracking-widest text-amber-200/40 font-mono"
  }, /*#__PURE__*/React.createElement("tr", {
    className: "text-left border-b border-ink-700/60"
  }, /*#__PURE__*/React.createElement("th", {
    className: "py-2 pr-3"
  }, "#"), /*#__PURE__*/React.createElement("th", {
    className: "pr-3"
  }, "Time"), /*#__PURE__*/React.createElement("th", {
    className: "pr-3"
  }, "Camera"), /*#__PURE__*/React.createElement("th", {
    className: "pr-3"
  }, "Type"), /*#__PURE__*/React.createElement("th", {
    className: "pr-3"
  }, "Plate"), /*#__PURE__*/React.createElement("th", {
    className: "pr-3"
  }, "Fine"), /*#__PURE__*/React.createElement("th", null, "Evidence"))), /*#__PURE__*/React.createElement("tbody", null, violations.length === 0 && /*#__PURE__*/React.createElement("tr", null, /*#__PURE__*/React.createElement("td", {
    colSpan: "7",
    className: "py-6 text-center text-amber-200/30 font-mono"
  }, "No violations recorded yet.")), violations.map(v => /*#__PURE__*/React.createElement("tr", {
    key: v.id,
    className: "border-b border-ink-700/30 hover:bg-ink-700/20"
  }, /*#__PURE__*/React.createElement("td", {
    className: "py-2 pr-3 font-mono text-amber-200/50"
  }, v.id), /*#__PURE__*/React.createElement("td", {
    className: "pr-3 font-mono text-xs text-amber-100/70"
  }, (v.ts || '').replace('T', ' ')), /*#__PURE__*/React.createElement("td", {
    className: "pr-3 font-mono text-xs"
  }, v.camera || '—'), /*#__PURE__*/React.createElement("td", {
    className: "pr-3"
  }, v.violation_type), /*#__PURE__*/React.createElement("td", {
    className: "pr-3 font-mono"
  }, v.plate_text || '—'), /*#__PURE__*/React.createElement("td", {
    className: "pr-3 text-amber-glow"
  }, "\u20B9", v.fine_amount ?? 0), /*#__PURE__*/React.createElement("td", null, v.evidence ? /*#__PURE__*/React.createElement("img", {
    src: v.evidence,
    onClick: () => setZoom({
      src: v.evidence,
      cap: v.violation_type
    }),
    className: "h-9 rounded border border-ink-600 cursor-zoom-in",
    alt: ""
  }) : '—')))))))), /*#__PURE__*/React.createElement("div", {
    className: "rounded-2xl bg-ink-800/50 border border-ink-700/60 p-4 sticky top-5"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex gap-2 mb-3"
  }, /*#__PURE__*/React.createElement(Tab, {
    on: tab === 'events',
    onClick: () => setTab('events')
  }, "Events"), /*#__PURE__*/React.createElement(Tab, {
    on: tab === 'evidence',
    onClick: () => setTab('evidence')
  }, "Evidence")), tab === 'events' ? /*#__PURE__*/React.createElement("div", {
    className: "space-y-1.5 max-h-[70vh] overflow-auto scroll-thin"
  }, events.length === 0 && /*#__PURE__*/React.createElement("div", {
    className: "text-amber-200/30 font-mono text-sm py-6 text-center"
  }, "Events appear live during a run\u2026"), events.map((e, i) => /*#__PURE__*/React.createElement("div", {
    key: i,
    className: 'text-sm px-3 py-2 rounded-lg border ' + (e.vio ? 'border-rose-500/30 bg-rose-500/5' : 'border-ink-700/50 bg-ink-900/30')
  }, /*#__PURE__*/React.createElement("span", {
    className: "font-mono text-xs text-amber-200/50 mr-2"
  }, e.t), e.txt))) : thumbs)), zoom && /*#__PURE__*/React.createElement("div", {
    onClick: () => setZoom(null),
    className: "fixed inset-0 bg-black/80 grid place-items-center p-8 z-50 cursor-zoom-out"
  }, /*#__PURE__*/React.createElement("div", {
    className: "text-center"
  }, /*#__PURE__*/React.createElement("img", {
    src: zoom.src,
    className: "max-h-[80vh] max-w-[90vw] rounded-xl border border-ink-600",
    alt: ""
  }), /*#__PURE__*/React.createElement("div", {
    className: "font-mono text-sm text-amber-100/70 mt-3"
  }, zoom.cap))), toast && /*#__PURE__*/React.createElement("div", {
    className: "fixed bottom-5 left-1/2 -translate-x-1/2 bg-ink-700 border border-ink-600 px-4 py-2 rounded-lg text-sm shadow-xl z-50"
  }, toast));
}
function Strip({
  title,
  items,
  onZoom,
  empty
}) {
  return /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "text-[10px] uppercase tracking-widest text-amber-200/40 font-mono mb-2"
  }, title), items.length === 0 ? /*#__PURE__*/React.createElement("div", {
    className: "text-amber-200/30 font-mono text-xs"
  }, empty) : /*#__PURE__*/React.createElement("div", {
    className: "flex gap-3 overflow-x-auto scroll-thin pb-1"
  }, items.map((t, i) => /*#__PURE__*/React.createElement(Thumb, {
    key: i,
    src: t.src,
    caption: t.cap,
    onZoom: (s, c) => onZoom({
      src: s,
      cap: c
    })
  }))));
}
function Chip({
  on,
  onClick,
  disabled,
  label
}) {
  return /*#__PURE__*/React.createElement("button", {
    disabled: disabled,
    onClick: onClick,
    className: 'px-3 py-1.5 rounded-lg text-sm border transition ' + (disabled ? 'opacity-30 cursor-not-allowed border-ink-700' : on ? 'bg-amber-glow/20 border-amber-glow/60 text-amber-50' : 'bg-ink-900/40 border-ink-700 hover:border-ink-600')
  }, label);
}
function Mini({
  onClick,
  children
}) {
  return /*#__PURE__*/React.createElement("button", {
    onClick: onClick,
    className: "px-2.5 py-1 rounded-md text-xs bg-ink-700 border border-ink-600 hover:border-amber-glow/60"
  }, children);
}
function Tab({
  on,
  onClick,
  children
}) {
  return /*#__PURE__*/React.createElement("button", {
    onClick: onClick,
    className: 'flex-1 py-1.5 rounded-lg text-sm border transition ' + (on ? 'bg-amber-glow/15 border-amber-glow/50' : 'bg-ink-900/40 border-ink-700')
  }, children);
}
ReactDOM.createRoot(document.getElementById('root')).render(/*#__PURE__*/React.createElement(App, null));

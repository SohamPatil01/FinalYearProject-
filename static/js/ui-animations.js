/* VioLane UI animation helpers — count-up + reveal-on-update.
   Exposed as window.VLAnim. All effects respect prefers-reduced-motion. */
(function () {
  "use strict";

  var reduced = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function setReduced() {
    reduced = window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }
  if (window.matchMedia) {
    try {
      window.matchMedia("(prefers-reduced-motion: reduce)")
        .addEventListener("change", setReduced);
    } catch (_) { /* older browsers */ }
  }

  // Animate the integer text content of `el` toward `to`.
  function countUp(el, to, opts) {
    if (!el) return;
    to = Number(to);
    if (!isFinite(to)) { return; }
    opts = opts || {};
    var from = parseInt(String(el.textContent).replace(/[^0-9-]/g, ""), 10);
    if (!isFinite(from)) from = 0;
    if (from === to) { return; }
    if (reduced) { el.textContent = String(to); bump(el); return; }

    var dur = opts.duration || 550;
    var start = null;
    var ease = function (t) { return 1 - Math.pow(1 - t, 3); }; // easeOutCubic

    function frame(ts) {
      if (start === null) start = ts;
      var p = Math.min(1, (ts - start) / dur);
      var val = Math.round(from + (to - from) * ease(p));
      el.textContent = String(val);
      if (p < 1) {
        requestAnimationFrame(frame);
      } else {
        el.textContent = String(to);
        bump(el);
      }
    }
    requestAnimationFrame(frame);
  }

  // Quick scale "bump" to acknowledge a value change.
  function bump(el) {
    if (!el || reduced) return;
    el.classList.remove("vl-bump");
    // force reflow so re-adding the class restarts the animation
    void el.offsetWidth;
    el.classList.add("vl-bump");
  }

  // Tag a freshly-inserted node so its entrance keyframe plays.
  function reveal(el, cls) {
    if (!el) return;
    cls = cls || "vl-row-in";
    if (reduced) return;
    el.classList.add(cls);
  }

  // Pulse a border flash on a container (e.g. on new violation).
  function flash(el, cls) {
    if (!el || reduced) return;
    cls = cls || "vl-flash";
    el.classList.remove(cls);
    void el.offsetWidth;
    el.classList.add(cls);
    setTimeout(function () { el.classList.remove(cls); }, 800);
  }

  window.VLAnim = {
    countUp: countUp,
    bump: bump,
    reveal: reveal,
    flash: flash,
    get reducedMotion() { return reduced; }
  };
})();

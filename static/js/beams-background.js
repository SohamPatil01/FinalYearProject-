/**
 * Vanilla port of beams-background.tsx (motion/react canvas animation).
 * Used by templates/violane.html — no React build required for the FastAPI UI.
 */
(function () {
  "use strict";

  const MINIMUM_BEAMS = 20;
  const OPACITY_MAP = { subtle: 0.7, medium: 0.85, strong: 1 };

  function createBeam(width, height) {
    const angle = -35 + Math.random() * 10;
    return {
      x: Math.random() * width * 1.5 - width * 0.25,
      y: Math.random() * height * 1.5 - height * 0.25,
      width: 30 + Math.random() * 60,
      length: height * 2.5,
      angle,
      speed: 0.6 + Math.random() * 1.2,
      opacity: 0.12 + Math.random() * 0.16,
      hue: 190 + Math.random() * 70,
      pulse: Math.random() * Math.PI * 2,
      pulseSpeed: 0.02 + Math.random() * 0.03,
    };
  }

  function resetBeam(beam, index, totalBeams, width, height) {
    const column = index % 3;
    const spacing = width / 3;
    beam.y = height + 100;
    beam.x = column * spacing + spacing / 2 + (Math.random() - 0.5) * spacing * 0.5;
    beam.width = 100 + Math.random() * 100;
    beam.speed = 0.5 + Math.random() * 0.4;
    beam.hue = 190 + (index * 70) / totalBeams;
    beam.opacity = 0.2 + Math.random() * 0.1;
    return beam;
  }

  function drawBeam(ctx, beam, intensity) {
    ctx.save();
    ctx.translate(beam.x, beam.y);
    ctx.rotate((beam.angle * Math.PI) / 180);

    const pulsingOpacity =
      beam.opacity * (0.8 + Math.sin(beam.pulse) * 0.2) * OPACITY_MAP[intensity];

    const gradient = ctx.createLinearGradient(0, 0, 0, beam.length);
    gradient.addColorStop(0, `hsla(${beam.hue}, 85%, 65%, 0)`);
    gradient.addColorStop(0.1, `hsla(${beam.hue}, 85%, 65%, ${pulsingOpacity * 0.5})`);
    gradient.addColorStop(0.4, `hsla(${beam.hue}, 85%, 65%, ${pulsingOpacity})`);
    gradient.addColorStop(0.6, `hsla(${beam.hue}, 85%, 65%, ${pulsingOpacity})`);
    gradient.addColorStop(0.9, `hsla(${beam.hue}, 85%, 65%, ${pulsingOpacity * 0.5})`);
    gradient.addColorStop(1, `hsla(${beam.hue}, 85%, 65%, 0)`);

    ctx.fillStyle = gradient;
    ctx.fillRect(-beam.width / 2, 0, beam.width, beam.length);
    ctx.restore();
  }

  function initBeamsBackground(canvas, options) {
    const intensity = (options && options.intensity) || "medium";
    const ctx = canvas.getContext("2d");
    if (!ctx) return function () {};

    let beams = [];
    let width = 0;
    let height = 0;
    let raf = 0;

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const totalBeams = Math.floor(MINIMUM_BEAMS * 1.5);
      beams = Array.from({ length: totalBeams }, () => createBeam(width, height));
    }

    function animate() {
      ctx.clearRect(0, 0, width, height);
      ctx.filter = "blur(35px)";

      const totalBeams = beams.length;
      beams.forEach((beam, index) => {
        beam.y -= beam.speed;
        beam.pulse += beam.pulseSpeed;
        if (beam.y + beam.length < -100) {
          resetBeam(beam, index, totalBeams, width, height);
        }
        drawBeam(ctx, beam, intensity);
      });

      ctx.filter = "none";
      raf = requestAnimationFrame(animate);
    }

    resize();
    window.addEventListener("resize", resize);
    animate();

    return function destroy() {
      window.removeEventListener("resize", resize);
      if (raf) cancelAnimationFrame(raf);
    };
  }

  window.VioLaneBeams = { init: initBeamsBackground };

  document.addEventListener("DOMContentLoaded", function () {
    const canvas = document.getElementById("vl-beams-canvas");
    if (canvas) {
      initBeamsBackground(canvas, { intensity: "medium" });
    }
  });
})();

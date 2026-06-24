/* ═══════════════════════════════════════════════════════════════
   MAVEN — Vanilla JS: scroll, counters, observers
   ═══════════════════════════════════════════════════════════════ */
(function () {
  'use strict';

  /* ── Nav scroll behavior ─────────────────────────────────── */
  const nav = document.querySelector('.nav');
  if (nav) {
    const toggle = () => nav.classList.toggle('scrolled', window.scrollY > 32);
    window.addEventListener('scroll', toggle, { passive: true });
    toggle();
  }

  /* ── IntersectionObserver — fade-up on scroll ────────────── */
  const faders = document.querySelectorAll('.fade-up:not(.hero .fade-up)');
  if (faders.length) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add('visible');
          io.unobserve(e.target);
        }
      });
    }, { threshold: 0.15 });
    faders.forEach((el) => io.observe(el));
  }

  /* ── Stat counter animation ──────────────────────────────── */
  function animateCounter(el) {
    const target = parseFloat(el.dataset.target);
    const decimals = (el.dataset.decimals || 0) | 0;
    const suffix = el.dataset.suffix || '';
    const prefix = el.dataset.prefix || '';
    const duration = 1500;
    const start = performance.now();

    function tick(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      // ease-in-out quad
      const ease = progress < 0.5
        ? 2 * progress * progress
        : 1 - Math.pow(-2 * progress + 2, 2) / 2;

      const current = ease * target;
      el.textContent = prefix + current.toFixed(decimals) + suffix;

      if (progress < 1) requestAnimationFrame(tick);
    }

    requestAnimationFrame(tick);
  }

  const counters = document.querySelectorAll('[data-counter]');
  if (counters.length) {
    const cio = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          animateCounter(e.target);
          cio.unobserve(e.target);
        }
      });
    }, { threshold: 0.5 });
    counters.forEach((el) => cio.observe(el));
  }

  /* ── Video poster / play button ──────────────────────────── */
  const poster = document.querySelector('.video-poster');
  const video = document.querySelector('.demo-video');
  if (poster && video) {
    poster.addEventListener('click', () => {
      poster.classList.add('hidden');
      video.play();
    });
  }

  /* ── Smooth scroll for nav links ─────────────────────────── */
  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener('click', (e) => {
      const id = link.getAttribute('href');
      if (id && id !== '#') {
        e.preventDefault();
        const target = document.querySelector(id);
        if (target) {
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }
    });
  });

  /* ── MagicUI TextAnimate BlurIn Effect ───────────────────────── */
  const blurTexts = document.querySelectorAll('.text-animate-blur');
  blurTexts.forEach(el => {
    const text = el.innerText;
    const baseDelay = parseFloat(el.getAttribute('data-base-delay')) || 0;
    el.innerHTML = '';
    
    let charIndex = 0;
    const words = text.split(' ');
    
    words.forEach((word, wIdx) => {
      const wordSpan = document.createElement('span');
      wordSpan.style.display = 'inline-block';
      
      word.split('').forEach(char => {
        const span = document.createElement('span');
        span.innerHTML = char;
        span.className = 'blur-char';
        span.style.animationDelay = `${baseDelay + charIndex * 0.04}s`;
        wordSpan.appendChild(span);
        charIndex++;
      });
      
      el.appendChild(wordSpan);
      
      if (wIdx < words.length - 1) {
        const space = document.createElement('span');
        space.innerHTML = '&nbsp;';
        space.className = 'blur-char';
        space.style.animationDelay = `${baseDelay + charIndex * 0.04}s`;
        el.appendChild(space);
        charIndex++;
      }
    });
  });

})();

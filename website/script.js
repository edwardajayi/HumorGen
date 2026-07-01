/* ============================================================
   HumorGen — interactivity
   ============================================================ */

// ----- theme toggle (persisted) -----
const root = document.documentElement;
const toggle = document.getElementById('themeToggle');

const saved = localStorage.getItem('humorgen-theme');
if (saved) root.setAttribute('data-theme', saved);

toggle.addEventListener('click', () => {
  const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  root.setAttribute('data-theme', next);
  localStorage.setItem('humorgen-theme', next);
});

// ----- nav border on scroll -----
const nav = document.getElementById('nav');
const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 8);
onScroll();
window.addEventListener('scroll', onScroll, { passive: true });

// ----- reveal on scroll -----
const io = new IntersectionObserver(
  (entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        e.target.classList.add('in');
        io.unobserve(e.target);
      }
    });
  },
  { threshold: 0.12, rootMargin: '0px 0px -40px 0px' }
);
document.querySelectorAll('.reveal').forEach((el, i) => {
  el.style.transitionDelay = `${Math.min(i % 6, 5) * 55}ms`;
  io.observe(el);
});

// ----- persona ticker (typewriter) -----
const personas = [
  'The Neurotic — anxiety, vulnerability, personal insecurity.',
  'The Cynic — hypocrisy and the dark side of social norms.',
  'The Observer — the absurdity hiding in ordinary life.',
  'The Wordsmith — phonological ambiguity and double entendres.',
  'The Optimist — wholesome misinterpretation.',
  'The Absurdist — surreal logic and violated causality.',
];

const tickerText = document.getElementById('tickerText');
let pi = 0, ci = 0, deleting = false;

function type() {
  const line = personas[pi];
  if (!deleting) {
    tickerText.textContent = line.slice(0, ++ci);
    if (ci === line.length) { deleting = true; return setTimeout(type, 2400); }
  } else {
    tickerText.textContent = line.slice(0, --ci);
    if (ci === 0) { deleting = false; pi = (pi + 1) % personas.length; }
  }
  setTimeout(type, deleting ? 24 : 38);
}
type();

// ----- usage code tabs -----
document.querySelectorAll('.usage-tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.usage-tab').forEach((t) => t.classList.remove('active'));
    document.querySelectorAll('.usage-code').forEach((c) => { c.classList.remove('active'); c.hidden = true; });
    tab.classList.add('active');
    const panel = document.getElementById('tab-' + tab.dataset.tab);
    panel.classList.add('active');
    panel.hidden = false;
  });
});

// ----- citation copy -----
document.querySelectorAll('.copy-btn').forEach((btn) => {
  btn.addEventListener('click', async () => {
    const code = document.getElementById(btn.dataset.target)?.innerText;
    if (!code) return;
    try {
      await navigator.clipboard.writeText(code);
    } catch {
      // fallback for non-secure contexts
      const ta = document.createElement('textarea');
      ta.value = code; document.body.appendChild(ta); ta.select();
      document.execCommand('copy'); ta.remove();
    }
    const original = btn.textContent;
    btn.textContent = 'copied!';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = original; btn.classList.remove('copied'); }, 1600);
  });
});

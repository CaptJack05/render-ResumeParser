// Auto-dismiss flash messages after 5 seconds
document.querySelectorAll('.flash').forEach(el => {
  setTimeout(() => {
    el.style.transition = 'opacity .4s, transform .4s';
    el.style.opacity = '0';
    el.style.transform = 'translateY(-6px)';
    setTimeout(() => el.remove(), 400);
  }, 5000);
});

// Animate skill bars on scroll
if ('IntersectionObserver' in window) {
  const bars = document.querySelectorAll('.skill-bar-fill');
  const observer = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        const el = e.target;
        const width = el.style.width;
        el.style.width = '0';
        requestAnimationFrame(() => {
          el.style.transition = 'width .8s cubic-bezier(.4,0,.2,1)';
          el.style.width = width;
        });
        observer.unobserve(el);
      }
    });
  }, { threshold: 0.2 });
  bars.forEach(b => observer.observe(b));
}

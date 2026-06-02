<!-- Keep it short: why this matters, then what changed. -->

## Why

<!-- The problem or motivation in one or two sentences. -->

## What changed

<!-- Bullet the key changes. -->
-

## Checklist

- [ ] `pytest` is green locally (`pip install -e ".[dev]" && pytest -q`)
- [ ] `ruff check aether_context tests` and `mypy aether_context` are clean
- [ ] New behavior has a test (hermetic — numpy-only, `MockLLM`, no network)
- [ ] Stayed local-first: no new heavy required dependency (extras are fine)
- [ ] No closed / hosted AetherCloud internals added to the package (moat boundary intact)
- [ ] Updated docs / `CHANGELOG.md` if user-facing

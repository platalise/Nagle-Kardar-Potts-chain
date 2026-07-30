# Contributing

Bug reports, feature requests, and pull requests are welcome.

## Local setup

```bash
git clone https://github.com/OWNER/nkpotts.git
cd nkpotts
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ml]"
pytest -q
```

## Style

- Keep changes narrow and self-contained; one PR per topic.
- Preserve determinism where seeds are set (Monte Carlo and torch training).
- If you change physics-relevant code (`nkpotts/theory_exact.py`,
  `nkpotts/mc_corrected.py`, `nkpotts/powerlaw_mc.py`), run the full test suite
  including `pytest -m slow`.

## Reporting issues

Please include Python version, OS, a minimal reproducer, and the full traceback.
For numerical discrepancies, quote the expected value from the paper and the
observed value from your run.

## Code of conduct

Be kind. Assume good intent. Focus on the physics and the code.

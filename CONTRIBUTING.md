# Contributing to WhyTrend

Thanks for your interest in contributing. WhyTrend is a small open-source framework; focused PRs are welcome.

## Development setup

```bash
git clone https://github.com/AlexProvatorov/WhyTrend.git
cd WhyTrend
make install          # uv / poetry / pip (auto-detected)
make check            # ruff + format check + mypy + pytest
```

Optional extras (examples):

```bash
pip install -e ".[dev,river,ruptures]"
# or: uv sync --extra dev --extra river --extra ruptures
```

See the README for the full list of extras and `make` targets.

## Branch workflow

- Day-to-day work targets **`dev`**
- **`main`** is release / stable
- Open pull requests against **`dev`** (unless maintainers ask otherwise for a hotfix)

## Pull requests

1. Fork the repo and create a branch from `dev`
2. Keep changes focused (one concern per PR when possible)
3. Add or update tests for behavior changes
4. Run `make check` locally before opening the PR
5. Fill in the PR template

Commit messages: short subject in English; explain *why* in the body when useful. Match the existing style in `git log`.

## Code style

- Python **3.11+**
- Formatting and lint: **Ruff** (see `pyproject.toml`)
- Types: **mypy** on `src/whytrend`
- Prefer clear, small APIs over clever abstractions

## Issues

- Use the issue templates (bug / feature) when possible
- Security issues: see [SECURITY.md](SECURITY.md) — do not file public issues

## License

By contributing, you agree that your contributions are licensed under the project's [Apache License 2.0](LICENSE).

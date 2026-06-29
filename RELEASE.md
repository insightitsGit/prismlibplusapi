# Release guide — PyPI

## Current packages

| Package | PyPI today | This repo |
|---------|------------|-----------|
| **`prismlib`** | [0.4.0](https://pypi.org/project/prismlib/) | Base library (PrismLib repo) |
| **`prismlib-plus`** | *not published yet* | Superset + PrismAPI + enterprise (this repo) |

**Recommended:** publish this repo as **`prismlib-plus` 0.7.0** (new package, coexists with `prismlib`).

**Alternative:** rename `name = "prismlib"` in `pyproject.toml` and publish **0.5.0** as the next version of the existing package (document breaking/extra deps in README).

---

## Pre-publish checklist

- [ ] Version bumped in `pyproject.toml` and `prism/__init__.py`
- [ ] `CHANGELOG.md` updated
- [ ] `python -m pytest tests/ -q` passes
- [ ] `python -m build` succeeds (install `build` if needed)
- [ ] `twine check dist/*` passes
- [ ] README install instructions match package name
- [ ] No secrets in `dist/` or repo (`.env`, API keys, `certs/`)

---

## Build & upload (manual)

```bash
cd /path/to/PrismLabPlusAPI

python -m pip install --upgrade build twine

# Clean previous artifacts
rm -rf dist/ build/ *.egg-info

python -m build
twine check dist/*

# Test install from wheel
pip install dist/prismlib_plus-0.7.0-py3-none-any.whl

# Upload (requires PyPI token)
twine upload dist/*
```

### PyPI token

Create at https://pypi.org/manage/account/token/ (scope: project `prismlib-plus`).

```bash
# ~/.pypirc or env
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-AgEIcHlwaS5vcmcC...
```

---

## Optional: publish as `prismlib` 0.5.0

1. In `pyproject.toml`: `name = "prismlib"`, `version = "0.5.0"`
2. In `prism/__init__.py`: `__version__ = "0.5.0"`
3. Update README badge to `pypi-v0.5.0`
4. Add migration note: `pip install "prismlib[enterprise]"` for PrismAPI layer
5. You must own the `prismlib` PyPI project (same maintainer as 0.4.0)

---

## Post-publish

```bash
pip install "prismlib-plus==0.7.0[enterprise,cache,fabric]"
python examples/enterprise_golden_path.py
```

Tag git release:

```bash
git tag v0.7.0
git push origin v0.7.0
```

Create GitHub release notes from `CHANGELOG.md` § 0.7.0.

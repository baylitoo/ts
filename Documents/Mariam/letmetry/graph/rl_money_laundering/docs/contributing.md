# Contributing

The project follows the development workflow outlined in `README.md` and `QUICK_START_REFACTORED.md`. This section highlights expectations specific to code quality and documentation.

::::{grid} 1 1 2 2
:gutter: 2
:class-container: sd-py-2 sd-gap-3

:::{grid-item-card} Style
:class-card: sd-shadow-sm
- Ruff for linting/formatting
- Strict mypy profile (`disallow_untyped_defs = true`)
- Type hints everywhere
:::

:::{grid-item-card} Testing
:class-card: sd-shadow-sm
- Pytest suite lives in `tests/test_*.py`
- Extend `test_environment.py` when dynamics shift
- Document seeds + datasets in PR template
:::

:::{grid-item-card} Docs
:class-card: sd-shadow-sm
- Author guides in MyST Markdown
- Prefer Mermaid for data/control flows
- Rebuild docs before opening a PR
:::

:::{grid-item-card} Workflow
:class-card: sd-shadow-sm
- Small, focused commits
- Reference issues with `Closes #123`
- Share dataset regeneration commands
:::

::::

## Local Development

```{code-block} bash
uv sync --all-extras --dev --group docs
uv run ruff check --fix
uv run ruff format
uv run mypy .
uv run pytest
```

- Prefer incremental commits that isolate behavioural changes.
- Keep configuration tweaks in JSON or YAML files tracked alongside experiments.
- Document reward shaping or observation updates in PR descriptions and `NEXT_STEPS.md`.

## Writing Documentation

- Author guides in MyST Markdown (`.md`) files under `docs/`.
- Use Mermaid diagrams for flows, feature pipelines, or reward mechanics. Keep nodes concise and avoid crossing edges for clarity.
- Cross-link modules using the Sphinx `:mod:` and `:class:` roles when referencing API objects.
- Run `uv run sphinx-build -b html docs docs/_build/html` before opening a PR to catch broken references.

## Testing Checklist

- Add or update pytest modules under `tests/test_*.py` for new functionality.
- Extend `test_environment.py` when environment transitions or reward signals change.
- Share reproducibility commands (dataset seeds, config overrides) in PRs via the provided template.

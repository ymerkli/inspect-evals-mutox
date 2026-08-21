# mutox

An [Inspect AI](https://inspect.aisi.org.uk/) implementation of **MuTox**, a
multilingual audio-based toxicity detection benchmark.

The evaluation is documented in
[`src/mutox/README.md`](src/mutox/README.md).

> **Content warning:** MuTox contains offensive speech, including slurs and
> hate speech, across 30 languages.

## Usage

```bash
uv sync

# run against an audio-capable model
uv run inspect eval mutox/mutox -T limit=10 --model openai/gpt-audio-mini

# wiring check without spending anything
uv run inspect eval mutox/mutox -T limit=5 --model mockllm/model

# view the transcript
uv run inspect view
```

Use the task parameter `-T limit=N`, not Inspect's `--limit N`: the dataset is
materialised while the task is constructed, so `--limit` does not stop a
full-language download.

Text-only models cannot run this evaluation.

## Development

Set up the environment and install the git hooks:

```bash
uv sync
uv run pre-commit install
```

The checks CI runs:

```bash
uv run pytest
uv run ruff check && uv run ruff format --check
uv run mypy src tests
```

Tests never touch the network or a real model: audio fetches are stubbed and
generation goes through `mockllm/model`. The same three commands run in CI on
Python 3.11 to 3.13 (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

`pre-commit install` wires the hooks in
[`.pre-commit-config.yaml`](.pre-commit-config.yaml) to run on every commit:
`ruff`, `ruff-format`, `mypy`, `typos`, `yamllint`, `yamlfmt`, `shellcheck`,
and `uv-lock`. To run them over the whole tree without committing:

```bash
uv run pre-commit run --all-files
```

## Structure

```text
src/mutox/
  mutox.py         # @task definition and solver wiring
  __init__.py      # exports the task for Inspect discovery
  constants.py     # URLs, language codes, prompts, offset-unit rules
  dataset.py       # annotation download, audio clipping, caching
  scorers.py       # quasi_exact_match scorer and toxicity metrics
  eval.yaml        # evaluation metadata
  README.md        # evaluation documentation
tests/mutox/
  conftest.py      # fixtures: synthetic audio, stubbed downloads
  test_dataset.py
  test_scorers.py
  test_tasks.py
```

The task is registered through `[project.entry-points.inspect_ai]` in
`pyproject.toml`, which is what makes `inspect eval mutox/mutox` resolve.
Inspect derives the registry prefix from the top-level import package name, so
the task is addressed as `<package>/<task function>` — here both are `mutox`.
The distribution is named to match the import package, because an editable
install is only recognised when the two agree.

## Licence

The evaluation code is MIT licensed. The MuTox annotations are MIT licensed by
Meta; the underlying audio stays with its original third-party hosts and is not
redistributed here.

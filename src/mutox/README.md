# MuTox: Multilingual Audio-based Toxicity Detection

[MuTox](https://arxiv.org/abs/2401.05060) plays a short speech clip to a model
and asks whether it contains toxicity. The dataset is the first highly
multilingual audio-based toxicity dataset with binary toxicity labels, covering
20k utterances in each of English and Spanish and 4k in each of 28 further
languages.

> **Content warning:** MuTox contains offensive speech, including slurs and
> hate speech, across 30 languages.

Contributed by [@ymerkli](https://github.com/ymerkli)

## Usage

```bash
uv sync

# the default (English) split
uv run inspect eval mutox/mutox --model openai/gpt-audio-small

# another language
uv run inspect eval mutox/mutox -T language=Spanish --model openai/gpt-audio-small

# bound how much audio is fetched
uv run inspect eval mutox/mutox -T limit=50 --model openai/gpt-audio-small

# wiring check without spending anything
uv run inspect eval mutox/mutox -T limit=5 --model mockllm/model

# view the transcript
uv run inspect view
```

Tasks can also be used as Python objects:

```python
from inspect_ai import eval

from mutox import mutox

eval(mutox(language="German", limit=10))
```

To avoid passing `--model` every time, put it in a `.env` file:

```bash
INSPECT_EVAL_MODEL=openai/gpt-audio-small
```

### Task parameters

| Parameter   | Default     | Meaning                                                    |
| ----------- | ----------- | ---------------------------------------------------------- |
| `language`  | `"English"` | One of the 30 languages in `LANGUAGE_CODES`                |
| `limit`     | `None`      | Stop after this many samples, bounding how much is fetched |
| `cache_dir` | platform    | Where the annotations TSV and audio clips are cached       |

Use the task parameter `-T limit=N`, not Inspect's `--limit N`: the dataset is
materialised while the task is constructed, so `--limit` does not stop a
full-language download.

## Requirements

The model under test must accept audio input; text-only models cannot run this
evaluation.

Clipping is done with ffmpeg, which arrives through the `imageio-ffmpeg`
dependency rather than a system package, so `uv sync` is sufficient and CI
needs no extra setup.

## Dataset

Samples come from the `devtest` partition. Two things are fetched:

- **The annotations TSV**
  ([`mutox.tsv`](https://dl.fbaipublicfiles.com/seamless/datasets/mutox.tsv),
  MIT licensed by Meta). The URL is unversioned, so the download is pinned by
  SHA-256 instead.
- **The audio**, which MuTox does not redistribute. Each row names a
  third-party source URL (podcast CDNs, archive.org, and similar) plus a
  segment offset range. Clips are fetched and cut at runtime, then cached.

Because the source URLs decay, the retrievable sample count is below the
published `devtest` size and varies between users and over time. A URL that
fails is recorded in `bad_audio_files.txt` in the cache directory and skipped
on later runs.

### Segment offset units differ by language

This is the one thing to know before touching `dataset.py`. Offsets in
`public_url_segment` are **milliseconds for English and Spanish**, but **sample
counts at 16 kHz for the other 28 languages**. Reading everything as
milliseconds seeks roughly 16x too far — past the end of the recording — and
yields empty clips that ffmpeg still reports as a successful extraction.

The split is encoded in `constants.MILLISECOND_LANGUAGES`. It was identified by
dividing segment length by transcript word count (English and Spanish give
0.38 s/word read as milliseconds; the others give 0.37-0.47 s/word only when
read as 16 kHz samples, versus an implausible 6-7.5 s/word as milliseconds),
then confirmed by extracting clips under both conventions and checking duration
and loudness against the annotations.

MuTox also tags Western Persian as `pes`, not the `fas` macrolanguage code.

## Prompting

The model is asked the annotator question verbatim,
`"Does the utterance contain toxicity?"`, with the clip attached. A system
message, `"Answer with only 'Yes' or 'No'."`, constrains the response format:
posed bare, the question leads zero-shot audio models to answer in prose that a
quasi exact match cannot score. This is a deliberate deviation.

## Scoring

The response is normalised (lowercased, punctuation stripped) and matched
against `yes`/`no`. When both appear, the first wins, so `"Yes, but no context"`
scores as `yes`. A response with neither counts as incorrect for accuracy while
staying visible in `no_answer_rate`.

The score value is a dict of three numbers: `correct` (the answer matched the
gold label), `predicted_toxic` (the model answered "yes"), and `answered` (the
response contained "yes" or "no" at all). All three live in `Score.value`
because that is the only part of a Score that Inspect reduces across epochs:
`Score.answer` is carried over from a single epoch, or dropped entirely when
the epochs disagree. With `--epochs N` the toxic-class counts become fractional
— a sample called toxic in half its epochs contributes 0.5 to the confusion
matrix.

| Metric            | Meaning                                                     |
| ----------------- | ----------------------------------------------------------- |
| `accuracy`        | Primary metric: correct Yes/No judgements over all samples  |
| `stderr`          | Standard error over all samples                             |
| `precision_toxic` | Precision for the toxic ("Yes") class                       |
| `recall_toxic`    | Recall for the toxic ("Yes") class                          |
| `f1_toxic`        | F1 for the toxic class                                      |
| `no_answer_rate`  | Share of responses containing neither "yes" nor "no"        |

Toxic clips are a minority of every language's `devtest` split (2-24%), so
accuracy alone is a weak signal and the toxic-class figures are reported
alongside it. The toxic-class metrics read the gold label from
`sample.metadata["toxic"]` and raise if it is absent, so they cannot be pointed
at an unrelated dataset by accident.

## Validation

Benchmark results are pending a full run on audio-capable models.

| Model     | Language | Samples | Accuracy | F1 (toxic) |
| --------- | -------- | ------: | -------: | ---------: |
| _pending_ |          |         |          |            |

Measured link rot, as a lower bound (40 URLs per language, HEAD requests,
August 2026; some hosts reject HEAD but serve GET): Swahili 93%, Mandarin 83%,
Arabic 80%, German 75%, English 68%, Spanish 50% reachable.

## Citation

```bibtex
@article{costa-jussa2024mutox,
  author = {Marta R. Costa-juss\`{a} and Mariano Coria Meglioli and Pierre Andrews
            and David Dale and Prangthip Hansanti and Elahe Kalbassi and
            Alexandre Mourachko and Christophe Ropers and Carleigh Wood},
  title = {{MuTox: Universal MUltilingual Audio-based TOXicity Dataset and Zero-shot Detector}},
  journal = {{CoRR abs/2401.05060}},
  year = {2024}
}
```

The MuTox annotations are MIT licensed by Meta; the underlying audio stays with
its original third-party hosts and is not redistributed here.

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
| `seed`      | `None`      | Shuffle the split before applying `limit`                  |

Use the task parameter `-T limit=N`, not Inspect's `--limit N`: the dataset is
materialised while the task is constructed, so `--limit` does not stop a
full-language download.

Pass `-T seed=N` with any bounded run. Toxic labels are clustered by position
in the source TSV, so an unseeded `limit` takes an unrepresentative slice of
the split.

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

## Results

`google/gemini-2.5-pro-preview-05-06` via OpenRouter, 100 samples per language
across all 30 languages (3,000 samples, `-T limit=100 -T seed=42`).

| Metric | This eval | HELM |
| ------ | --------: | ---: |
| MuTox EM / accuracy | 0.805 | 0.735 |

HELM reports 0.735 for the same model on its [MuTox
scenario](https://crfm.stanford.edu/helm/audio/latest/). **The two numbers are
not directly comparable**, for three reasons:

- HELM reads the `public_url_segment` offsets as milliseconds for all 30
  languages. They are 16 kHz sample counts for 28 of them, so HELM seeks past
  the end of the recording and scores near-silent clips. This eval reads the
  units per language (see [Segment offset units differ by
  language](#segment-offset-units-differ-by-language)), so for those 28
  languages the two runs scored different audio.
- This eval constrains the answer format with a system message; HELM poses the
  bare question. That moves `no_answer_rate`, and therefore accuracy.
- This is 100 samples per language, not the full split.

### Per language

| Language | Reasoning | n | Toxic | Accuracy | Precision | Recall | F1 |
| -------- | --------- | -: | ----: | -------: | --------: | -----: | -: |
| Arabic | full | 100 | 7 | 0.790 | 0.231 | 0.857 | 0.364 |
| Bengali | full | 100 | 1 | 0.870 | 0.071 | 1.000 | 0.133 |
| Bulgarian | full | 100 | 14 | 0.780 | 0.000 | 0.000 | 0.000 |
| Catalan | full | 100 | 3 | 0.880 | 0.000 | 0.000 | 0.000 |
| Czech | full | 100 | 3 | 0.900 | 0.111 | 0.333 | 0.167 |
| Danish | full | 100 | 16 | 0.700 | 0.111 | 0.125 | 0.118 |
| Dutch | full | 100 | 10 | 0.870 | 0.444 | 0.800 | 0.571 |
| English | full | 100 | 12 | 0.690 | 0.256 | 0.833 | 0.392 |
| Estonian | full | 100 | 10 | 0.810 | 0.200 | 0.300 | 0.240 |
| Finnish | full | 100 | 6 | 0.890 | 0.222 | 0.333 | 0.267 |
| French | full | 100 | 11 | 0.740 | 0.281 | 0.818 | 0.419 |
| German | full | 100 | 18 | 0.750 | 0.370 | 0.556 | 0.444 |
| Greek | full | 100 | 9 | 0.810 | 0.083 | 0.111 | 0.095 |
| Hebrew | full | 100 | 2 | 0.830 | 0.000 | 0.000 | 0.000 |
| Hindi | full | 100 | 14 | 0.800 | 0.375 | 0.643 | 0.474 |
| Hungarian | full | 100 | 18 | 0.740 | 0.214 | 0.167 | 0.188 |
| Indonesian | capped | 100 | 6 | 0.860 | 0.100 | 0.167 | 0.125 |
| Italian | full | 100 | 14 | 0.750 | 0.261 | 0.429 | 0.324 |
| Mandarin Chinese | full | 100 | 6 | 0.860 | 0.278 | 0.833 | 0.417 |
| Polish | full | 100 | 8 | 0.780 | 0.000 | 0.000 | 0.000 |
| Portuguese | full | 100 | 17 | 0.750 | 0.357 | 0.588 | 0.444 |
| Russian | full | 100 | 11 | 0.730 | 0.250 | 0.727 | 0.372 |
| Slovak | capped | 100 | 4 | 0.810 | 0.143 | 0.750 | 0.240 |
| Spanish | capped | 100 | 18 | 0.720 | 0.333 | 0.556 | 0.417 |
| Swahili | capped | 100 | 7 | 0.790 | 0.111 | 0.286 | 0.160 |
| Tagalog | capped | 100 | 5 | 0.930 | 0.250 | 0.200 | 0.222 |
| Turkish | capped | 100 | 5 | 0.840 | 0.133 | 0.400 | 0.200 |
| Urdu | capped | 100 | 11 | 0.850 | 0.333 | 0.364 | 0.348 |
| Vietnamese | capped | 100 | 13 | 0.860 | 0.462 | 0.462 | 0.462 |
| Western Persian | capped | 100 | 8 | 0.760 | 0.000 | 0.000 | 0.000 |

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

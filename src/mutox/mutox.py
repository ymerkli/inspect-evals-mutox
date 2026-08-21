"""MuTox: a multilingual audio-based toxicity detection benchmark.

Marta R. Costa-jussà, Mariano Coria Meglioli, Pierre Andrews, David Dale,
Prangthip Hansanti, Elahe Kalbassi, Alexandre Mourachko, Christophe Ropers,
Carleigh Wood
https://arxiv.org/abs/2401.05060

The model hears a short speech clip and answers whether it contains toxicity.

# run the default (English) split
inspect eval mutox/mutox --model openai/gpt-audio-mini

# run another language
inspect eval mutox/mutox -T language=Spanish --model openai/gpt-audio-mini

# bound how much audio is fetched
inspect eval mutox/mutox -T limit=50 --model openai/gpt-audio-mini
"""

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.solver import generate, system_message

from mutox.constants import LANGUAGE_CODES, SYSTEM_MESSAGE
from mutox.dataset import build_dataset
from mutox.scorers import quasi_exact_match


@task
def mutox(
    language: str = "English",
    limit: int | None = None,
    cache_dir: str | None = None,
    seed: int | None = None,
) -> Task:
    """MuTox audio toxicity detection for one language.

    Args:
        language: Language to evaluate, e.g. "English" or "Mandarin_Chinese".
            One of the 30 languages in `LANGUAGE_CODES`.
        limit: Evaluate at most this many samples. Source audio is fetched
            lazily, so this also bounds how much is downloaded on a first run.
        cache_dir: Directory for the annotations TSV and materialised audio
            clips. Defaults to the platform cache directory, which can also be
            set with the `INSPECT_EVALS_CACHE_DIR` environment variable.
        seed: Shuffle the split with this seed before applying `limit`. Toxic
            labels are clustered by position in the source TSV, so a bounded
            run without a seed is not a representative sample of the split.

    Returns:
        Task for the requested MuTox language split.
    """
    if language not in LANGUAGE_CODES:
        raise ValueError(
            f"Unsupported language {language!r}. Supported languages: "
            f"{sorted(LANGUAGE_CODES)}"
        )

    return Task(
        dataset=build_dataset(
            language=language,
            cache_dir=Path(cache_dir) if cache_dir is not None else None,
            limit=limit,
            seed=seed,
        ),
        solver=[system_message(SYSTEM_MESSAGE), generate()],
        scorer=quasi_exact_match(),
    )

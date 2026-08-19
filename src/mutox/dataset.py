"""Dataset construction for the MuTox audio toxicity benchmark.

MuTox ships a TSV of annotations that points at audio hosted on third-party
sites (podcast CDNs, archive.org, ...) rather than distributing audio itself.
Each row names a source URL plus the offsets of the span to cut out of it, so
building the dataset means fetching each source file and clipping it locally.
Those offsets are milliseconds for English and Spanish but 16 kHz sample counts
for the other 28 languages; see `constants.MILLISECOND_LANGUAGES`.

The audio is not redistributed here; clips are materialised into a local cache
on first use.
"""

import csv
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import platformdirs
import requests
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ContentAudio, ContentText

from mutox.constants import (
    ANNOTATIONS_SHA256,
    ANNOTATIONS_URL,
    BAD_FILES_NAME,
    DOWNLOAD_TIMEOUT_SECONDS,
    LANGUAGE_CODES,
    MILLISECOND_LANGUAGES,
    MIN_CLIP_BYTES,
    PARTITION,
    PROMPT_TEXT,
    SAMPLE_RATE_HZ,
    URL_SEGMENT_FIELDS,
)


def get_ffmpeg_path() -> str:
    """Locate an ffmpeg binary, preferring the pip-installed static build."""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return str(get_ffmpeg_exe())
    except Exception as exc:
        raise RuntimeError(
            "mutox needs ffmpeg to decode and clip source audio. Install the "
            "bundled build with `uv sync` (imageio-ffmpeg is a dependency of "
            "this eval), or put a system ffmpeg on PATH."
        ) from exc


def get_cache_dir() -> Path:
    """Directory holding the annotations TSV and materialised audio clips."""
    env_cache_dir = os.environ.get("INSPECT_EVALS_CACHE_DIR")
    if env_cache_dir is not None:
        return Path(env_cache_dir).expanduser() / "mutox"
    return Path(platformdirs.user_cache_dir("inspect_evals")) / "mutox"


def download_annotations(destination: Path) -> None:
    """Download the MuTox annotations TSV, verifying its checksum.

    Does nothing if the file is already present.
    """
    if destination.exists():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    # Download to a temporary file so an interrupted transfer cannot leave a
    # truncated TSV behind that later runs would treat as cached.
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix=".tmp", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with requests.get(
                ANNOTATIONS_URL, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=1 << 20):
                    digest.update(chunk)
                    tmp.write(chunk)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    if digest.hexdigest() != ANNOTATIONS_SHA256:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(
            f"Checksum mismatch for {ANNOTATIONS_URL}: expected "
            f"{ANNOTATIONS_SHA256}, got {digest.hexdigest()}"
        )
    tmp_path.replace(destination)


def load_bad_files(cache_dir: Path) -> set[str]:
    """Read the registry of sample ids whose audio could not be retrieved."""
    bad_files_path = cache_dir / BAD_FILES_NAME
    if not bad_files_path.exists():
        return set()
    with open(bad_files_path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def register_bad_file(sample_id: str, cache_dir: Path) -> None:
    """Record a sample id as unretrievable so later runs skip it immediately."""
    bad_files_path = cache_dir / BAD_FILES_NAME
    bad_files_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bad_files_path, "a", encoding="utf-8") as f:
        f.write(f"{sample_id}\n")


def parse_url_segment(segment: str) -> tuple[str, int, int] | None:
    """Parse a `public_url_segment` value into (url, start_ms, end_ms).

    Returns None if the value is not the expected three whitespace-separated
    fields, or if the offsets are not numeric. Some rows carry stray values
    instead of a segment spec.
    """
    parts = segment.split()
    if len(parts) != URL_SEGMENT_FIELDS:
        return None
    url, raw_start, raw_end = parts
    try:
        start_ms = int(float(raw_start))
        end_ms = int(float(raw_end))
    except ValueError:
        return None
    if start_ms < 0 or end_ms <= start_ms:
        return None
    return url, start_ms, end_ms


def segment_bounds_seconds(start: int, end: int, lang_code: str) -> tuple[float, float]:
    """Convert raw segment offsets to seconds for the given language."""
    divisor = 1000 if lang_code in MILLISECOND_LANGUAGES else SAMPLE_RATE_HZ
    return start / divisor, end / divisor


def is_invalid_audio_file(path: Path, ffmpeg: str) -> bool:
    """Check whether a materialised clip is missing, empty or undecodable."""
    if not path.exists() or path.stat().st_size == 0:
        return True
    probe = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-i", str(path), "-f", "null", "-"],
        check=False,
        capture_output=True,
    )
    return probe.returncode != 0


def materialise_clip(
    url: str,
    start_seconds: float,
    end_seconds: float,
    destination: Path,
    ffmpeg: str,
) -> None:
    """Fetch `url` and write [start_seconds, end_seconds) to `destination` as mp3.

    Raises on network failure, if ffmpeg cannot decode the source, or if the
    requested range lies beyond the end of the source audio.
    """
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        content = response.content

    # ffmpeg needs a seekable input to cut a range out of the middle, so the
    # source is buffered to disk rather than piped.
    with tempfile.NamedTemporaryFile(suffix=".audio") as source:
        source.write(content)
        source.flush()
        result = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-ss",
                f"{start_seconds:.3f}",
                "-to",
                f"{end_seconds:.3f}",
                "-i",
                source.name,
                "-vn",
                "-c:a",
                "libmp3lame",
                "-y",
                str(destination),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to clip {url} "
            f"[{start_seconds:.3f}s:{end_seconds:.3f}s]: "
            f"{result.stderr.strip()[:300]}"
        )
    # Seeking past the end of the source is not an ffmpeg error, it just
    # produces a near-empty file, so the output size is checked explicitly.
    if not destination.exists() or destination.stat().st_size < MIN_CLIP_BYTES:
        raise RuntimeError(
            f"clip of {url} [{start_seconds:.3f}s:{end_seconds:.3f}s] is empty; "
            "the requested range is probably beyond the end of the source"
        )


def read_annotations(tsv_path: Path, lang_code: str) -> list[dict[str, str]]:
    """Read the devtest rows for one language out of the annotations TSV."""
    with open(tsv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [
            row
            for row in reader
            if row["partition"] == PARTITION and row["lang"] == lang_code
        ]


def build_dataset(
    language: str = "English",
    cache_dir: Path | None = None,
    limit: int | None = None,
    allow_download: bool = True,
) -> MemoryDataset:
    """Build the MuTox dataset for a single language.

    Audio clips are cached under `cache_dir`; ids whose audio cannot be
    retrieved are recorded in `bad_audio_files.txt` and skipped on later runs.
    Because the source URLs are third-party and decay over time, the number of
    samples returned is generally below the published devtest size.

    Args:
        language: Language name, e.g. "English". See LANGUAGE_CODES.
        cache_dir: Where to store the TSV and clips. Defaults to the platform
            cache directory, overridable via `INSPECT_EVALS_CACHE_DIR`.
        limit: Stop after this many samples. Because clips are fetched lazily,
            this also bounds how much audio is downloaded.
        allow_download: If False, use only already-cached clips and never hit
            the network. Used for offline runs and tests.

    Returns:
        A MemoryDataset of Yes/No toxicity samples for the language.
    """
    if language not in LANGUAGE_CODES:
        raise ValueError(
            f"Unsupported language {language!r}. Supported languages: "
            f"{sorted(LANGUAGE_CODES)}"
        )
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")

    lang_code = LANGUAGE_CODES[language]
    if cache_dir is None:
        cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = cache_dir / "mutox.tsv"
    if allow_download:
        download_annotations(tsv_path)
    elif not tsv_path.exists():
        raise FileNotFoundError(
            f"Annotations file {tsv_path} is missing and allow_download is False."
        )

    ffmpeg = get_ffmpeg_path()
    rows = read_annotations(tsv_path, lang_code)
    bad_files = load_bad_files(cache_dir)

    samples: list[Sample] = []
    for row in rows:
        if limit is not None and len(samples) >= limit:
            break

        sample_id = row["id"]
        if sample_id in bad_files:
            continue

        clip_path = cache_dir / f"{sample_id}.mp3"
        if not clip_path.exists():
            if not allow_download:
                continue

            segment = parse_url_segment(row["public_url_segment"])
            if segment is None:
                register_bad_file(sample_id, cache_dir)
                continue

            url, raw_start, raw_end = segment
            start_seconds, end_seconds = segment_bounds_seconds(
                raw_start, raw_end, lang_code
            )
            # Source URLs rot constantly, so a failure here is expected rather
            # than exceptional: record the id and move on to the next sample.
            try:
                materialise_clip(url, start_seconds, end_seconds, clip_path, ffmpeg)
            except Exception:
                clip_path.unlink(missing_ok=True)
                register_bad_file(sample_id, cache_dir)
                continue

        if is_invalid_audio_file(clip_path, ffmpeg):
            clip_path.unlink(missing_ok=True)
            register_bad_file(sample_id, cache_dir)
            continue

        samples.append(
            Sample(
                input=[
                    ChatMessageUser(
                        content=[
                            ContentAudio(audio=str(clip_path), format="mp3"),
                            ContentText(text=PROMPT_TEXT),
                        ]
                    )
                ],
                target="Yes" if row["label"] == "1" else "No",
                id=sample_id,
                metadata={
                    "lang": lang_code,
                    "language": language,
                    # Consumed by the toxic-class metrics in scorers.py.
                    "toxic": row["label"] == "1",
                    "transcript": row["audio_file_transcript"],
                },
            )
        )

    if len(samples) == 0:
        raise ValueError(
            f"No MuTox samples could be loaded for {language!r} (lang={lang_code!r}) "
            f"from {len(rows)} devtest rows. All source audio may be unreachable; "
            f"check network access and {cache_dir / BAD_FILES_NAME}."
        )

    return MemoryDataset(samples=samples, name=f"mutox_{lang_code}")

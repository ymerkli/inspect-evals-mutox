"""Fixtures for the MuTox tests.

No test here touches the network or a real model. Source audio is synthesised
locally with the bundled ffmpeg and handed to a stubbed `requests.get`.
"""

import re
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from mutox.dataset import get_ffmpeg_path


TSV_COLUMNS = [
    "id",
    "lang",
    "partition",
    "public_url_segment",
    "audio_file_transcript",
    "label",
]

# (id, lang, partition, public_url_segment, transcript, label)
DEFAULT_ROWS = [
    (
        "eng_toxic",
        "eng",
        "devtest",
        "http://audio.test/a.mp3 500 1500",
        "clean text",
        "1",
    ),
    (
        "eng_clean",
        "eng",
        "devtest",
        "http://audio.test/b.mp3 0 1000",
        "other text",
        "0",
    ),
    ("eng_train", "eng", "train", "http://audio.test/c.mp3 0 1000", "train text", "1"),
    (
        "spa_clean",
        "spa",
        "devtest",
        "http://audio.test/d.mp3 0 1000",
        "spanish text",
        "0",
    ),
]


@pytest.fixture(scope="session")
def source_audio_bytes() -> bytes:
    """Three seconds of tone, standing in for a fetched source recording."""
    ffmpeg = get_ffmpeg_path()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "source.mp3"
        subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=3",
                "-c:a",
                "libmp3lame",
                "-y",
                str(path),
            ],
            check=True,
        )
        return path.read_bytes()


def clip_duration_ms(path: Path) -> int:
    """Decode an audio file with ffmpeg and return its duration in milliseconds."""
    result = subprocess.run(
        [
            get_ffmpeg_path(),
            "-nostdin",
            "-v",
            "info",
            "-i",
            str(path),
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    timestamps = re.findall(r"time=(\d+):(\d\d):(\d\d\.\d+)", result.stderr)
    if len(timestamps) == 0:
        raise AssertionError(f"ffmpeg reported no duration for {path}")
    hours, minutes, seconds = timestamps[-1]
    return round((int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000)


def write_tsv(path: Path, rows: Sequence[tuple[str, ...]]) -> None:
    """Write a MuTox-shaped annotations TSV."""
    lines = ["\t".join(TSV_COLUMNS)]
    lines.extend("\t".join(row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """A cache directory pre-populated with the default annotations TSV."""
    write_tsv(tmp_path / "mutox.tsv", DEFAULT_ROWS)
    return tmp_path


class FakeResponse:
    """Minimal stand-in for a streamed `requests` response."""

    def __init__(self, content: bytes, status_error: Exception | None = None) -> None:
        self.content = content
        self._status_error = status_error

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self._status_error is not None:
            raise self._status_error

    def iter_content(self, chunk_size: int = 1) -> Iterator[bytes]:
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]


@pytest.fixture
def fake_get(monkeypatch: pytest.MonkeyPatch, source_audio_bytes: bytes) -> list[str]:
    """Patch `requests.get` in the dataset module; returns the list of URLs hit."""
    requested: list[str] = []

    def _get(url: str, **kwargs: Any) -> FakeResponse:
        requested.append(url)
        return FakeResponse(source_audio_bytes)

    monkeypatch.setattr("mutox.dataset.requests.get", _get)
    return requested

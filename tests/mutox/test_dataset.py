"""Tests for the MuTox dataset builder."""

from pathlib import Path
from typing import Any

import pytest
import requests
from inspect_ai.model import ContentAudio, ContentText

from mutox.constants import (
    BAD_FILES_NAME,
    LANGUAGE_CODES,
    MILLISECOND_LANGUAGES,
    PROMPT_TEXT,
    SAMPLE_RATE_HZ,
)
from mutox.dataset import (
    build_dataset,
    download_annotations,
    get_ffmpeg_path,
    is_invalid_audio_file,
    load_bad_files,
    parse_url_segment,
    register_bad_file,
    segment_bounds_seconds,
)
from tests.mutox.conftest import (
    clip_duration_ms,
    DEFAULT_ROWS,
    FakeResponse,
    write_tsv,
)


class TestParseUrlSegment:
    @pytest.mark.parametrize(
        "segment,expected",
        [
            ("http://a.test/x.mp3 100 200", ("http://a.test/x.mp3", 100, 200)),
            ("http://a.test/x.mp3 100.0 200.9", ("http://a.test/x.mp3", 100, 200)),
        ],
    )
    def test_parses_valid_segments(
        self, segment: str, expected: tuple[str, int, int]
    ) -> None:
        assert parse_url_segment(segment) == expected

    @pytest.mark.parametrize(
        "segment",
        [
            "",
            "1.5",
            "http://a.test/x.mp3",
            "http://a.test/x.mp3 100",
            "http://a.test/x.mp3 100 200 300",
            "http://a.test/x.mp3 abc 200",
            "http://a.test/x.mp3 200 100",
            "http://a.test/x.mp3 200 200",
            "http://a.test/x.mp3 -50 200",
        ],
    )
    def test_rejects_malformed_segments(self, segment: str) -> None:
        assert parse_url_segment(segment) is None


class TestBadFileRegistry:
    def test_round_trips(self, tmp_path: Path) -> None:
        assert load_bad_files(tmp_path) == set()
        register_bad_file("abc", tmp_path)
        register_bad_file("def", tmp_path)
        assert load_bad_files(tmp_path) == {"abc", "def"}

    def test_creates_missing_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "does" / "not" / "exist"
        register_bad_file("abc", nested)
        assert load_bad_files(nested) == {"abc"}


class TestIsInvalidAudioFile:
    def test_missing_file_is_invalid(self, tmp_path: Path) -> None:
        assert is_invalid_audio_file(tmp_path / "nope.mp3", get_ffmpeg_path())

    def test_empty_file_is_invalid(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.mp3"
        empty.touch()
        assert is_invalid_audio_file(empty, get_ffmpeg_path())

    def test_garbage_file_is_invalid(self, tmp_path: Path) -> None:
        garbage = tmp_path / "garbage.mp3"
        garbage.write_bytes(b"definitely not audio" * 100)
        assert is_invalid_audio_file(garbage, get_ffmpeg_path())

    def test_real_clip_is_valid(
        self, tmp_path: Path, source_audio_bytes: bytes
    ) -> None:
        good = tmp_path / "good.mp3"
        good.write_bytes(source_audio_bytes)
        assert not is_invalid_audio_file(good, get_ffmpeg_path())


class TestBuildDataset:
    def test_maps_labels_to_yes_no_targets(
        self, cache_dir: Path, fake_get: list[str]
    ) -> None:
        dataset = build_dataset("English", cache_dir=cache_dir)
        targets = {sample.id: sample.target for sample in dataset}
        assert targets == {"eng_toxic": "Yes", "eng_clean": "No"}

    def test_filters_to_devtest_and_language(
        self, cache_dir: Path, fake_get: list[str]
    ) -> None:
        dataset = build_dataset("English", cache_dir=cache_dir)
        ids = {sample.id for sample in dataset}
        assert "eng_train" not in ids, "train partition must be excluded"
        assert "spa_clean" not in ids, "other languages must be excluded"

    def test_sample_carries_audio_then_prompt(
        self, cache_dir: Path, fake_get: list[str]
    ) -> None:
        dataset = build_dataset("English", cache_dir=cache_dir, limit=1)
        messages = dataset[0].input
        assert isinstance(messages, list)
        content = messages[0].content
        assert isinstance(content, list)

        assert isinstance(content[0], ContentAudio)
        assert content[0].format == "mp3"
        assert Path(content[0].audio).exists()
        assert isinstance(content[1], ContentText)
        assert content[1].text == PROMPT_TEXT

    def test_sample_metadata(self, cache_dir: Path, fake_get: list[str]) -> None:
        dataset = build_dataset("English", cache_dir=cache_dir)
        toxic = next(s for s in dataset if s.id == "eng_toxic")
        clean = next(s for s in dataset if s.id == "eng_clean")

        assert toxic.metadata == {
            "lang": "eng",
            "language": "English",
            "toxic": True,
            "transcript": "clean text",
        }
        assert clean.metadata is not None
        assert clean.metadata["toxic"] is False

    def test_clip_is_trimmed_to_segment_length(
        self, cache_dir: Path, fake_get: list[str]
    ) -> None:
        # eng_toxic asks for 500-1500ms out of a three second source.
        build_dataset("English", cache_dir=cache_dir, limit=1)

        duration_ms = clip_duration_ms(cache_dir / "eng_toxic.mp3")
        assert duration_ms == pytest.approx(1000, abs=50)

    def test_respects_limit(self, cache_dir: Path, fake_get: list[str]) -> None:
        dataset = build_dataset("English", cache_dir=cache_dir, limit=1)
        assert len(dataset) == 1
        assert len(fake_get) == 1, "limit must also bound how much audio is fetched"

    def test_rejects_non_positive_limit(self, cache_dir: Path) -> None:
        with pytest.raises(ValueError, match="limit must be positive"):
            build_dataset("English", cache_dir=cache_dir, limit=0)

    def test_rejects_unknown_language(self, cache_dir: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported language"):
            build_dataset("Klingon", cache_dir=cache_dir)

    def test_skips_previously_registered_bad_files(
        self, cache_dir: Path, fake_get: list[str]
    ) -> None:
        register_bad_file("eng_toxic", cache_dir)
        dataset = build_dataset("English", cache_dir=cache_dir)
        assert [s.id for s in dataset] == ["eng_clean"]
        assert fake_get == ["http://audio.test/b.mp3"]

    def test_reuses_cached_clips_without_refetching(
        self, cache_dir: Path, fake_get: list[str]
    ) -> None:
        build_dataset("English", cache_dir=cache_dir)
        assert len(fake_get) == 2
        build_dataset("English", cache_dir=cache_dir)
        assert len(fake_get) == 2, "second build must hit the cache"

    def test_registers_and_skips_failed_download(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        source_audio_bytes: bytes,
    ) -> None:
        def _get(url: str, **kwargs: Any) -> FakeResponse:
            if url.endswith("a.mp3"):
                return FakeResponse(b"", requests.HTTPError("404 Not Found"))
            return FakeResponse(source_audio_bytes)

        monkeypatch.setattr("mutox.dataset.requests.get", _get)

        dataset = build_dataset("English", cache_dir=cache_dir)

        assert [s.id for s in dataset] == ["eng_clean"]
        assert "eng_toxic" in load_bad_files(cache_dir)
        assert not (cache_dir / "eng_toxic.mp3").exists()

    def test_registers_and_skips_undecodable_source(
        self, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "mutox.dataset.requests.get",
            lambda url, **kwargs: FakeResponse(b"not audio at all" * 50),
        )
        with pytest.raises(ValueError, match="No MuTox samples could be loaded"):
            build_dataset("English", cache_dir=cache_dir)
        assert load_bad_files(cache_dir) == {"eng_toxic", "eng_clean"}

    def test_registers_and_skips_malformed_segment(
        self, tmp_path: Path, fake_get: list[str]
    ) -> None:
        rows = [
            ("eng_bad", "eng", "devtest", "1.5", "stray value", "1"),
            DEFAULT_ROWS[1],
        ]
        write_tsv(tmp_path / "mutox.tsv", rows)

        dataset = build_dataset("English", cache_dir=tmp_path)

        assert [s.id for s in dataset] == ["eng_clean"]
        assert "eng_bad" in load_bad_files(tmp_path)
        assert fake_get == ["http://audio.test/b.mp3"]

    def test_raises_when_nothing_loadable(
        self, tmp_path: Path, fake_get: list[str]
    ) -> None:
        write_tsv(tmp_path / "mutox.tsv", [DEFAULT_ROWS[3]])
        with pytest.raises(ValueError, match="No MuTox samples could be loaded"):
            build_dataset("English", cache_dir=tmp_path)


class TestOfflineMode:
    def test_uses_only_cached_clips(
        self, cache_dir: Path, fake_get: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        build_dataset("English", cache_dir=cache_dir, limit=1)
        (cache_dir / "eng_clean.mp3").unlink(missing_ok=True)

        def _explode(*args: object, **kwargs: object) -> None:
            raise AssertionError("offline mode must not hit the network")

        monkeypatch.setattr("mutox.dataset.requests.get", _explode)

        dataset = build_dataset("English", cache_dir=cache_dir, allow_download=False)
        assert [s.id for s in dataset] == ["eng_toxic"]

    def test_requires_cached_annotations(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="allow_download is False"):
            build_dataset("English", cache_dir=tmp_path, allow_download=False)


class TestDownloadAnnotations:
    def test_rejects_checksum_mismatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "mutox.dataset.requests.get",
            lambda url, **kwargs: FakeResponse(b"wrong content"),
        )
        destination = tmp_path / "mutox.tsv"

        with pytest.raises(ValueError, match="Checksum mismatch"):
            download_annotations(destination)

        assert not destination.exists(), "a bad download must not be left cached"
        assert list(tmp_path.glob("*.tmp")) == []

    def test_skips_when_already_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _explode(*args: object, **kwargs: object) -> None:
            raise AssertionError("must not download when the file is cached")

        monkeypatch.setattr("mutox.dataset.requests.get", _explode)
        destination = tmp_path / "mutox.tsv"
        destination.write_text("cached")

        download_annotations(destination)

        assert destination.read_text() == "cached"


class TestSegmentBoundsSeconds:
    """Offset unit handling.

    Units differ by language: milliseconds for English and Spanish, 16 kHz
    samples for the other 28. Reading everything as milliseconds seeks past the
    end of the source and yields empty clips.
    """

    @pytest.mark.parametrize("lang_code", ["eng", "spa"])
    def test_english_and_spanish_are_milliseconds(self, lang_code: str) -> None:
        assert segment_bounds_seconds(504288, 509502, lang_code) == (504.288, 509.502)

    @pytest.mark.parametrize("lang_code", ["deu", "cmn", "arb", "swh", "pes"])
    def test_other_languages_are_16khz_samples(self, lang_code: str) -> None:
        start, end = segment_bounds_seconds(29592576, 29671392, lang_code)
        assert start == pytest.approx(1849.536)
        assert end == pytest.approx(1854.462)
        # A real utterance, not the 79 seconds a millisecond reading implies.
        assert end - start == pytest.approx(4.926)

    def test_millisecond_languages_are_exactly_english_and_spanish(self) -> None:
        assert MILLISECOND_LANGUAGES == {"eng", "spa"}

    def test_sample_rate_is_16khz(self) -> None:
        assert SAMPLE_RATE_HZ == 16000

    def test_every_language_code_is_covered(self) -> None:
        for lang_code in LANGUAGE_CODES.values():
            start, end = segment_bounds_seconds(16000, 32000, lang_code)
            assert end > start


def test_language_codes_cover_thirty_languages() -> None:
    assert len(LANGUAGE_CODES) == 30
    assert len(set(LANGUAGE_CODES.values())) == 30
    # MuTox tags Western Persian as `pes`; `fas` matches no rows.
    assert LANGUAGE_CODES["Western_Persian"] == "pes"


def test_bad_files_name_is_unchanged() -> None:
    assert BAD_FILES_NAME == "bad_audio_files.txt"


class TestSeededSampling:
    """Seeded sampling of a language split.

    Toxic labels cluster by position in the TSV, so a bounded run needs a seed
    to be representative of the split.
    """

    @staticmethod
    def _rows() -> list[tuple[str, ...]]:
        return [
            (f"id{i}", "eng", "devtest", f"http://audio.test/{i}.mp3 0 1000", "t", "0")
            for i in range(6)
        ]

    def test_unseeded_limit_takes_the_head_in_file_order(
        self, tmp_path: Path, fake_get: list[str]
    ) -> None:
        write_tsv(tmp_path / "mutox.tsv", self._rows())

        dataset = build_dataset("English", cache_dir=tmp_path, limit=3)

        assert [s.id for s in dataset] == ["id0", "id1", "id2"]

    def test_seed_draws_a_different_sample(
        self, tmp_path: Path, fake_get: list[str]
    ) -> None:
        write_tsv(tmp_path / "mutox.tsv", self._rows())

        dataset = build_dataset("English", cache_dir=tmp_path, limit=3, seed=7)

        assert len(dataset) == 3
        assert [s.id for s in dataset] != ["id0", "id1", "id2"]

    def test_same_seed_gives_the_same_sample(
        self, tmp_path: Path, fake_get: list[str]
    ) -> None:
        write_tsv(tmp_path / "mutox.tsv", self._rows())

        first = build_dataset("English", cache_dir=tmp_path, limit=3, seed=7)
        second = build_dataset("English", cache_dir=tmp_path, limit=3, seed=7)

        assert [s.id for s in first] == [s.id for s in second]

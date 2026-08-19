"""End-to-end tests for the MuTox task, driven by mockllm.

These never contact a real model or the network: the dataset is built from a
fixture TSV with stubbed audio downloads, and generation is mocked.
"""

from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import (
    ChatMessage,
    ContentAudio,
    GenerateConfig,
    get_model,
    ModelOutput,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from mutox.constants import SYSTEM_MESSAGE
from mutox.dataset import build_dataset
from mutox.mutox import mutox


def fixed_answer_model(answer: str):
    """A mockllm model that replies with `answer` to every request."""

    def respond(
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        return ModelOutput.from_content(model="mockllm", content=answer)

    return get_model("mockllm/model", custom_outputs=respond)


@pytest.fixture
def task_cache(cache_dir: Path, fake_get: list[str]) -> Path:
    """Materialise the fixture clips so the task can be built offline."""
    build_dataset("English", cache_dir=cache_dir)
    return cache_dir


class TestTaskConstruction:
    def test_builds_expected_task(self, task_cache: Path) -> None:
        task = mutox(language="English", cache_dir=str(task_cache))

        assert len(task.dataset) == 2
        assert task.scorer is not None
        assert task.version is not None

    def test_rejects_unknown_language(self, task_cache: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported language"):
            mutox(language="Elvish", cache_dir=str(task_cache))

    def test_system_message_constrains_the_answer_format(self) -> None:
        assert "Yes" in SYSTEM_MESSAGE
        assert "No" in SYSTEM_MESSAGE


class TestEndToEnd:
    # The fixture has one toxic ("Yes") and one clean ("No") sample, so a model
    # that always gives the same answer gets exactly half of them right.
    @pytest.mark.parametrize("answer", ["Yes", "No"])
    def test_scores_a_full_run(self, task_cache: Path, answer: str) -> None:
        logs = inspect_eval(
            mutox(language="English", cache_dir=str(task_cache)),
            model=fixed_answer_model(answer),
            display="none",
        )
        log = logs[0]

        assert log.status == "success", log.error
        assert log.results is not None
        assert log.results.completed_samples == 2

        accuracy = next(
            metric.value
            for score in log.results.scores
            for name, metric in score.metrics.items()
            if name == "accuracy"
        )
        assert accuracy == 0.5

    def test_reports_toxic_class_metrics(self, task_cache: Path) -> None:
        # Always answering "Yes" finds the one toxic clip and flags the clean one.
        logs = inspect_eval(
            mutox(language="English", cache_dir=str(task_cache)),
            model=fixed_answer_model("Yes"),
            display="none",
        )
        results = logs[0].results
        assert results is not None
        metrics = {
            name: metric.value
            for score in results.scores
            for name, metric in score.metrics.items()
        }

        assert metrics["recall_toxic"] == 1.0
        assert metrics["precision_toxic"] == 0.5
        assert metrics["no_answer_rate"] == 0.0

    def test_unparseable_answers_are_recorded(self, task_cache: Path) -> None:
        logs = inspect_eval(
            mutox(language="English", cache_dir=str(task_cache)),
            model=fixed_answer_model("I would rather not say."),
            display="none",
        )
        results = logs[0].results
        assert results is not None
        metrics = {
            name: metric.value
            for score in results.scores
            for name, metric in score.metrics.items()
        }

        assert metrics["no_answer_rate"] == 1.0
        assert metrics["accuracy"] == 0.0

    def test_audio_reaches_the_model(self, task_cache: Path) -> None:
        logs = inspect_eval(
            mutox(language="English", cache_dir=str(task_cache)),
            model=fixed_answer_model("Yes"),
            display="none",
            log_samples=True,
        )
        samples = logs[0].samples
        assert samples is not None

        user_message = next(m for m in samples[0].messages if m.role == "user")
        assert isinstance(user_message.content, list)
        assert any(isinstance(part, ContentAudio) for part in user_message.content), (
            "the user message must carry audio content"
        )


def alternating_answer_model(first: str, second: str):
    """A mockllm model that answers `first` then `second` for each sample.

    Keyed by message content, so with `epochs=2` every sample is answered one
    way in its first epoch and the other way in its second. Message ids are a
    fresh uuid per request, so they cannot be part of the key.
    """
    seen: dict[str, int] = {}

    def respond(
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        key = str([message.content for message in input])
        count = seen.get(key, 0)
        seen[key] = count + 1
        return ModelOutput.from_content(
            model="mockllm", content=first if count == 0 else second
        )

    return get_model("mockllm/model", custom_outputs=respond, memoize=False)


class TestEpochReduction:
    """Metrics must aggregate over Score.value, the only epoch-reduced field.

    `Score.answer` is dropped (set to None) when a sample's epochs disagree, so
    metrics that read it silently miscount exactly the inconsistent samples.
    """

    def test_disagreeing_epochs_are_counted_fractionally(
        self, task_cache: Path
    ) -> None:
        logs = inspect_eval(
            mutox(language="English", cache_dir=str(task_cache)),
            model=alternating_answer_model("Yes", "No"),
            epochs=2,
            display="none",
        )

        log = logs[0]
        assert log.status == "success"
        assert log.results is not None
        metrics = {
            name: metric.value
            for score in log.results.scores
            for name, metric in score.metrics.items()
        }

        # One toxic and one clean sample, each answered "Yes" once and "No"
        # once: tp = fp = fn = 0.5, so precision and recall are both 0.5.
        # Reading the prediction from Score.answer would give 0.0 for both.
        assert metrics["precision_toxic"] == 0.5
        assert metrics["recall_toxic"] == 0.5
        assert metrics["accuracy"] == 0.5
        assert metrics["no_answer_rate"] == 0.0

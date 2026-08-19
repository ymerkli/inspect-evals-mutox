"""Tests for the MuTox scorer and its toxic-class metrics."""

from typing import cast

import pytest
from inspect_ai.model import ModelName, ModelOutput
from inspect_ai.scorer import (
    Metric,
    MetricProtocol,
    SampleScore,
    Score,
    Target,
)
from inspect_ai.solver import TaskState

from mutox.scorers import (
    extract_yes_no,
    f1_toxic,
    NO,
    NO_ANSWER,
    no_answer_rate,
    precision_toxic,
    quasi_exact_match,
    recall_toxic,
    YES,
)


class TestExtractYesNo:
    @pytest.mark.parametrize(
        "completion,expected",
        [
            ("Yes", "yes"),
            ("Yes.", "yes"),
            ("yes, it does", "yes"),
            ("No", "no"),
            ("No\n", "no"),
            ("  NO!  ", "no"),
            ("**Yes**", "yes"),
            ("The answer is: yes", "yes"),
            # First occurrence wins when both appear.
            ("Yes and no", "yes"),
            ("no, actually yes", "no"),
            # Not whole words.
            ("nope", None),
            ("yesterday", None),
            ("", None),
            ("I'm not sure", None),
            ("I cannot help with that request", None),
        ],
    )
    def test_normalises_responses(self, completion: str, expected: str | None) -> None:
        assert extract_yes_no(completion) == expected


def make_state(completion: str) -> TaskState:
    """A TaskState carrying just the model completion the scorer reads."""
    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id="s1",
        epoch=1,
        input="",
        messages=[],
    )
    state.output = ModelOutput.from_content("mockllm/model", completion)
    return state


class TestQuasiExactMatch:
    @pytest.mark.parametrize(
        "completion,target,expected_value,expected_answer",
        [
            (
                "Yes",
                "Yes",
                {"correct": 1.0, "predicted_toxic": 1.0, "answered": 1.0},
                "yes",
            ),
            (
                "yes, clearly toxic",
                "Yes",
                {"correct": 1.0, "predicted_toxic": 1.0, "answered": 1.0},
                "yes",
            ),
            (
                "No",
                "No",
                {"correct": 1.0, "predicted_toxic": 0.0, "answered": 1.0},
                "no",
            ),
            (
                "No",
                "Yes",
                {"correct": 0.0, "predicted_toxic": 0.0, "answered": 1.0},
                "no",
            ),
            (
                "Yes",
                "No",
                {"correct": 0.0, "predicted_toxic": 1.0, "answered": 1.0},
                "yes",
            ),
            (
                "I'm not sure",
                "Yes",
                {"correct": 0.0, "predicted_toxic": 0.0, "answered": 0.0},
                "",
            ),
            ("", "No", {"correct": 0.0, "predicted_toxic": 0.0, "answered": 0.0}, ""),
        ],
    )
    async def test_scores_responses(
        self,
        completion: str,
        target: str,
        expected_value: dict[str, float],
        expected_answer: str,
    ) -> None:
        score_fn = quasi_exact_match()
        score = await score_fn(make_state(completion), Target(target))

        assert score is not None
        assert score.value == expected_value
        assert score.answer == expected_answer
        assert score.explanation == completion


def compute(metric: Metric, scores: list[SampleScore]) -> float:
    """Call a metric, narrowing it to the SampleScore-based protocol."""
    return cast(float, cast(MetricProtocol, metric)(scores))


def sample_score(toxic: bool, answer: str, sample_id: str = "s") -> SampleScore:
    """A scored sample as the eval produces it.

    `answer` is the normalised prediction ("yes", "no", or "" for no answer);
    the gold label lives in sample metadata. The values the metrics aggregate
    live in `Score.value`, which is what Inspect reduces across epochs.
    """
    gold = YES if toxic else NO
    value = {
        "correct": 1.0 if answer == gold else 0.0,
        "predicted_toxic": 1.0 if answer == YES else 0.0,
        "answered": 0.0 if answer == NO_ANSWER else 1.0,
    }
    return SampleScore(
        score=Score(value=value, answer=answer),
        sample_id=sample_id,
        sample_metadata={"lang": "eng", "toxic": toxic},
    )


class TestToxicClassMetrics:
    def test_perfect_predictions(self) -> None:
        scores = [
            sample_score(toxic=True, answer=YES),
            sample_score(toxic=True, answer=YES),
            sample_score(toxic=False, answer=NO),
        ]
        assert compute(precision_toxic(), scores) == 1.0
        assert compute(recall_toxic(), scores) == 1.0
        assert compute(f1_toxic(), scores) == 1.0

    def test_counts_false_positives_and_negatives(self) -> None:
        # 2 true positives, 1 false negative (missed toxic), 1 false positive.
        scores = [
            sample_score(toxic=True, answer=YES),
            sample_score(toxic=True, answer=YES),
            sample_score(toxic=True, answer=NO),
            sample_score(toxic=False, answer=YES),
            sample_score(toxic=False, answer=NO),
        ]
        assert compute(precision_toxic(), scores) == pytest.approx(2 / 3)
        assert compute(recall_toxic(), scores) == pytest.approx(2 / 3)
        assert compute(f1_toxic(), scores) == pytest.approx(2 / 3)

    def test_no_answer_never_counts_as_a_false_positive(self) -> None:
        # An unparseable response on a clean clip is not a toxicity prediction.
        scores = [
            sample_score(toxic=True, answer=YES),
            sample_score(toxic=False, answer=NO_ANSWER),
        ]
        assert compute(precision_toxic(), scores) == 1.0
        assert compute(recall_toxic(), scores) == 1.0

    def test_no_answer_counts_as_a_false_negative(self) -> None:
        scores = [
            sample_score(toxic=True, answer=NO_ANSWER),
            sample_score(toxic=False, answer=NO),
        ]
        assert compute(recall_toxic(), scores) == 0.0
        assert compute(precision_toxic(), scores) == 0.0

    def test_all_clean_gold_gives_zero_not_an_error(self) -> None:
        scores = [sample_score(toxic=False, answer=NO)]
        assert compute(precision_toxic(), scores) == 0.0
        assert compute(recall_toxic(), scores) == 0.0
        assert compute(f1_toxic(), scores) == 0.0

    def test_requires_toxic_metadata(self) -> None:
        scores = [
            SampleScore(
                score=Score(
                    value={"correct": 1.0, "predicted_toxic": 1.0, "answered": 1.0},
                    answer=YES,
                ),
                sample_id="s",
                sample_metadata={"lang": "eng"},
            )
        ]
        with pytest.raises(ValueError, match="no 'toxic' metadata"):
            compute(f1_toxic(), scores)


class TestNoAnswerRate:
    def test_counts_unparseable_responses(self) -> None:
        scores = [
            sample_score(toxic=True, answer=YES),
            sample_score(toxic=False, answer=NO_ANSWER),
            sample_score(toxic=False, answer=NO_ANSWER),
            sample_score(toxic=False, answer=YES),
        ]
        assert compute(no_answer_rate(), scores) == 0.5

    def test_zero_when_all_answered(self) -> None:
        assert compute(no_answer_rate(), [sample_score(toxic=True, answer=YES)]) == 0.0

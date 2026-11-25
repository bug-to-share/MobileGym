import pytest

from rllm.types import Episode

from .evaluator import mobilegym_evaluator


def test_wrong_answer_removes_submitted_check_from_base_reward(monkeypatch):
    monkeypatch.setenv("MOBILEGYM_REWARD_MODE", "progress")
    monkeypatch.setenv("MOBILEGYM_FALSE_COMPLETE_MODE", "legacy")
    episode = Episode(
        artifacts={
            "progress": 2 / 3,
            "goal_checks": [
                {"field": "answer_sheet.submitted", "actual": True, "passed": True},
                {"field": "answer_sheet.answers.0", "passed": True},
                {"field": "answer_sheet.answers.1", "passed": False},
            ],
        }
    )

    result = mobilegym_evaluator.evaluate({}, episode)
    breakdown = result.metadata["reward_breakdown"]

    assert breakdown["wrong_answer_submitted"] is True
    assert breakdown["raw_progress"] == pytest.approx(2 / 3)
    assert breakdown["base_reward"] == pytest.approx(0.5)
    assert result.reward == pytest.approx(0.5)
    assert episode.artifacts["reward_breakdown"] == breakdown

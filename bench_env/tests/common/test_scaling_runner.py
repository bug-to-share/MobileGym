import json
from dataclasses import asdict
from unittest.mock import Mock

import pytest

from bench_env.env.base import Action, ActionType, Observation
from bench_env.llm import ChatResult
from bench_env.runner.base import EpisodeResult, ExecutionResult
from bench_env.runner.test_time_scaling import (
    ActionSample,
    BaseActionVerifier,
    BaseTrajectoryVerifier,
    BestOfNAgent,
    HeuristicActionVerifier,
    StepScalingConfig,
    TokenUsage,
    TrajectoryCandidate,
    VerificationResult,
    VLMVerifier,
    best_of_n_step,
    best_of_n_trajectory,
    canonicalize_action,
    rerank_trajectories,
)


OBS = Observation(screenshot_bytes=b"screen", route={"path": "/private"})


def test_action_keys_compare_behavior_and_keep_meaningful_values():
    a = Action(ActionType.TYPE, {"value": " a ", "point": (10, 20)}, thought="first")
    b = Action(
        ActionType.TYPE,
        {"point": [10.0, 20.0], "clear": False, "value": " a "},
        thought="second",
    )
    assert canonicalize_action(a) == canonicalize_action(b)
    assert canonicalize_action(a) != canonicalize_action(
        Action.type_text("a", [10, 20])
    )
    assert canonicalize_action(Action.wait(1)) != canonicalize_action(Action.wait(2))
    assert canonicalize_action(Action(ActionType.WAIT)) == canonicalize_action(
        Action.wait(1.0)
    )


def test_consensus_uses_full_budget_and_strict_threshold_before_early_stop():
    first, second = Action.back(thought="first"), Action.back(thought="second")
    sample = Mock(
        side_effect=[
            ActionSample(a, TokenUsage(2, 3)) for a in (first, second, second, second)
        ]
    )
    verifier = Mock(spec=BaseActionVerifier)
    result = best_of_n_step(
        sample,
        OBS,
        "go back",
        verifier=verifier,
        config=StepScalingConfig(n=5, seed=11, consensus_threshold=0.6),
    )

    assert result.action is first
    assert [call.args for call in sample.call_args_list] == [
        (0.1, 11),
        (0.4, 12),
        (0.7, 13),
        (0.1, 14),
    ]
    verifier.score.assert_not_called()
    telemetry = asdict(result.telemetry)
    assert telemetry.pop("latency_s") >= 0
    assert telemetry == {
        "candidates_generated": 4,
        "unique_candidates": 1,
        "verifier_calls": 0,
        "early_stopped": True,
        "total_prompt_tokens": 8,
        "total_completion_tokens": 12,
    }


def test_reranking_scores_each_unique_action_once_and_tracks_usage():
    first, duplicate, preferred = (
        Action.click([10, 20]),
        Action.click([10.0, 20]),
        Action.home(),
    )
    sample = Mock(
        side_effect=[
            ActionSample(a, TokenUsage(3, 2)) for a in (first, duplicate, preferred)
        ]
    )
    verifier = Mock(spec=BaseActionVerifier)
    verifier.score.side_effect = [
        VerificationResult(0.2, TokenUsage(7, 1)),
        VerificationResult(0.9, TokenUsage(11, 4)),
    ]

    result = best_of_n_step(
        sample,
        OBS,
        "go home",
        verifier=verifier,
        config=StepScalingConfig(consensus_threshold=1),
    )

    assert result.action is preferred
    assert [call.args for call in verifier.score.call_args_list] == [
        (first, OBS, "go home"),
        (preferred, OBS, "go home"),
    ]
    assert result.telemetry.candidates_generated == 3
    assert result.telemetry.unique_candidates == result.telemetry.verifier_calls == 2
    assert result.telemetry.early_stopped is False
    assert result.telemetry.total_prompt_tokens == 27
    assert result.telemetry.total_completion_tokens == 11


def test_equal_scores_break_ties_by_votes_then_sampling_order():
    first, second, third = Action.back(), Action.home(), Action.click([15, 23])
    sample = Mock(
        side_effect=[
            ActionSample(action) for action in (first, second, third, third, second)
        ]
    )
    result = best_of_n_step(
        sample, OBS, "task", config=StepScalingConfig(n=5, consensus_threshold=1)
    )
    assert result.action is second


def test_bad_actions_cannot_win_consensus_and_heuristic_prefers_valid_tap():
    invalid = Action.click([23, 1100])
    valid = Action.click([23, 25])
    sample = Mock(side_effect=[ActionSample(a) for a in (invalid, invalid, valid)])
    result = best_of_n_step(sample, OBS, "tap")
    assert result.action is valid
    assert result.telemetry.candidates_generated == 3
    assert result.telemetry.verifier_calls == 2
    assert (
        HeuristicActionVerifier().score(Action(ActionType.NOOP), OBS, "tap").score == 0
    )


def test_vlm_verifier_uses_visual_context_and_provider_usage():
    llm = Mock()
    llm.chat.return_value = ChatResult(
        '<think>reasoning</think>\n```json\n{"score": 0.75}\n```',
        raw={"usage": {"prompt_tokens": 20, "completion_tokens": 5}},
    )
    action = Action.click([10, 20], thought="candidate reasoning")

    result = VLMVerifier(llm, model_args={"max_tokens": 48}).score(
        action, OBS, "Tap icon"
    )

    assert result == VerificationResult(0.75, TokenUsage(20, 5))
    request = llm.chat.call_args.kwargs
    assert request["args"] == {"temperature": 0, "max_tokens": 48, "stream": False}
    text, image = request["messages"][1]["content"]
    assert json.loads(text["text"]) == {
        "task": "Tap icon",
        "action": "CLICK",
        "data": {"point": [10, 20]},
    }
    assert image["image_url"]["url"] == OBS.image_data_url
    assert "/private" not in str(request)


@pytest.mark.parametrize(
    "content", ["invalid", "[]", "{}", '{"score": true}', '{"score": 1.5}']
)
def test_vlm_verifier_rejects_invalid_scores(content):
    llm = Mock()
    llm.chat.return_value = ChatResult(content)
    with pytest.raises(ValueError, match="verifier"):
        VLMVerifier(llm).score(Action.back(), Observation(), "go back")


def test_runner_agent_records_only_the_winning_sample_and_then_resets():
    llm = Mock()
    llm.chat.side_effect = [
        ChatResult('{"action": "WAIT", "value": "not-a-number"}'),
        ChatResult('{"action": "WAIT", "value": "not-a-number"}'),
        ChatResult(
            '<answer>{"action": "TAP", "point": [10, 20]}</answer>',
            raw={"usage": {"prompt_tokens": 5, "completion_tokens": 4}},
        ),
    ]
    agent = BestOfNAgent(llm, max_history_images=2)
    agent.reset("Tap")
    winner = agent.act(OBS)

    assert winner.action_type == ActionType.CLICK
    assert len(agent.history) == len(agent.step_telemetry) == 1
    assert agent.history[0].action is winner
    assert agent.history[0].llm_response == winner.raw_response
    assert agent.history[0].llm_prompt is llm.chat.call_args.kwargs["messages"]
    assert agent.step_telemetry[0].unique_candidates == 2
    assert agent.step_telemetry[0].total_prompt_tokens == 5
    assert agent.step_telemetry[0].total_completion_tokens == 4

    agent.reset("Another task")
    assert not agent.history
    assert not agent.step_telemetry


def test_runner_agent_parser_aliases_vote_as_one_action():
    llm = Mock()
    llm.chat.side_effect = [
        ChatResult('{"action": "TAP", "point": [10, 20]}'),
        ChatResult('{"action": "CLICK", "point": [10.0, 20.0]}'),
    ]
    agent = BestOfNAgent(llm)
    agent.reset("Tap")

    assert agent.act(OBS) == agent.parse_response(
        '{"action": "TAP", "point": [10, 20]}'
    )
    assert llm.chat.call_count == 2
    assert agent.step_telemetry[0].early_stopped


def make_trajectory(trial, *, answer=None, vote_key=None, trace=None):
    execution = ExecutionResult(
        steps=len(trace or []),
        trace=trace or [],
        runtime_s=0,
        finished=True,
        truncated=False,
        stop_reason="COMPLETE",
        agent_answer=answer,
    )
    result = EpisodeResult(
        "test.task", "instruction", "test", execution, trial_id=trial
    )
    return TrajectoryCandidate(result, TokenUsage(10, 3), vote_key)


def test_majority_groups_normalized_answers_and_picks_first_matching_rollout():
    choices = [
        make_trajectory(0, answer="blue"),
        make_trajectory(1, answer=" green "),
        make_trajectory(2, answer="green"),
    ]
    result = rerank_trajectories(choices)

    assert result.selected is choices[1]
    assert result.telemetry.candidates_generated == 3
    assert result.telemetry.unique_candidates == 2
    assert result.telemetry.verifier_calls == 0
    assert result.telemetry.early_stopped is False
    assert result.telemetry.total_prompt_tokens == 30
    assert result.telemetry.total_completion_tokens == 9
    assert result.telemetry.latency_s >= 0


def test_no_answer_groups_equivalent_action_traces_ignoring_reasoning():
    step = {
        "action_type": ActionType.CLICK,
        "data": {"point": [1, 2]},
        "thought": "tap",
    }
    choices = [
        make_trajectory(0, trace=[{"action_type": ActionType.BACK, "data": {}}]),
        make_trajectory(1, trace=[step]),
        make_trajectory(
            2, trace=[{**step, "data": {"point": [1.0, 2.0]}, "thought": "try"}]
        ),
    ]
    result = rerank_trajectories(choices)
    assert result.selected is choices[1]
    assert result.telemetry.unique_candidates == 2


@pytest.mark.asyncio
async def test_trajectory_best_of_n_runs_all_trials_and_verifier_can_override_majority():
    trials = []

    async def rollout(trial):
        trials.append(trial)
        return make_trajectory(
            trial, vote_key="same outcome" if trial < 2 else "different"
        )

    verifier = Mock(spec=BaseTrajectoryVerifier)
    verifier.score.side_effect = [
        VerificationResult(0.1, TokenUsage(5, 2)),
        VerificationResult(0.3, TokenUsage(6, 2)),
        VerificationResult(0.7, TokenUsage(7, 2)),
    ]

    result = await best_of_n_trajectory(rollout, k=3, verifier=verifier)

    assert trials == [0, 1, 2]
    assert result.selected.result.trial_id == 2
    assert [
        call.args[0].result.trial_id for call in verifier.score.call_args_list
    ] == trials
    assert result.telemetry.candidates_generated == result.telemetry.verifier_calls == 3
    assert result.telemetry.unique_candidates == 2
    assert result.telemetry.total_prompt_tokens == 48
    assert result.telemetry.total_completion_tokens == 15
    assert result.telemetry.latency_s >= 0


@pytest.mark.parametrize(
    "config",
    [
        {"n": 0},
        {"temperatures": ()},
        {"temperatures": (3,)},
        {"consensus_threshold": 2},
    ],
)
def test_invalid_step_config(config):
    with pytest.raises(ValueError):
        StepScalingConfig(**config)


@pytest.mark.asyncio
async def test_empty_trajectory_search_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        rerank_trajectories([])
    with pytest.raises(ValueError, match="positive"):
        await best_of_n_trajectory(Mock(), k=0)

"""Opt-in Best-of-N selection for MobileGym actions and candidate episodes.

Pass a BestOfNAgent to an existing runner for step scaling. For trajectory
scaling, pass an async callback that runs an independent episode for each trial
to best_of_n_trajectory, or pass completed trials to rerank_trajectories.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from bench_env.agent.base import AgentConfig
from bench_env.agent.generic_v2 import GenericAgentV2
from bench_env.env.base import Action, ActionType, Observation
from bench_env.llm import ChatResult, LLMClient
from bench_env.runner.base import EpisodeResult

__all__ = [
    "ActionSample",
    "BaseActionVerifier",
    "BaseTrajectoryVerifier",
    "BestOfNAgent",
    "HeuristicActionVerifier",
    "ScalingTelemetry",
    "StepScalingConfig",
    "StepScalingResult",
    "TokenUsage",
    "TrajectoryCandidate",
    "TrajectoryScalingResult",
    "VerificationResult",
    "VLMVerifier",
    "best_of_n_step",
    "best_of_n_trajectory",
    "canonicalize_action",
    "rerank_trajectories",
]


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @classmethod
    def from_chat(cls, response: ChatResult) -> TokenUsage:
        """Read provider-reported usage. Missing counts contribute zero."""
        usage = (response.raw or {}).get("usage") or {}
        return cls(usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0)


@dataclass
class ScalingTelemetry:
    candidates_generated: int = 0
    unique_candidates: int = 0
    verifier_calls: int = 0
    early_stopped: bool = False
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    latency_s: float = 0.0

    def add_usage(self, usage: TokenUsage) -> None:
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens


@dataclass(frozen=True)
class VerificationResult:
    score: float
    usage: TokenUsage = TokenUsage()

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError("verifier score must be finite")


class BaseActionVerifier(ABC):
    @abstractmethod
    def score(
        self, action: Action, observation: Observation, task: str
    ) -> VerificationResult:
        """Score a single candidate. Higher scores are preferred."""


@dataclass(frozen=True)
class ActionSample:
    action: Action
    usage: TokenUsage = TokenUsage()


@dataclass(frozen=True)
class StepScalingConfig:
    n: int = 3
    temperatures: tuple[float, ...] = (0.1, 0.4, 0.7)
    seed: int = 0
    consensus_threshold: float = 0.6

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be positive")
        if not self.temperatures or any(
            not 0 <= value <= 2 for value in self.temperatures
        ):
            raise ValueError("temperatures must contain values between 0 and 2")
        if not 0 <= self.consensus_threshold <= 1:
            raise ValueError("consensus_threshold must be between 0 and 1")


@dataclass(frozen=True)
class StepScalingResult:
    action: Action
    telemetry: ScalingTelemetry


_ACTION_DEFAULTS: dict[ActionType, dict[str, Any]] = {
    ActionType.TYPE: {"value": "", "point": None, "clear": False},
    ActionType.WAIT: {"value": 1},
    ActionType.SWIPE: {"duration": 400},
    ActionType.DRAG: {"duration": 400},
    ActionType.LONG_PRESS: {"duration": 800},
    ActionType.COMPLETE: {"return": ""},
    ActionType.ABORT: {"value": ""},
}


def _normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _action_payload(action: Action) -> dict[str, Any]:
    data = {**_ACTION_DEFAULTS.get(action.action_type, {}), **action.data}
    return {"action": action.action_type.value, "data": _normalize_json(data)}


def canonicalize_action(action: Action) -> str:
    """Build a key from parsed action parameters, ignoring model commentary."""
    return json.dumps(
        _action_payload(action),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _valid_point(point: object) -> bool:
    return (
        isinstance(point, (list, tuple))
        and len(point) == 2
        and all(type(coord) in (int, float) and 0 <= coord <= 1000 for coord in point)
    )


def _action_quality(action: Action) -> float:
    kind, data = action.action_type, action.data
    if kind == ActionType.NOOP:
        return 0.0
    if kind in {ActionType.CLICK, ActionType.DOUBLE_TAP, ActionType.LONG_PRESS}:
        return float(_valid_point(data.get("point")))
    if kind in {ActionType.SWIPE, ActionType.DRAG}:
        return float(
            _valid_point(data.get("point1")) and _valid_point(data.get("point2"))
        )
    if kind == ActionType.TYPE:
        point = data.get("point")
        return float(
            isinstance(data.get("value", ""), str)
            and (point is None or _valid_point(point))
        )
    if kind in {ActionType.AWAKE, ActionType.ANSWER, ActionType.INFO}:
        value = data.get("value")
        return float(isinstance(value, str) and bool(value.strip()))
    if kind == ActionType.WAIT:
        duration = data.get("value", 1)
        return (
            0.5 if type(duration) in (int, float) and 0 <= duration < math.inf else 0.0
        )
    if kind == ActionType.ABORT:
        if data.get("value") in ("empty_response", "invalid_json", "invalid_action"):
            return 0.0
        return 0.25
    return 1.0


class HeuristicActionVerifier(BaseActionVerifier):
    """Deterministically favor well-formed actions in the 0–1000 coordinate space."""

    def score(
        self, action: Action, observation: Observation, task: str
    ) -> VerificationResult:
        return VerificationResult(_action_quality(action))


class VLMVerifier(BaseActionVerifier):
    """Score actions with an OpenAI-compatible LLMClient and the current screenshot."""

    def __init__(self, llm: LLMClient, *, model_args: dict[str, Any] | None = None):
        self.llm = llm
        self.model_args = {"temperature": 0, **(model_args or {}), "stream": False}

    def score(
        self, action: Action, observation: Observation, task: str
    ) -> VerificationResult:
        prompt = json.dumps(
            {"task": task, **_action_payload(action)}, ensure_ascii=False
        )
        content = [{"type": "text", "text": prompt}]
        if url := observation.image_data_url:
            content.append({"type": "image_url", "image_url": {"url": url}})
        response = self.llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rate how appropriate the candidate action is for the task and screenshot. "
                        "Use a score from 0 (inappropriate) to 1 (best next action). "
                        'Return a JSON object with a numeric "score" field.'
                    ),
                },
                {"role": "user", "content": content},
            ],
            args=self.model_args,
        )
        text = re.sub(
            r"^\s*<think>.*?</think>",
            "",
            response.content,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()
        if text.startswith("```json") and text.endswith("```"):
            text = text[7:-3].strip()
        try:
            score = json.loads(text)["score"]
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(
                "VLM verifier must return a JSON object with a score"
            ) from exc
        if type(score) not in (int, float) or not 0 <= score <= 1:
            raise ValueError("VLM verifier score must be numeric and between 0 and 1")
        return VerificationResult(score, TokenUsage.from_chat(response))


def best_of_n_step(
    sample: Callable[[float, int], ActionSample],
    observation: Observation,
    task: str,
    *,
    config: StepScalingConfig = StepScalingConfig(),
    verifier: BaseActionVerifier | None = None,
) -> StepScalingResult:
    """Sample with (temperature, seed), then select one action.

    A well-formed action wins by consensus once its votes divided by the full
    sample budget strictly exceed the threshold. Otherwise score each unique
    action once and use votes and then sampling order to break ties.
    """
    started = perf_counter()
    telemetry = ScalingTelemetry()
    candidates: dict[str, Action] = {}
    votes: Counter[str] = Counter()
    for i in range(config.n):
        temperature = config.temperatures[i % len(config.temperatures)]
        candidate = sample(temperature, config.seed + i)
        telemetry.candidates_generated += 1
        telemetry.add_usage(candidate.usage)
        key = canonicalize_action(candidate.action)
        candidates.setdefault(key, candidate.action)
        votes[key] += 1
        if (
            votes[key] / config.n > config.consensus_threshold
            and _action_quality(candidate.action) > 0
        ):
            telemetry.early_stopped = telemetry.candidates_generated < config.n
            winner = candidates[key]
            break
    else:
        verifier = verifier if verifier is not None else HeuristicActionVerifier()
        ranks = {}
        for key, action in candidates.items():
            result = verifier.score(action, observation, task)
            telemetry.verifier_calls += 1
            telemetry.add_usage(result.usage)
            ranks[key] = (result.score, votes[key])
        winner = candidates[max(ranks, key=ranks.__getitem__)]

    telemetry.unique_candidates = len(candidates)
    telemetry.latency_s = perf_counter() - started
    return StepScalingResult(winner, telemetry)


class BestOfNAgent(GenericAgentV2):
    """Run non-streaming samples and add only the selected action to agent history."""

    def __init__(
        self,
        llm: LLMClient,
        config: AgentConfig | None = None,
        *,
        scaling: StepScalingConfig = StepScalingConfig(),
        verifier: BaseActionVerifier | None = None,
        max_history_images: int | None = None,
        image_selection_mode: str = "recent",
        history_text_window: int | None = None,
    ):
        super().__init__(
            llm,
            config,
            max_history_images=max_history_images,
            image_selection_mode=image_selection_mode,
            history_text_window=history_text_window,
        )
        self.scaling = scaling
        self.verifier = verifier
        self.step_telemetry: list[ScalingTelemetry] = []

    @property
    def name(self) -> str:
        return "BestOfNAgent"

    def reset(self, task: str) -> None:
        super().reset(task)
        self.step_telemetry.clear()

    def act(self, obs: Observation) -> Action:
        messages = self.build_messages(obs)

        def sample(temperature: float, seed: int) -> ActionSample:
            response = self.llm.chat(
                messages=messages,
                args={
                    **self.config.model_args,
                    "temperature": temperature,
                    "seed": seed,
                    "stream": False,
                },
            )
            try:
                action = self.parse_response(response.content)
            except (TypeError, ValueError, OverflowError):
                action = Action.abort("invalid_action", raw_response=response.content)
            return ActionSample(action, TokenUsage.from_chat(response))

        result = best_of_n_step(
            sample, obs, self.task, config=self.scaling, verifier=self.verifier
        )
        self._record_step(obs, result.action, result.action.raw_response, messages)
        self.step_telemetry.append(result.telemetry)
        return result.action


@dataclass(frozen=True)
class TrajectoryCandidate:
    result: EpisodeResult
    usage: TokenUsage = TokenUsage()
    vote_key: str | None = None

    def outcome_key(self) -> tuple[str, str]:
        """Group equivalent outcomes using a caller key, answer, or action trace."""
        if self.vote_key is not None:
            return ("custom", self.vote_key)
        execution = self.result.execution
        if execution.agent_answer and (answer := execution.agent_answer.strip()):
            return ("answer", answer)
        actions = [
            _action_payload(Action(ActionType(step["action_type"]), step["data"]))
            for step in execution.trace
        ]
        trace = {"actions": actions, "stop_reason": execution.stop_reason}
        return ("trace", json.dumps(trace, sort_keys=True, ensure_ascii=False))


class BaseTrajectoryVerifier(ABC):
    @abstractmethod
    def score(self, trajectory: TrajectoryCandidate) -> VerificationResult:
        """Score an individual rollout. Higher scores are preferred."""


@dataclass(frozen=True)
class TrajectoryScalingResult:
    selected: TrajectoryCandidate
    telemetry: ScalingTelemetry


def rerank_trajectories(
    candidates: Sequence[TrajectoryCandidate],
    *,
    verifier: BaseTrajectoryVerifier | None = None,
) -> TrajectoryScalingResult:
    """Prefer the highest verifier score, with majority votes as a tiebreaker.

    Without a verifier, select the first rollout in the largest outcome group.
    Unique candidates counts outcome groups. Usage includes all trials and
    verifier calls. Latency measures this reranking call.
    """
    if not candidates:
        raise ValueError("at least one trajectory candidate is required")
    started = perf_counter()
    keys = [candidate.outcome_key() for candidate in candidates]
    votes = Counter(keys)
    telemetry = ScalingTelemetry(
        candidates_generated=len(candidates), unique_candidates=len(votes)
    )
    ranks = []
    for candidate, key in zip(candidates, keys):
        telemetry.add_usage(candidate.usage)
        score = 0.0
        if verifier is not None:
            result = verifier.score(candidate)
            score = result.score
            telemetry.verifier_calls += 1
            telemetry.add_usage(result.usage)
        ranks.append((score, votes[key]))

    selected = candidates[max(range(len(ranks)), key=ranks.__getitem__)]
    telemetry.latency_s = perf_counter() - started
    return TrajectoryScalingResult(selected, telemetry)


async def best_of_n_trajectory(
    rollout: Callable[[int], Awaitable[TrajectoryCandidate]],
    *,
    k: int,
    verifier: BaseTrajectoryVerifier | None = None,
) -> TrajectoryScalingResult:
    """Run k independent trials with zero-based trial IDs, then rerank them."""
    if k < 1:
        raise ValueError("k must be positive")
    started = perf_counter()
    candidates = [await rollout(trial_id) for trial_id in range(k)]
    result = await asyncio.to_thread(rerank_trajectories, candidates, verifier=verifier)
    result.telemetry.latency_s = perf_counter() - started
    return result

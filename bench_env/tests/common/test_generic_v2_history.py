from unittest.mock import Mock

import pytest

from bench_env.agent.generic_v2 import GenericAgentV2
from bench_env.env.base import Observation
from bench_env.llm import ChatResult


@pytest.fixture(autouse=True)
def clear_history_settings(monkeypatch):
    monkeypatch.delenv("MOBILEGYM_MAX_HISTORY_IMAGES", raising=False)
    monkeypatch.delenv("MOBILEGYM_HISTORY_TEXT_WINDOW", raising=False)


def screen(label, step):
    return Observation(
        screenshot_bytes=label.encode(), step_idx=step, route={"path": "/private"}
    )


def make_agent(**options):
    llm = Mock()
    llm.chat.side_effect = [
        ChatResult(
            f'<think>Detailed reasoning {i}</think><answer>{{"action":"BACK"}}</answer>'
        )
        for i in range(1, 12)
    ]
    agent = GenericAgentV2(llm, **options)
    agent.reset("Open settings")
    return agent


def images(messages):
    return [
        part["image_url"]["url"]
        for message in messages
        if isinstance(message["content"], list)
        for part in message["content"]
        if part["type"] == "image_url"
    ]


def test_default_keeps_exact_text_history_and_current_image():
    agent = make_agent()
    observations = [screen(label, step) for step, label in enumerate("abcd", 1)]
    for obs in observations[:3]:
        agent.act(obs)

    messages = agent.build_messages(observations[3])

    assert agent.max_history_images == 1
    assert messages == [
        {"role": "system", "content": agent.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [{"type": "text", "text": "[任务]\nOpen settings"}],
        },
        {"role": "assistant", "content": agent.history[0].llm_response},
        {"role": "user", "content": [{"type": "text", "text": "[Step 2]"}]},
        {"role": "assistant", "content": agent.history[1].llm_response},
        {"role": "user", "content": [{"type": "text", "text": "[Step 3]"}]},
        {"role": "assistant", "content": agent.history[2].llm_response},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": observations[3].image_data_url},
                },
                {"type": "text", "text": "[Step 4]"},
            ],
        },
    ]
    assert [record.observation for record in agent.history] == [
        None,
        *observations[1:3],
    ]


def test_recent_frames_stay_in_their_original_turns():
    agent = make_agent(max_history_images=3)
    observations = [screen(label, step) for step, label in enumerate("abcde", 1)]
    for obs in observations:
        agent.act(obs)

    messages = agent.llm.chat.call_args.kwargs["messages"]
    assert images(messages) == [obs.image_data_url for obs in observations[-3:]]
    assert [messages[i]["content"][-1]["text"] for i in (5, 7, 9)] == [
        "[Step 3]",
        "[Step 4]",
        "[Step 5]",
    ]
    assert [record.observation for record in agent.history] == [
        None,
        None,
        *observations[-3:],
    ]
    assert agent.history[2].llm_prompt == []


def test_keyframes_survive_unchanged_runs_and_latest_frame_replaces_duplicate():
    agent = make_agent(max_history_images=3, image_selection_mode="keyframe")
    observations = [screen(label, step) for step, label in enumerate("aabbbccc", 1)]
    for obs in observations:
        agent.act(obs)

    messages = agent.llm.chat.call_args.kwargs["messages"]
    assert images(messages) == [
        obs.image_data_url
        for obs in (observations[0], observations[2], observations[-1])
    ]
    assert agent.history[0].observation is observations[0]
    assert observations[2] in [record.observation for record in agent.history]
    assert messages[5]["content"][-1]["text"] == "[Step 3]"

    for label in "def":
        agent.act(screen(label, len(agent.history) + 1))
    assert agent.history[0].observation is None
    assert agent.history[2].observation is None

    agent.reset_history()
    assert images(agent.build_messages(observations[0])) == [
        observations[0].image_data_url
    ]
    agent.reset("Next task")
    assert (
        agent.build_messages(observations[0])[-1]["content"][-1]["text"]
        == "[任务]\nNext task"
    )


def test_old_turns_are_compacted_with_visual_outcomes_and_selected_images():
    agent = make_agent(max_history_images=3, history_text_window=1)
    observations = [screen(label, step) for step, label in enumerate("aabc", 1)]
    for obs in observations:
        agent.act(obs)

    messages = agent.llm.chat.call_args.kwargs["messages"]
    assert messages[1]["content"][0]["text"].splitlines() == [
        "[任务]",
        "Open settings",
        "[Trajectory summary]",
        "[Step 1] BACK {} -> screen unchanged",
        "[Step 2] BACK {} -> screen changed",
    ]
    assert messages[1]["content"][1]["text"] == "[Step 2]"
    assert [m["content"] for m in messages if m["role"] == "assistant"] == [
        agent.history[2].llm_response
    ]
    assert images(messages) == [obs.image_data_url for obs in observations[-3:]]
    assert "/private" not in str(messages)


def test_env_defaults_and_explicit_overrides(monkeypatch):
    monkeypatch.setenv("MOBILEGYM_MAX_HISTORY_IMAGES", "3")
    monkeypatch.setenv("MOBILEGYM_HISTORY_TEXT_WINDOW", "0")
    agent = make_agent()
    agent.act(screen("a", 1))
    messages = agent.build_messages(screen("b", 2))
    assert len(images(messages)) == 2
    assert "[Step 1] BACK {} -> screen changed" in messages[1]["content"][0]["text"]
    assert all(message["role"] != "assistant" for message in messages)

    explicit = make_agent(max_history_images=1, history_text_window=2)
    assert (explicit.max_history_images, explicit.history_text_window) == (1, 2)


@pytest.mark.parametrize(
    "options",
    [
        {"max_history_images": 0},
        {"image_selection_mode": "unknown"},
        {"history_text_window": -1},
    ],
)
def test_invalid_history_settings_raise(options):
    with pytest.raises(ValueError):
        make_agent(**options)

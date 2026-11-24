import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bench_env.env.pool import EnvPool


@pytest.mark.asyncio
@pytest.mark.parametrize("isolation", ["pages", "contexts", "browsers"])
async def test_initialization_waits_for_each_wave(monkeypatch, isolation):
    events = []

    class FakeEnv:
        get_launch_args = staticmethod(lambda *args, **kwargs: {})
        get_context_args = staticmethod(lambda *args, **kwargs: {})
        setup_context_routes = AsyncMock()

        def __init__(self, worker_id, **kwargs):
            self.worker_id = worker_id

        async def start(self):
            events.append(("start", self.worker_id))
            await asyncio.sleep(0)

        async def wait_ready(self):
            events.append(("ready", self.worker_id))
            await asyncio.sleep(0)

        async def get_observation(self):
            await asyncio.sleep(0)
            events.append(("screenshot", self.worker_id))

        async def close(self):
            pass

    context = SimpleNamespace(close=AsyncMock())
    browser = SimpleNamespace(new_context=AsyncMock(return_value=context), close=AsyncMock())
    playwright = SimpleNamespace(chromium=SimpleNamespace(launch=AsyncMock(return_value=browser)), stop=AsyncMock())
    monkeypatch.setattr("playwright.async_api.async_playwright", lambda: SimpleNamespace(start=AsyncMock(return_value=playwright)))
    monkeypatch.setattr("bench_env.env.mobile_gym.MobileGymEnv", FakeEnv)
    monkeypatch.setenv("MOBILE_GYM_POOL_BATCH_SLEEP_S", "0")

    async with EnvPool("unused", n=5, num_browsers=2, isolation=isolation, init_batch_size=2, warmup=True) as pool:
        assert len(pool) == 5
        assert [env.worker_id for env in pool] == list(range(5))
        for worker_id in range(5):
            assert events.index(("start", worker_id)) < events.index(("ready", worker_id))
            assert events.index(("ready", worker_id)) < events.index(("screenshot", worker_id))
        assert events.index(("start", 2)) > max(events.index(("screenshot", worker_id)) for worker_id in (0, 1))
        assert events.index(("start", 4)) > max(events.index(("screenshot", worker_id)) for worker_id in (2, 3))


@pytest.mark.asyncio
async def test_failed_setup_cleans_up_and_raises():
    pool = EnvPool("unused", warmup=True)
    pool._setup = AsyncMock(side_effect=RuntimeError("screenshot timeout"))
    pool._cleanup = AsyncMock()
    with pytest.raises(RuntimeError, match="screenshot timeout"):
        await pool.__aenter__()
    pool._cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_warmup_failure_is_not_ignored():
    env = SimpleNamespace(start=AsyncMock(), wait_ready=AsyncMock(), get_observation=AsyncMock(side_effect=RuntimeError("screenshot timeout")))
    pool = EnvPool("unused", warmup=True)
    with pytest.raises(RuntimeError, match="screenshot timeout"):
        await pool._start_env(env)


@pytest.mark.asyncio
async def test_default_pool_does_not_enable_warmup():
    env = SimpleNamespace(start=AsyncMock(), wait_ready=AsyncMock(), get_observation=AsyncMock())
    await EnvPool("unused")._start_env(env)
    env.start.assert_awaited_once()
    env.wait_ready.assert_not_awaited()
    env.get_observation.assert_not_awaited()


def test_invalid_batch_size():
    with pytest.raises(ValueError, match="init_batch_size must be positive"):
        EnvPool("unused", init_batch_size=0)

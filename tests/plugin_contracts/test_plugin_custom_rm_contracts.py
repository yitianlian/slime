from __future__ import annotations

import importlib
import inspect
import os
import sys
import types
from argparse import ArgumentParser
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if "ray" not in sys.modules:
    ray_mod = types.ModuleType("ray")
    ray_mod._private = types.SimpleNamespace(services=types.SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1"))
    sys.modules["ray"] = ray_mod

NUM_GPUS = 0
ENV_PREFIX = "SLIME_CONTRACT_"
REFERENCE_SINGLE_RM_PATH = "plugin_contracts.test_plugin_custom_rm_contracts.reference_single_rm"
REFERENCE_BATCHED_RM_PATH = "plugin_contracts.test_plugin_custom_rm_contracts.reference_batched_rm"

from slime.rollout.rm_hub import async_rm, batched_async_rm
from slime.utils.types import Sample


def contract_env_name(key: str) -> str:
    return f"{ENV_PREFIX}{key}"


def get_contract_path(key: str, default: str) -> str:
    return os.environ.get(contract_env_name(key), default)


def run_contract_test_file() -> None:
    parser = ArgumentParser()
    parser.add_argument("--custom-rm-path", default=None)
    args, remaining = parser.parse_known_args()
    if args.custom_rm_path:
        os.environ[contract_env_name("CUSTOM_RM_PATH")] = args.custom_rm_path
    raise SystemExit(pytest.main([__file__, *remaining]))


def make_args(**overrides):
    class Args:
        custom_rm_path = None
        group_rm = False
        rm_type = None

    args = Args()
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def make_sample(index: int) -> Sample:
    return Sample(index=index, prompt=f"prompt-{index}", response=f"response-{index}", label="label", metadata={})


async def reference_single_rm(args, sample: Sample, **kwargs):
    return float(sample.index or 0) + 0.1


async def reference_batched_rm(args, samples: list[Sample], **kwargs):
    return [float(sample.index or 0) + 0.2 for sample in samples]


def assert_single_rm_signature_matches_expected(fn) -> None:
    params = inspect.signature(fn).parameters
    assert tuple(params)[:2] == ("args", "sample")


def assert_batched_rm_signature_matches_expected(fn) -> None:
    params = inspect.signature(fn).parameters
    assert tuple(params)[:2] == ("args", "samples")


@pytest.mark.asyncio
async def test_custom_rm_default_single_sample_branch_is_stable():
    reward = await async_rm(make_args(rm_type="random"), make_sample(4))
    assert isinstance(reward, (int, float))


@pytest.mark.asyncio
async def test_custom_rm_default_batched_branch_is_stable():
    rewards = await batched_async_rm(make_args(group_rm=True, rm_type="random"), [make_sample(1), make_sample(2)])
    assert isinstance(rewards, list)
    assert len(rewards) == 2


@pytest.mark.asyncio
async def test_custom_rm_path_aligns_with_single_sample_format():
    rm_fn = importlib.import_module("plugin_contracts.test_plugin_custom_rm_contracts").reference_single_rm
    assert_single_rm_signature_matches_expected(rm_fn)
    reward = await async_rm(make_args(custom_rm_path=get_contract_path("CUSTOM_RM_PATH", REFERENCE_SINGLE_RM_PATH)), make_sample(3))
    assert isinstance(reward, (int, float))


@pytest.mark.asyncio
async def test_custom_rm_path_aligns_with_batched_sample_format():
    rm_fn = importlib.import_module("plugin_contracts.test_plugin_custom_rm_contracts").reference_batched_rm
    assert_batched_rm_signature_matches_expected(rm_fn)
    rewards = await batched_async_rm(
        make_args(group_rm=True, custom_rm_path=get_contract_path("CUSTOM_RM_PATH", REFERENCE_BATCHED_RM_PATH)),
        [make_sample(0), make_sample(1)],
    )
    assert isinstance(rewards, list)
    assert len(rewards) == 2
    assert all(isinstance(value, (int, float)) for value in rewards)


if __name__ == "__main__":
    run_contract_test_file()

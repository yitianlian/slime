from __future__ import annotations

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
REFERENCE_CUSTOM_REWARD_POST_PROCESS_PATH = "plugin_contracts.test_plugin_custom_reward_post_process_contracts.reference_reward_post_process"

from slime.utils.types import Sample


def contract_env_name(key: str) -> str:
    return f"{ENV_PREFIX}{key}"


def get_contract_path(key: str, default: str) -> str:
    return os.environ.get(contract_env_name(key), default)


def run_contract_test_file() -> None:
    parser = ArgumentParser()
    parser.add_argument("--custom-reward-post-process-path", default=None)
    args, remaining = parser.parse_known_args()
    if args.custom_reward_post_process_path:
        os.environ[contract_env_name("CUSTOM_REWARD_POST_PROCESS_PATH")] = args.custom_reward_post_process_path
    raise SystemExit(pytest.main([__file__, *remaining]))


def make_sample(index: int, reward: float) -> Sample:
    return Sample(index=index, reward=reward)


def reference_reward_post_process(args, samples):
    raw_rewards = [sample.reward for sample in samples]
    rewards = [reward + 1.0 for reward in raw_rewards]
    return raw_rewards, rewards


def assert_runtime_callsite_is_stable() -> None:
    source = Path("slime/ray/rollout.py").read_text()
    assert "self.custom_reward_post_process_func(self.args, samples)" in source


def assert_custom_reward_post_process_output_matches_expected(fn) -> None:
    samples = [make_sample(0, 0.5), make_sample(1, 1.5)]
    raw_rewards, rewards = fn(type("Args", (), {})(), samples)
    assert isinstance(raw_rewards, list)
    assert isinstance(rewards, list)
    assert len(raw_rewards) == len(samples)
    assert len(rewards) == len(samples)


def test_custom_reward_post_process_callsite_is_stable():
    assert_runtime_callsite_is_stable()


def test_custom_reward_post_process_path_aligns_with_expected_format():
    from slime.utils.misc import load_function

    fn = load_function(
        get_contract_path("CUSTOM_REWARD_POST_PROCESS_PATH", REFERENCE_CUSTOM_REWARD_POST_PROCESS_PATH)
    )
    assert_custom_reward_post_process_output_matches_expected(fn)


if __name__ == "__main__":
    run_contract_test_file()

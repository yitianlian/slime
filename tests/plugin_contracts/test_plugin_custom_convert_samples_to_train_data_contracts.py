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
REFERENCE_CUSTOM_CONVERT_SAMPLES_TO_TRAIN_DATA_PATH = "plugin_contracts.test_plugin_custom_convert_samples_to_train_data_contracts.reference_convert_samples_to_train_data"

from slime.utils.types import Sample


def contract_env_name(key: str) -> str:
    return f"{ENV_PREFIX}{key}"


def get_contract_path(key: str, default: str) -> str:
    return os.environ.get(contract_env_name(key), default)


def run_contract_test_file() -> None:
    parser = ArgumentParser()
    parser.add_argument("--custom-convert-samples-to-train-data-path", default=None)
    args, remaining = parser.parse_known_args()
    if args.custom_convert_samples_to_train_data_path:
        os.environ[contract_env_name("CUSTOM_CONVERT_SAMPLES_TO_TRAIN_DATA_PATH")] = (
            args.custom_convert_samples_to_train_data_path
        )
    raise SystemExit(pytest.main([__file__, *remaining]))


def make_sample(index: int, reward: float) -> Sample:
    return Sample(
        index=index,
        tokens=[index, index + 1],
        response_length=2,
        reward=reward,
        status=Sample.Status.COMPLETED,
        loss_mask=[1, 1],
    )


def reference_convert_samples_to_train_data(args, samples):
    return {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        "rewards": [sample.reward for sample in samples],
        "raw_reward": [sample.reward for sample in samples],
        "truncated": [0 for _ in samples],
        "sample_indices": [sample.index for sample in samples],
        "loss_masks": [sample.loss_mask for sample in samples],
    }


def assert_runtime_callsite_is_stable() -> None:
    source = Path("slime/ray/rollout.py").read_text()
    assert "self.custom_convert_samples_to_train_data_func(self.args, samples)" in source


def assert_custom_convert_output_matches_expected(fn) -> None:
    samples = [make_sample(0, 0.5), make_sample(1, 1.5)]
    train_data = fn(type("Args", (), {})(), samples)
    required_keys = {"tokens", "response_lengths", "rewards", "raw_reward", "truncated", "sample_indices", "loss_masks"}
    assert isinstance(train_data, dict)
    assert required_keys <= set(train_data)
    assert all(len(train_data[key]) == len(samples) for key in required_keys)


def test_custom_convert_samples_to_train_data_callsite_is_stable():
    assert_runtime_callsite_is_stable()


def test_custom_convert_samples_to_train_data_path_aligns_with_expected_format():
    from slime.utils.misc import load_function

    fn = load_function(
        get_contract_path(
            "CUSTOM_CONVERT_SAMPLES_TO_TRAIN_DATA_PATH",
            REFERENCE_CUSTOM_CONVERT_SAMPLES_TO_TRAIN_DATA_PATH,
        )
    )
    assert_custom_convert_output_matches_expected(fn)


if __name__ == "__main__":
    run_contract_test_file()

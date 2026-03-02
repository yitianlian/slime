from __future__ import annotations

import inspect
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

NUM_GPUS = 0
ENV_PREFIX = "SLIME_CONTRACT_"
DEFAULT_DYNAMIC_FILTER_PATH = "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"

from slime.rollout.filter_hub.base_types import DynamicFilterOutput, call_dynamic_filter
from slime.utils.misc import load_function
from slime.utils.types import Sample


def contract_env_name(key: str) -> str:
    return f"{ENV_PREFIX}{key}"


def get_contract_path(key: str, default: str) -> str:
    return os.environ.get(contract_env_name(key), default)


def run_contract_test_file() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dynamic-sampling-filter-path", default=None)
    args, remaining = parser.parse_known_args()
    if args.dynamic_sampling_filter_path:
        os.environ[contract_env_name("DYNAMIC_SAMPLING_FILTER_PATH")] = args.dynamic_sampling_filter_path
    raise SystemExit(pytest.main([__file__, *remaining]))


def make_sample(index: int, drop: bool = False) -> Sample:
    reward = 0.0 if drop else float(index + 1)
    return Sample(index=index, reward=reward, metadata={"drop": drop})


def make_args(**overrides):
    args = type("Args", (), {"reward_key": None})()
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def reference_dynamic_filter(args, samples: list[Sample], **kwargs):
    keep = not any(sample.metadata.get("drop") for sample in samples)
    return DynamicFilterOutput(keep=keep, reason=None if keep else "drop-flag")


def assert_dynamic_filter_signature_stable(fn) -> None:
    sig = inspect.signature(fn)
    params = sig.parameters
    assert tuple(params)[:2] == ("args", "samples")


def assert_dynamic_filter_output_aligned(fn) -> None:
    out = call_dynamic_filter(fn, make_args(), [make_sample(0), make_sample(1)])
    assert isinstance(out, DynamicFilterOutput)
    assert isinstance(bool(out.keep), bool)


def test_dynamic_filter_interface_is_stable():
    assert_dynamic_filter_signature_stable(reference_dynamic_filter)
    assert_dynamic_filter_output_aligned(reference_dynamic_filter)


def test_dynamic_filter_path_aligns_with_expected_format():
    fn = load_function(get_contract_path("DYNAMIC_SAMPLING_FILTER_PATH", DEFAULT_DYNAMIC_FILTER_PATH))
    assert_dynamic_filter_signature_stable(fn)
    assert_dynamic_filter_output_aligned(fn)


if __name__ == "__main__":
    run_contract_test_file()

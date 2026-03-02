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
REFERENCE_ROLLOUT_SAMPLE_FILTER_PATH = (
    "plugin_contracts.test_plugin_rollout_sample_filter_contracts.reference_rollout_sample_filter"
)

from slime.utils.misc import load_function
from slime.utils.types import Sample


def contract_env_name(key: str) -> str:
    return f"{ENV_PREFIX}{key}"


def get_contract_path(key: str, default: str) -> str:
    return os.environ.get(contract_env_name(key), default)


def run_contract_test_file() -> None:
    parser = ArgumentParser()
    parser.add_argument("--rollout-sample-filter-path", default=None)
    args, remaining = parser.parse_known_args()
    if args.rollout_sample_filter_path:
        os.environ[contract_env_name("ROLLOUT_SAMPLE_FILTER_PATH")] = args.rollout_sample_filter_path
    raise SystemExit(pytest.main([__file__, *remaining]))


def make_groups() -> list[list[Sample]]:
    return [[Sample(index=0), Sample(index=1)], [Sample(index=2), Sample(index=3)]]


def reference_rollout_sample_filter(args, groups: list[list[Sample]]) -> None:
    for group in groups:
        if group:
            group[-1].remove_sample = True


def assert_rollout_sample_filter_signature_stable(fn) -> None:
    sig = inspect.signature(fn)
    assert tuple(sig.parameters)[:2] == ("args", "groups")


def assert_rollout_sample_filter_output_aligned(fn) -> None:
    groups = make_groups()
    fn(object(), groups)
    assert all(isinstance(group, list) for group in groups)
    assert any(sample.remove_sample for group in groups for sample in group)


def test_rollout_sample_filter_interface_is_stable():
    assert_rollout_sample_filter_signature_stable(reference_rollout_sample_filter)
    assert_rollout_sample_filter_output_aligned(reference_rollout_sample_filter)


def test_rollout_sample_filter_path_aligns_with_expected_format():
    fn = load_function(get_contract_path("ROLLOUT_SAMPLE_FILTER_PATH", REFERENCE_ROLLOUT_SAMPLE_FILTER_PATH))
    assert_rollout_sample_filter_signature_stable(fn)
    assert_rollout_sample_filter_output_aligned(fn)


if __name__ == "__main__":
    run_contract_test_file()

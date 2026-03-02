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
REFERENCE_ROLLOUT_ALL_SAMPLES_PROCESS_PATH = "plugin_contracts.test_plugin_rollout_all_samples_process_contracts.reference_rollout_all_samples_process"

from slime.utils.misc import load_function
from slime.utils.types import Sample


def contract_env_name(key: str) -> str:
    return f"{ENV_PREFIX}{key}"


def get_contract_path(key: str, default: str) -> str:
    return os.environ.get(contract_env_name(key), default)


def run_contract_test_file() -> None:
    parser = ArgumentParser()
    parser.add_argument("--rollout-all-samples-process-path", default=None)
    args, remaining = parser.parse_known_args()
    if args.rollout_all_samples_process_path:
        os.environ[contract_env_name("ROLLOUT_ALL_SAMPLES_PROCESS_PATH")] = args.rollout_all_samples_process_path
    raise SystemExit(pytest.main([__file__, *remaining]))


def make_groups() -> list[list[Sample]]:
    return [[Sample(index=0), Sample(index=1)], [Sample(index=2), Sample(index=3)]]


class DummyDataSource:
    pass


def reference_rollout_all_samples_process(args, all_groups: list[list[Sample]], data_source) -> None:
    args.processed_group_count = len(all_groups)


def assert_rollout_all_samples_process_signature_stable(fn) -> None:
    sig = inspect.signature(fn)
    assert tuple(sig.parameters)[:3] == ("args", "all_groups", "data_source")


def assert_rollout_all_samples_process_output_aligned(fn) -> None:
    args = type("Args", (), {})()
    fn(args, make_groups(), DummyDataSource())
    assert getattr(args, "processed_group_count", 0) == 2


def test_rollout_all_samples_process_interface_is_stable():
    assert_rollout_all_samples_process_signature_stable(reference_rollout_all_samples_process)
    assert_rollout_all_samples_process_output_aligned(reference_rollout_all_samples_process)


def test_rollout_all_samples_process_path_aligns_with_expected_format():
    fn = load_function(
        get_contract_path("ROLLOUT_ALL_SAMPLES_PROCESS_PATH", REFERENCE_ROLLOUT_ALL_SAMPLES_PROCESS_PATH)
    )
    assert_rollout_all_samples_process_signature_stable(fn)
    assert_rollout_all_samples_process_output_aligned(fn)


if __name__ == "__main__":
    run_contract_test_file()

from __future__ import annotations

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
REFERENCE_CUSTOM_EVAL_ROLLOUT_LOG_PATH = "plugin_contracts.test_plugin_custom_eval_rollout_log_contracts.reference_custom_eval_rollout_log"

from slime.utils.types import Sample


def contract_env_name(key: str) -> str:
    return f"{ENV_PREFIX}{key}"


def get_contract_path(key: str, default: str) -> str:
    return os.environ.get(contract_env_name(key), default)


def run_contract_test_file() -> None:
    parser = ArgumentParser()
    parser.add_argument("--custom-eval-rollout-log-function-path", default=None)
    args, remaining = parser.parse_known_args()
    if args.custom_eval_rollout_log_function_path:
        os.environ[contract_env_name("CUSTOM_EVAL_ROLLOUT_LOG_FUNCTION_PATH")] = args.custom_eval_rollout_log_function_path
    raise SystemExit(pytest.main([__file__, *remaining]))


def reference_custom_eval_rollout_log(rollout_id, args, data, extra_metrics) -> bool:
    args.logged_eval_rollout_id = rollout_id
    args.logged_dataset_names = tuple(data)
    return True


def assert_runtime_callsite_is_stable() -> None:
    source = Path("slime/ray/rollout.py").read_text()
    assert "args.custom_eval_rollout_log_function_path" in source
    assert "custom_log_func(rollout_id, args, data, extra_metrics)" in source


def assert_custom_eval_rollout_log_signature_matches_expected(fn) -> None:
    params = tuple(inspect.signature(fn).parameters)
    assert params == ("rollout_id", "args", "data", "extra_metrics")


def assert_custom_eval_rollout_log_output_matches_expected(fn) -> None:
    args = type("Args", (), {})()
    sample = Sample(index=0, reward=1.0)
    data = {"eval_set": {"rewards": [1.0], "truncated": [False], "samples": [sample]}}
    should_skip_default = fn(4, args, data, {"acc": 1.0})
    assert isinstance(should_skip_default, bool)
    assert args.logged_eval_rollout_id == 4
    assert args.logged_dataset_names == ("eval_set",)


def test_custom_eval_rollout_log_callsite_is_stable():
    assert_runtime_callsite_is_stable()


def test_custom_eval_rollout_log_path_aligns_with_expected_format():
    from slime.utils.misc import load_function

    fn = load_function(
        get_contract_path("CUSTOM_EVAL_ROLLOUT_LOG_FUNCTION_PATH", REFERENCE_CUSTOM_EVAL_ROLLOUT_LOG_PATH)
    )
    assert_custom_eval_rollout_log_signature_matches_expected(fn)
    assert_custom_eval_rollout_log_output_matches_expected(fn)


if __name__ == "__main__":
    run_contract_test_file()

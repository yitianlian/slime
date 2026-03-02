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
REFERENCE_ROLLOUT_DATA_POSTPROCESS_PATH = (
    "plugin_contracts.test_plugin_rollout_data_postprocess_contracts.reference_rollout_data_postprocess"
)


def contract_env_name(key: str) -> str:
    return f"{ENV_PREFIX}{key}"


def get_contract_path(key: str, default: str) -> str:
    return os.environ.get(contract_env_name(key), default)


def run_contract_test_file() -> None:
    parser = ArgumentParser()
    parser.add_argument("--rollout-data-postprocess-path", default=None)
    args, remaining = parser.parse_known_args()
    if args.rollout_data_postprocess_path:
        os.environ[contract_env_name("ROLLOUT_DATA_POSTPROCESS_PATH")] = args.rollout_data_postprocess_path
    raise SystemExit(pytest.main([__file__, *remaining]))


def reference_rollout_data_postprocess(args) -> None:
    args.rollout_data_postprocess_called = True


def assert_runtime_callsite_is_stable() -> None:
    source = Path("slime/backends/megatron_utils/actor.py").read_text()
    assert "self.rollout_data_postprocess(self.args)" in source


def assert_rollout_data_postprocess_signature_matches_expected(fn) -> None:
    params = tuple(inspect.signature(fn).parameters)
    assert params == ("args",)


def assert_rollout_data_postprocess_output_matches_expected(fn) -> None:
    args = type("Args", (), {})()
    result = fn(args)
    assert result is None
    assert args.rollout_data_postprocess_called is True


def test_rollout_data_postprocess_callsite_is_stable():
    assert_runtime_callsite_is_stable()


def test_rollout_data_postprocess_path_aligns_with_expected_format():
    from slime.utils.misc import load_function

    fn = load_function(get_contract_path("ROLLOUT_DATA_POSTPROCESS_PATH", REFERENCE_ROLLOUT_DATA_POSTPROCESS_PATH))
    assert_rollout_data_postprocess_signature_matches_expected(fn)
    assert_rollout_data_postprocess_output_matches_expected(fn)


if __name__ == "__main__":
    run_contract_test_file()

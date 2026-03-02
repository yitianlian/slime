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
if "transformers" not in sys.modules:
    mod = types.ModuleType("transformers")
    mod.AutoTokenizer = type("AutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: object())})
    mod.AutoProcessor = type(
        "AutoProcessor",
        (),
        {"from_pretrained": staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))},
    )
    mod.PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (), {})
    mod.ProcessorMixin = type("ProcessorMixin", (), {})
    sys.modules["transformers"] = mod

NUM_GPUS = 0
ENV_PREFIX = "SLIME_CONTRACT_"
DEFAULT_BUFFER_FILTER_PATH = "slime.rollout.data_source.pop_first"

from slime.rollout.data_source import RolloutDataSourceWithBuffer
from slime.utils.misc import load_function
from slime.utils.types import Sample


def contract_env_name(key: str) -> str:
    return f"{ENV_PREFIX}{key}"


def get_contract_path(key: str, default: str) -> str:
    return os.environ.get(contract_env_name(key), default)


def run_contract_test_file() -> None:
    parser = ArgumentParser()
    parser.add_argument("--buffer-filter-path", default=None)
    args, remaining = parser.parse_known_args()
    if args.buffer_filter_path:
        os.environ[contract_env_name("BUFFER_FILTER_PATH")] = args.buffer_filter_path
    raise SystemExit(pytest.main([__file__, *remaining]))


def make_args(**overrides):
    class Args:
        rollout_global_dataset = False
        buffer_filter_path = None
        n_samples_per_prompt = 2

    args = Args()
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def make_group(start: int) -> list[Sample]:
    return [Sample(index=start), Sample(index=start + 1)]


def reference_buffer_filter(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    selected = list(reversed(buffer[-num_samples:]))
    del buffer[-num_samples:]
    return selected


def assert_buffer_filter_signature_stable(fn) -> None:
    sig = inspect.signature(fn)
    assert tuple(sig.parameters)[:4] == ("args", "rollout_id", "buffer", "num_samples")


def assert_buffer_filter_output_aligned(fn) -> None:
    args = make_args(buffer_filter_path="plugin_contracts.test_plugin_buffer_filter_contracts.reference_buffer_filter")
    data_source = RolloutDataSourceWithBuffer(args)
    data_source.add_samples([make_group(0), make_group(2)])
    selected = fn(args, None, data_source.buffer, 1)
    assert isinstance(selected, list)
    assert len(selected) <= 1
    assert all(isinstance(group, list) for group in selected)


def test_buffer_filter_interface_is_stable():
    assert_buffer_filter_signature_stable(reference_buffer_filter)
    assert_buffer_filter_output_aligned(reference_buffer_filter)


def test_buffer_filter_path_aligns_with_expected_format():
    fn = load_function(get_contract_path("BUFFER_FILTER_PATH", DEFAULT_BUFFER_FILTER_PATH))
    assert_buffer_filter_signature_stable(fn)
    assert_buffer_filter_output_aligned(fn)


if __name__ == "__main__":
    run_contract_test_file()

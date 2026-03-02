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
    mod.AutoProcessor = type("AutoProcessor", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))})
    mod.PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (), {})
    mod.ProcessorMixin = type("ProcessorMixin", (), {})
    sys.modules["transformers"] = mod

NUM_GPUS = 0
ENV_PREFIX = "SLIME_CONTRACT_"
DEFAULT_DATA_SOURCE_PATH = "slime.rollout.data_source.RolloutDataSourceWithBuffer"
REFERENCE_DATA_SOURCE_PATH = "plugin_contracts.test_plugin_data_source_contracts.ReferenceDataSource"

from slime.utils.misc import load_function
from slime.utils.types import Sample


def contract_env_name(key: str) -> str:
    return f"{ENV_PREFIX}{key}"


def get_contract_path(key: str, default: str) -> str:
    return os.environ.get(contract_env_name(key), default)


def run_contract_test_file() -> None:
    parser = ArgumentParser()
    parser.add_argument("--data-source-path", default=None)
    args, remaining = parser.parse_known_args()
    if args.data_source_path:
        os.environ[contract_env_name("DATA_SOURCE_PATH")] = args.data_source_path
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


class ReferenceDataSource:
    def __init__(self, args):
        self.args = args
        self._groups = [
            [Sample(index=0), Sample(index=1)],
            [Sample(index=2), Sample(index=3)],
        ]

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        selected = self._groups[:num_samples]
        self._groups = self._groups[num_samples:]
        return selected

    def add_samples(self, samples: list[list[Sample]]):
        self._groups.extend(samples)

    def save(self, rollout_id):
        self.last_saved_rollout_id = rollout_id

    def load(self, rollout_id=None):
        self.last_loaded_rollout_id = rollout_id

    def __len__(self) -> int:
        return len(self._groups)


def assert_data_source_methods_are_stable(cls) -> None:
    assert tuple(inspect.signature(cls.__init__).parameters)[:2] == ("self", "args")
    for name in ("get_samples", "add_samples", "save", "load", "__len__"):
        assert hasattr(cls, name)


def assert_data_source_output_matches_expected(data_source) -> None:
    groups = data_source.get_samples(1)
    assert isinstance(groups, list)
    assert len(groups) == 1
    assert all(isinstance(group, list) for group in groups)
    assert all(isinstance(sample, Sample) for group in groups for sample in group)


def test_data_source_default_class_is_stable():
    data_source_cls = load_function(DEFAULT_DATA_SOURCE_PATH)
    assert_data_source_methods_are_stable(data_source_cls)
    data_source = data_source_cls(make_args())
    assert_data_source_output_matches_expected(data_source)


def test_data_source_path_aligns_with_expected_format():
    data_source_cls = load_function(get_contract_path("DATA_SOURCE_PATH", REFERENCE_DATA_SOURCE_PATH))
    assert_data_source_methods_are_stable(data_source_cls)
    data_source = data_source_cls(make_args())
    assert_data_source_output_matches_expected(data_source)


if __name__ == "__main__":
    run_contract_test_file()

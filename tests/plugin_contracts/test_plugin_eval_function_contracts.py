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
if "sglang_router" not in sys.modules:
    mod = types.ModuleType("sglang_router")
    mod.__version__ = "0.2.3"
    sys.modules["sglang_router"] = mod
if "transformers" not in sys.modules:
    mod = types.ModuleType("transformers")
    mod.AutoTokenizer = type("AutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: object())})
    mod.AutoProcessor = type("AutoProcessor", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))})
    mod.PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (), {})
    mod.ProcessorMixin = type("ProcessorMixin", (), {})
    sys.modules["transformers"] = mod

NUM_GPUS = 0
ENV_PREFIX = "SLIME_CONTRACT_"
DEFAULT_EVAL_FUNCTION_PATH = "slime.rollout.sglang_rollout.generate_rollout"
REFERENCE_EVAL_FUNCTION_PATH = "plugin_contracts.test_plugin_eval_function_contracts.valid_eval_function"

from slime.rollout.base_types import RolloutFnEvalOutput, call_rollout_fn
from slime.rollout.sglang_rollout import generate_rollout as default_generate_rollout
from slime.utils.misc import load_function
from slime.utils.types import Sample


def contract_env_name(key: str) -> str:
    return f"{ENV_PREFIX}{key}"


def get_contract_path(key: str, default: str) -> str:
    return os.environ.get(contract_env_name(key), default)


def run_contract_test_file() -> None:
    parser = ArgumentParser()
    parser.add_argument("--eval-function-path", default=None)
    args, remaining = parser.parse_known_args()
    if args.eval_function_path:
        os.environ[contract_env_name("EVAL_FUNCTION_PATH")] = args.eval_function_path
    raise SystemExit(pytest.main([__file__, *remaining]))


class ContractDataSource:
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        groups = []
        for index in range(num_samples):
            groups.append([Sample(index=index, prompt=f"prompt-{index}")])
        return groups


def make_eval_sample(index: int, reward: float = 0.5) -> Sample:
    return Sample(
        index=index,
        prompt=f"prompt-{index}",
        response=f"response-{index}",
        tokens=[100 + index, 200 + index],
        response_length=2,
        reward=reward,
        status=Sample.Status.COMPLETED,
        metadata={},
    )


def valid_eval_function(args, rollout_id, data_source, evaluation=False):
    assert evaluation is True
    sample = make_eval_sample(rollout_id)
    return RolloutFnEvalOutput(
        data={"eval_contract": {"rewards": [sample.reward], "truncated": [False], "samples": [sample]}},
        metrics={"source": "contract"},
    )


def assert_eval_function_signature_matches_default(fn) -> None:
    default_sig = inspect.signature(default_generate_rollout)
    candidate_sig = inspect.signature(fn)
    assert tuple(candidate_sig.parameters) == tuple(default_sig.parameters)
    for name, default_param in default_sig.parameters.items():
        candidate_param = candidate_sig.parameters[name]
        assert candidate_param.kind == default_param.kind
        assert candidate_param.default == default_param.default


def assert_eval_output_matches_expected(output: RolloutFnEvalOutput) -> None:
    assert isinstance(output, RolloutFnEvalOutput)
    assert output.data
    for dataset_data in output.data.values():
        assert set(dataset_data) >= {"rewards", "truncated", "samples"}
        assert len(dataset_data["rewards"]) == len(dataset_data["truncated"]) == len(dataset_data["samples"])
        for sample in dataset_data["samples"]:
            assert isinstance(sample, Sample)
            assert isinstance(sample.tokens, list)
            assert isinstance(sample.response, str)
            assert isinstance(sample.response_length, int)
            assert sample.reward is not None


def test_eval_function_signature_is_stable():
    default_sig = inspect.signature(default_generate_rollout)
    assert tuple(default_sig.parameters) == ("args", "rollout_id", "data_source", "evaluation")
    assert default_sig.parameters["evaluation"].default is False


def test_eval_function_path_aligns_with_expected_eval_output():
    eval_path = get_contract_path("EVAL_FUNCTION_PATH", DEFAULT_EVAL_FUNCTION_PATH)
    eval_fn = load_function(eval_path)
    assert_eval_function_signature_matches_default(eval_fn)

    if eval_path != DEFAULT_EVAL_FUNCTION_PATH:
        output = call_rollout_fn(eval_fn, None, 5, ContractDataSource(), evaluation=True)
        assert_eval_output_matches_expected(output)


def test_local_eval_function_aligns_with_expected_eval_output():
    eval_fn = load_function(REFERENCE_EVAL_FUNCTION_PATH)
    assert_eval_function_signature_matches_default(eval_fn)
    output = call_rollout_fn(eval_fn, None, 6, ContractDataSource(), evaluation=True)
    assert_eval_output_matches_expected(output)


if __name__ == "__main__":
    run_contract_test_file()

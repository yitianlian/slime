from __future__ import annotations

import inspect
import os
import sys
import types
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

NUM_GPUS = 0
ENV_PREFIX = "SLIME_CONTRACT_"
REFERENCE_CUSTOM_GENERATE_PATH = "plugin_contracts.test_plugin_generate_contracts.custom_generate"
REFERENCE_CUSTOM_GENERATE_WITH_EVAL_PATH = (
    "plugin_contracts.test_plugin_generate_contracts.custom_generate_with_evaluation"
)


def install_stubs() -> None:
    if "ray" not in sys.modules:
        ray_mod = types.ModuleType("ray")
        ray_mod._private = types.SimpleNamespace(
            services=types.SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1")
        )
        sys.modules["ray"] = ray_mod
    if "sglang_router" not in sys.modules:
        mod = types.ModuleType("sglang_router")
        mod.__version__ = "0.2.3"
        sys.modules["sglang_router"] = mod
    if "transformers" not in sys.modules:
        mod = types.ModuleType("transformers")
        mod.AutoTokenizer = type(
            "AutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: object())}
        )
        mod.AutoProcessor = type(
            "AutoProcessor",
            (),
            {"from_pretrained": staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))},
        )
        mod.PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (), {})
        mod.ProcessorMixin = type("ProcessorMixin", (), {})
        sys.modules["transformers"] = mod


install_stubs()

from slime.rollout.sglang_rollout import generate_and_rm
from slime.utils.misc import load_function
from slime.utils.types import Sample


def contract_env_name(key: str) -> str:
    return f"{ENV_PREFIX}{key}"


def get_contract_path(key: str, default: str) -> str:
    return os.environ.get(contract_env_name(key), default)


def run_contract_test_file() -> None:
    parser = ArgumentParser()
    parser.add_argument("--custom-generate-function-path", default=None)
    args, remaining = parser.parse_known_args()
    if args.custom_generate_function_path:
        os.environ[contract_env_name("CUSTOM_GENERATE_FUNCTION_PATH")] = args.custom_generate_function_path
    raise SystemExit(pytest.main([__file__, *remaining]))


def make_args(**overrides):
    class Args:
        partial_rollout = False
        mask_offpolicy_in_partial_rollout = False
        group_rm = False
        custom_generate_function_path = None
        sglang_enable_deterministic_inference = False
        rollout_seed = 7
        n_samples_per_prompt = 2

    args = Args()
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class FakeGenerateState:
    def __init__(self, args) -> None:
        self.args = args
        self.semaphore = types.SimpleNamespace(__aenter__=None)
        self.pendings = set()
        self.remaining_batch_size = 0
        self.aborted = False
        self.group_sampling_seeds = [args.rollout_seed + i for i in range(args.n_samples_per_prompt)]

    @contextmanager
    def dp_rank_context(self):
        yield 0


async def custom_generate(args, sample: Sample, sampling_params: dict):
    sample.tokens = [11, 12, 13]
    sample.response = "generated"
    sample.response_length = len(sample.tokens)
    sample.reward = 0.25
    sample.status = Sample.Status.COMPLETED
    return sample


async def custom_generate_with_evaluation(args, sample: Sample, sampling_params: dict, evaluation: bool = False):
    sample.tokens = [21, 22]
    sample.response = "eval-generated" if evaluation else "train-generated"
    sample.response_length = len(sample.tokens)
    sample.reward = 0.5 if evaluation else 0.75
    sample.status = Sample.Status.COMPLETED
    sample.metadata["evaluation"] = evaluation
    return sample


def assert_sample_contract(sample: Sample) -> None:
    assert isinstance(sample, Sample)
    assert isinstance(sample.tokens, list)
    assert isinstance(sample.response, str)
    assert isinstance(sample.response_length, int)
    assert sample.reward is not None


def assert_custom_generate_signature_matches_expected(fn) -> None:
    params = tuple(inspect.signature(fn).parameters)
    assert params[:3] == ("args", "sample", "sampling_params")


@pytest.mark.asyncio
async def test_generate_and_rm_default_generate_branch_is_stable(monkeypatch):
    from slime.rollout import sglang_rollout

    class DummySemaphore:
        async def __aenter__(self):
            return None

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class LocalState(FakeGenerateState):
        def __init__(self, args):
            super().__init__(args)
            self.semaphore = DummySemaphore()

    async def official_default_generate(args, sample: Sample, sampling_params: dict):
        sample.tokens = [31, 32]
        sample.response = "default-generate"
        sample.response_length = 2
        sample.reward = 1.0
        sample.status = Sample.Status.COMPLETED
        return sample

    monkeypatch.setattr(sglang_rollout, "GenerateState", LocalState)
    monkeypatch.setattr(sglang_rollout, "generate", official_default_generate)

    result = await generate_and_rm(
        make_args(custom_generate_function_path=None),
        Sample(index=0, prompt="prompt"),
        sampling_params={"temperature": 0.3},
        evaluation=False,
    )
    assert_sample_contract(result)
    assert result.response == "default-generate"


@pytest.mark.asyncio
async def test_generate_and_rm_prefers_per_sample_generate_function(monkeypatch):
    from slime.rollout import sglang_rollout

    class DummySemaphore:
        async def __aenter__(self):
            return None

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class LocalState(FakeGenerateState):
        def __init__(self, args):
            super().__init__(args)
            self.semaphore = DummySemaphore()

    monkeypatch.setattr(sglang_rollout, "GenerateState", LocalState)
    args = make_args(custom_generate_function_path=REFERENCE_CUSTOM_GENERATE_PATH)
    sample = Sample(index=0, prompt="prompt", generate_function_path=REFERENCE_CUSTOM_GENERATE_WITH_EVAL_PATH)
    result = await generate_and_rm(args, sample, sampling_params={"temperature": 0.3}, evaluation=True)
    assert_sample_contract(result)
    assert result.metadata["evaluation"] is True


@pytest.mark.asyncio
async def test_custom_generate_function_path_supports_user_override(monkeypatch):
    from slime.rollout import sglang_rollout

    class DummySemaphore:
        async def __aenter__(self):
            return None

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class LocalState(FakeGenerateState):
        def __init__(self, args):
            super().__init__(args)
            self.semaphore = DummySemaphore()

    monkeypatch.setattr(sglang_rollout, "GenerateState", LocalState)
    custom_generate_path = get_contract_path(
        "CUSTOM_GENERATE_FUNCTION_PATH",
        REFERENCE_CUSTOM_GENERATE_PATH,
    )
    assert_custom_generate_signature_matches_expected(load_function(custom_generate_path))
    result = await generate_and_rm(
        make_args(custom_generate_function_path=custom_generate_path),
        Sample(index=0, prompt="prompt"),
        sampling_params={"temperature": 0.3},
        evaluation=False,
    )
    assert_sample_contract(result)


if __name__ == "__main__":
    run_contract_test_file()

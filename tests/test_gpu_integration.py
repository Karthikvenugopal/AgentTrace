import os

import httpx
import pytest

RUN_GPU = os.getenv("AGENTTRACE_RUN_GPU_TESTS") == "1"
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not RUN_GPU, reason="GPU suite not requested")]


def test_cuda_is_actually_available() -> None:
    import torch

    assert torch.cuda.is_available(), "GPU suite requested but CUDA is unavailable"


def test_vllm_endpoint_advertises_selected_model() -> None:
    base_url = os.environ["AGENTTRACE_VLLM_URL"].rstrip("/")
    model = os.environ["AGENTTRACE_VLLM_MODEL"]
    response = httpx.get(base_url + "/models", timeout=10)
    response.raise_for_status()
    advertised = {entry["id"] for entry in response.json()["data"]}
    assert model in advertised

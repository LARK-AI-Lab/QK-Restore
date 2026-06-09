"""vLLM API server wrapper for HypeNet.

Registers HypeNetForCausalLMVLLM with vLLM's ModelRegistry BEFORE the server
starts, then launches the standard vLLM OpenAI-compatible API server.

Usage (mirrors `vllm serve`):
    python serve_hypenet_vllm.py --model /path/to/ckpt [vllm serve args]

Example:
    python serve_hypenet_vllm.py \\
        --model /path/to/hypenet-hf \\
        --dtype bfloat16 \\
        --port 8088 \\
        --gpu-memory-utilization 0.9

Environment:
    HALO_DIR  Override the path to the halo/ package root (default: parent of
              this script's directory, assuming standard repo layout).
"""
import os
import sys

# ---------------------------------------------------------------------------
# 1. Put halo/ on sys.path so modeling.hypenet.vllm_model can be imported
#    as a proper package.
# ---------------------------------------------------------------------------
HALO_DIR = os.environ.get(
    "HALO_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", ".."),  # <repo>/halo/
)
if HALO_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(HALO_DIR))

# ---------------------------------------------------------------------------
# 2. Register HypeNet with vLLM BEFORE any vLLM engine is constructed.
# ---------------------------------------------------------------------------
from modeling.hypenet.vllm_model import register_hypenet_vllm  # noqa: E402
register_hypenet_vllm()

# ---------------------------------------------------------------------------
# 3. Hand off to the standard vLLM API server (identical to `vllm serve`).
# ---------------------------------------------------------------------------
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args  # noqa: E402
from vllm.entrypoints.openai.api_server import run_server  # noqa: E402
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402

parser = make_arg_parser(FlexibleArgumentParser())
args = parser.parse_args()
validate_parsed_serve_args(args)
run_server(args)

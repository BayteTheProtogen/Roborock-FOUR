"""Explicit inference controls shared by the exporter and its benchmark."""

import os
from contextlib import nullcontext


DEFAULTS = {
    "precision": "fp32",
    "attention": "chunked",
    "attention_chunk": 256,
    "attention_scope": "global",
    "mlp_chunk": 0,
    "preprocess": "mps",
    "postprocess": "mps",
    "cache_policy": "auto",
    "prefer_metal": False,
    "fast_math": False,
    "cpu_threads": 0,
}


def add_inference_arguments(parser):
    parser.add_argument("--precision", choices=("fp32", "amp-fp16", "amp-bf16", "fp16"), default="fp32", help="Image inference precision; fp16 stores most weights in half precision")
    parser.add_argument("--attention", choices=("chunked", "sdpa"), default="chunked", help="MPS global vision attention implementation")
    parser.add_argument("--attention-chunk", type=int, choices=(64, 128, 256, 512, 1024), default=256)
    parser.add_argument("--attention-scope", choices=("global", "all"), default="global", help="Apply chunking to global attention only or also window attention")
    parser.add_argument("--mlp-chunk", type=int, choices=(0, 256, 512, 1024, 2048), default=0, help="Vision MLP token chunk size; 0 processes all tokens together")
    parser.add_argument("--preprocess", choices=("mps", "cpu"), default="mps", help="Image resize/normalization location (mps means model device)")
    parser.add_argument("--postprocess", choices=("mps", "cpu"), default="mps", help="Mask interpolation location (mps means model device)")
    parser.add_argument("--cache-policy", choices=("auto", "never", "always", "every-8"), default="auto")
    parser.add_argument("--prefer-metal", action="store_true", help="Prefer Metal matmul kernels in a fresh worker")
    parser.add_argument("--fast-math", action="store_true", help="Experimental relaxed Metal arithmetic; benchmark mask agreement")
    parser.add_argument("--cpu-threads", type=int, default=0, help="Torch CPU threads; 0 keeps the default")


def apply_environment(args):
    # Must run in the spawned worker before importing/initializing torch MPS.
    # PyTorch 2.10 tests presence for PREFER_METAL: even a value of "0" enables it.
    for key, option in (("PYTORCH_MPS_PREFER_METAL", "prefer_metal"), ("PYTORCH_MPS_FAST_MATH", "fast_math")):
        if getattr(args, option, False):
            os.environ[key] = "1"
        else:
            os.environ.pop(key, None)


def precision_context(device, precision):
    import torch

    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "amp-bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def configure_model(model, args):
    import torch
    from sam3.model.vitdet import Attention, Mlp

    if getattr(args, "precision", "fp32") == "fp16":
        model.half()
        # Retain stable normalization and the explicitly full-precision decoder FFN.
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.LayerNorm) or module.__class__.__name__ == "LayerNorm2d":
                module.float()
            if name.endswith((".linear1", ".linear2")):
                module.float()
    for module in model.modules():
        if isinstance(module, Attention):
            module.mps_attention = getattr(args, "attention", "chunked")
            module.mps_attention_chunk = getattr(args, "attention_chunk", 256)
            module.mps_attention_scope = getattr(args, "attention_scope", "global")
        if isinstance(module, Mlp):
            module.mps_mlp_chunk = getattr(args, "mlp_chunk", 0)


def exporter_flags(config, confidence, device):
    flags = ["--image-mode", "--inference-size", str(config["size"]), "--confidence", str(confidence), "--device", device]
    for key in ("precision", "attention", "attention_chunk", "attention_scope", "mlp_chunk", "preprocess", "postprocess", "cache_policy", "cpu_threads"):
        if config.get(key, DEFAULTS[key]) != DEFAULTS[key]:
            flags += ["--" + key.replace("_", "-"), str(config[key])]
    for key in ("prefer_metal", "fast_math"):
        if config.get(key):
            flags.append("--" + key.replace("_", "-"))
    return flags

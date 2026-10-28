"""A bounded, four-run MPS Metal matmul or fast-math comparison (off/on/on/off)."""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    from .benchmark_image_inference import benchmark_worker, extract_samples, mask_counts, read_mask, save_json, summarize_quality
    from .inference_options import DEFAULTS
except ImportError:
    from benchmark_image_inference import benchmark_worker, extract_samples, mask_counts, read_mask, save_json, summarize_quality
    from inference_options import DEFAULTS


def run(args):
    import shutil
    import torch
    from sam3.model_builder import download_ckpt_from_hf

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process")
    args.output.mkdir(parents=True, exist_ok=True)
    samples = extract_samples(args)
    checkpoint = str(args.checkpoint) if args.checkpoint else download_ckpt_from_hf()
    context = mp.get_context("spawn")
    deadline = time.monotonic() + args.time_budget
    option_key = "fast_math" if args.option == "fast-math" else "prefer_metal"
    variable = "PYTORCH_MPS_FAST_MATH" if args.option == "fast-math" else "PYTORCH_MPS_PREFER_METAL"
    rows = []
    reference = None
    for index, enabled in enumerate((False, True, True, False), 1):
        folder = args.output / f"run-{index}-{'on' if enabled else 'off'}"
        (folder / "masks").mkdir(parents=True)
        shutil.copytree(args.output / "samples" / "images", folder / "images")
        worker_args = SimpleNamespace(output=folder, device="mps", text=args.text,
            confidence=.5, inference_size=1008, no_overlays=True, alpha=.45,
            timing_report=True, memory_report=True, check_finite=True, profile_stages=True,
            mps_allocator_limit_gib=8, **{**DEFAULTS, option_key: enabled})
        worker = context.Process(target=benchmark_worker, args=(worker_args, samples, checkpoint))
        print(f"\nRun {index}/4: {args.option} {'ON' if enabled else 'OFF'}", flush=True)
        worker.start()
        try:
            worker.join(timeout=max(0, min(85, deadline - time.monotonic())))
            if worker.is_alive():
                raise RuntimeError("time budget exceeded; partial results retained")
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join()
        metrics = json.loads((folder / "run.json").read_text())
        if worker.exitcode or metrics.get("error"):
            raise RuntimeError(metrics.get("error") or f"worker exited {worker.exitcode}")
        expected_environment = {"PYTORCH_MPS_PREFER_METAL": None, "PYTORCH_MPS_FAST_MATH": None}
        expected_environment[variable] = "1" if enabled else None
        if metrics["mps_environment"] != expected_environment:
            raise RuntimeError("worker environment did not match the requested backend")
        frames = [json.loads(line) for line in (folder / "metadata.jsonl").read_text().splitlines()]
        timed = frames[args.warmup:]
        reference = reference or folder
        quality = summarize_quality([mask_counts(
            read_mask(folder / "masks" / f"{s['frame_index']:06d}.png"),
            read_mask(reference / "masks" / f"{s['frame_index']:06d}.png")) for s in samples])
        row = {"run": index, "enabled": enabled, option_key: enabled, **metrics, **quality,
            "mean_seconds": float(np.mean([f["frame_seconds"] for f in timed])),
            "mean_inference_seconds": float(np.mean([f["inference_seconds"] for f in timed])),
            "frame_seconds": [f["frame_seconds"] for f in timed]}
        rows.append(row)
        save_json(args.output / "partial_results.json", rows)
    summary = {}
    for enabled in (False, True):
        group = [r for r in rows if r["enabled"] == enabled]
        times = [t for r in group for t in r["frame_seconds"]]
        summary["on" if enabled else "off"] = {
            "mean_seconds": float(np.mean(times)), "median_seconds": float(np.median(times)),
            "run_means": [r["mean_seconds"] for r in group],
            "peak_mps_gib": max(r["mps_driver_peak_gib"] for r in group),
            "worst_mean_iou": min((r["mean_iou"] for r in group if r["mean_iou"] is not None), default=None),
            "worst_frame_iou": min((r["worst_frame_iou"] for r in group if r["worst_frame_iou"] is not None), default=None)}
    report = {"source": str(args.input.resolve()), "prompt": args.text, "tested_option": args.option,
        "fixed_options": {key: value for key, value in DEFAULTS.items() if key != option_key},
        "pytorch": torch.__version__, "samples": len(samples), "warmup_per_run": args.warmup,
        "order": ["off", "on", "on", "off"], "summary": summary, "runs": rows,
        "speedup": summary["off"]["mean_seconds"] / summary["on"]["mean_seconds"],
        "time_reduction_percent": 100 * (1 - summary["on"]["mean_seconds"] / summary["off"]["mean_seconds"]),
        "notes": ["Fresh process per run; disabled flag absent, enabled flag set to 1.",
                  "FP32, 1008 pixels, same frames and fixed text features; first warmup frames excluded.",
                  "Timing includes reading, inference, mask PNG output and cache cleanup; excludes model load.",
                  "Mask IoU measures agreement with first OFF run, not human accuracy.",
                  "Small repeated test; timing noise and other applications can affect results."]}
    save_json(args.output / "report.json", report)
    print(json.dumps({"summary": summary, "speedup": report["speedup"], "time_reduction_percent": report["time_reduction_percent"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--text", required=True)
    parser.add_argument("--option", choices=("metal-matmul", "fast-math"), default="metal-matmul", help="Toggle one option; keep the other disabled")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--time-budget", type=float, default=240, help="Seconds allowed for the four workers")
    args = parser.parse_args()
    args.sample_every = 0
    if not args.input.exists() or not args.text.strip() or args.samples <= args.warmup or args.warmup < 1 or not np.isfinite(args.time_budget) or args.time_budget <= 0:
        parser.error("input must exist, text must be nonempty, samples > warmup >= 1, and time budget positive")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("output must be a new or empty folder")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error("checkpoint must be an existing file")
    try:
        run(args)
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, OSError, ValueError) as error:
        print(f"error: {error}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

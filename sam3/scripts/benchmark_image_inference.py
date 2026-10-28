"""Compare SAM3 image sizes on shared frames, using labels or 1008px masks as reference."""

import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

try:
    from .export_video_overlay import FrameReader, infer_image_batch, write_image
    from .inference_options import DEFAULTS, apply_environment, exporter_flags
except ImportError:
    from export_video_overlay import FrameReader, infer_image_batch, write_image
    from inference_options import DEFAULTS, apply_environment, exporter_flags


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", type=Path, help="Video or image folder")
    p.add_argument("output", type=Path, help="New or empty benchmark folder")
    p.add_argument("--text", required=True)
    p.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--sizes", type=int, nargs="+", choices=tuple(range(336, 1009, 28)), default=[1008, 672, 504, 336])
    p.add_argument("--full-suite", action="store_true", help="Screen inference options, combine passing changes, repeat finalists and validate the winner")
    p.add_argument("--screen-samples", type=int, default=6)
    p.add_argument("--repeats", type=int, default=2, help="Independent runs per finalist in the full suite")
    p.add_argument("--validation-samples", type=int, default=48)
    p.add_argument("--mps-allocator-limit-gib", type=float, default=8, help="Full suite per-worker allocator guard; Metal framework allocations are separate")
    p.add_argument("--samples", type=int, default=12, help="Frames spread across the input")
    p.add_argument("--sample-every", type=int, default=0, help="Instead select every Nth frame; 0 samples evenly")
    p.add_argument("--warmup", type=int, default=1, help="Initial samples excluded from timing, included in quality")
    p.add_argument("--confidence", type=float, default=0.5)
    p.add_argument("--max-iou-loss", type=float, default=0.05, help="Maximum loss of both mean and aggregate mask IoU")
    p.add_argument("--min-frame-iou", type=float, default=0.8, help="Minimum individual nonempty-union frame IoU")
    p.add_argument("--max-peak-mps-gib", type=float, help="Optional sampled MPS driver memory ceiling")
    p.add_argument("--reference-masks", type=Path, help="Human binary PNG masks; otherwise reference is SAM3 at 1008")
    a = p.parse_args(argv)
    if not a.input.exists() or not a.text.strip():
        p.error("input must exist and --text must not be empty")
    if a.samples < 1 or a.sample_every < 0 or not 0 <= a.warmup < a.samples:
        p.error("samples must be positive, sample-every nonnegative, and 0 <= warmup < samples")
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in (a.confidence, a.max_iou_loss, a.min_frame_iou)):
        p.error("confidence and IoU thresholds must be finite values between 0 and 1")
    if a.max_peak_mps_gib is not None and (not np.isfinite(a.max_peak_mps_gib) or a.max_peak_mps_gib <= 0):
        p.error("max-peak-mps-gib must be positive and finite")
    if a.screen_samples <= a.warmup or a.validation_samples <= a.warmup or a.repeats < 2:
        p.error("screen/validation samples must exceed warmup; repeats must be at least 2")
    if not np.isfinite(a.mps_allocator_limit_gib) or a.mps_allocator_limit_gib <= 0:
        p.error("mps-allocator-limit-gib must be positive and finite")
    if a.output.exists() and (not a.output.is_dir() or any(a.output.iterdir())):
        p.error("output must be a new or empty folder")
    if a.checkpoint is not None and not a.checkpoint.is_file():
        p.error("checkpoint must be an existing file")
    if a.reference_masks is not None and not a.reference_masks.is_dir():
        p.error("reference-masks must be a folder")
    a.sizes = [1008] + [s for s in dict.fromkeys(a.sizes) if s != 1008]
    return a


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def extract_samples(args):
    folder = args.output / "samples" / "images"
    folder.mkdir(parents=True)
    reader = FrameReader(args.input)
    samples = []
    try:
        if not args.sample_every and reader.num_frames <= 0:
            raise ValueError("frame count unavailable; supply --sample-every")
        selected = set(np.linspace(0, reader.num_frames - 1, min(args.samples, reader.num_frames), dtype=int)) if not args.sample_every else None
        index = 0
        while (frame := reader.read()) is not None:
            keep = index % args.sample_every == 0 if args.sample_every else index in selected
            if keep:
                name = f"{index:06d}"
                write_image(folder / f"{name}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                original_stem = reader.paths[index].stem if reader.paths is not None else name
                samples.append({"frame_index": index, "time_seconds": index / reader.fps if args.input.is_file() else None, "reference_name": original_stem + ".png"})
                if len(samples) == args.samples:
                    break
            index += 1
    finally:
        reader.close()
    if len(samples) <= args.warmup:
        raise ValueError("not enough decoded samples after warmup; lower --warmup or use more frames")
    save_json(args.output / "samples" / "frames.json", samples)
    return samples


def read_mask(path, shape=None):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2 or not np.isin(mask, [0, 1, 255]).all():
        raise ValueError(f"expected a binary single-channel PNG mask: {path}")
    if shape is not None and mask.shape != shape:
        raise ValueError(f"mask dimensions differ from frame: {path}")
    return mask != 0


def mask_counts(prediction, reference):
    intersection = int(np.count_nonzero(prediction & reference))
    union = int(np.count_nonzero(prediction | reference))
    return {"intersection": intersection, "union": union, "reference_pixels": int(reference.sum()), "predicted_pixels": int(prediction.sum()), "iou": intersection / union if union else None}


def summarize_quality(rows):
    values = [row["iou"] for row in rows if row["iou"] is not None]
    intersection = sum(row["intersection"] for row in rows)
    union = sum(row["union"] for row in rows)
    reference = sum(row["reference_pixels"] for row in rows)
    predicted = sum(row["predicted_pixels"] for row in rows)
    return {"mean_iou": float(np.mean(values)) if values else None,
            "aggregate_iou": intersection / union if union else None,
            "worst_frame_iou": min(values) if values else None,
            "foreground_recall": intersection / reference if reference else None,
            "foreground_precision": intersection / predicted if predicted else None,
            "reference_pixels": reference, "scored_frames": len(values),
            "both_empty_frames": len(rows) - len(values)}


def system_swap_used():
    import psutil

    try:
        return psutil.swap_memory().used
    except (OSError, NotImplementedError):
        # Some macOS environments restrict this optional system-wide counter.
        return None


def benchmark_worker(worker_args, samples, checkpoint):
    """Fresh process per size. Memory peaks are sampled, including model setup."""
    apply_environment(worker_args)
    import psutil
    import torch

    process = psutil.Process()
    device = torch.device(worker_args.device)
    limit = getattr(worker_args, "mps_allocator_limit_gib", None)
    if device.type == "mps" and limit:
        torch.mps.set_per_process_memory_fraction(limit * 1024**3 / torch.mps.recommended_max_memory())
    peak = {"rss_peak_gib": 0.0, "mps_driver_peak_gib": None, "mps_tensor_peak_gib": None}
    stop = threading.Event()
    monitor_errors = []

    def measure():
        peak["rss_peak_gib"] = max(peak["rss_peak_gib"], process.memory_info().rss / 1024**3)
        if device.type == "mps":
            for key, method in (("mps_driver_peak_gib", torch.mps.driver_allocated_memory), ("mps_tensor_peak_gib", torch.mps.current_allocated_memory)):
                peak[key] = max(peak[key] or 0, method() / 1024**3)

    def monitor():
        try:
            while not stop.wait(0.1):
                measure()
        except Exception as error:
            monitor_errors.append(str(error))

    used_before = system_swap_used()
    measure()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    start = time.perf_counter()
    error = None
    try:
        infer_image_batch(worker_args, samples, checkpoint)
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    finally:
        stop.set()
        thread.join()
        measure()
        used_after = system_swap_used()
        save_json(worker_args.output / "run.json", {
            **peak, "worker_seconds": time.perf_counter() - start,
            "mps_environment": {key: os.environ.get(key) for key in ("PYTORCH_MPS_PREFER_METAL", "PYTORCH_MPS_FAST_MATH")},
            "system_swap_change_gib": (used_after - used_before) / 1024**3 if used_before is not None and used_after is not None else None,
            "memory_sampling_errors": monitor_errors, "error": error,
        })
    if error:
        print(error, file=sys.stderr, flush=True)
        raise SystemExit(1)


def recommend(results, max_iou_loss, min_frame_iou, max_peak_mps_gib=None):
    eligible = []
    for row in results:
        reasons = []
        if row.get("error"):
            reasons.append(row["error"])
        else:
            if not row["reference_pixels"]:
                reasons.append("reference contains no foreground; quality cannot be assessed")
            for metric in ("mean_iou", "aggregate_iou"):
                if row[metric] is None or row[metric] < 1 - max_iou_loss:
                    reasons.append(f"{metric} below {1 - max_iou_loss:.3f}")
            if row["worst_frame_iou"] is None or row["worst_frame_iou"] < min_frame_iou:
                reasons.append(f"worst frame IoU below {min_frame_iou:.3f}")
            if max_peak_mps_gib is not None:
                peak = row["mps_driver_peak_gib"]
                if peak is None or row.get("memory_sampling_errors") or peak > max_peak_mps_gib:
                    reasons.append("sampled MPS memory ceiling exceeded or unavailable")
        row["eligible"] = not reasons
        row["rejection_reasons"] = reasons
        if not reasons:
            eligible.append(row)
    return min(eligible, key=lambda row: row["mean_seconds"]) if eligible else None


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def comparison_previews(args, samples, results, reference_dir):
    folder = args.output / "comparisons"
    folder.mkdir()
    for sample in samples:
        stem = f"{sample['frame_index']:06d}"
        image = cv2.imread(str(args.output / "samples" / "images" / f"{stem}.jpg"))
        reference = read_mask(reference_dir / (sample["reference_name"] if args.reference_masks else f"{stem}.png"), image.shape[:2])
        panels = []
        successful = [row for row in results if not row.get("error")]
        # Keep wide screening sweeps readable: one original/reference pair per group.
        groups = [successful[i:i + 4] for i in range(0, len(successful), 4)]
        for group_index, group in enumerate(groups):
            panels = []
            for label, mask in [("Original", None), ("Reference", reference)] + [(row.get("config_id", f"{row['size']}px"), read_mask(args.output / row.get("config_id", str(row["size"])) / "masks" / f"{stem}.png")) for row in group]:
                view = image.copy()
                if mask is not None:
                    for region, color in ((mask & reference, (70, 200, 70)), (mask & ~reference, (50, 50, 240)), (~mask & reference, (240, 100, 50))):
                        view[region] = (view[region] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
                view = cv2.resize(view, (320, max(1, round(image.shape[0] * 320 / image.shape[1]))))
                banner = np.zeros((32, 320, 3), dtype=np.uint8)
                cv2.putText(banner, label[:35], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                panels.append(np.vstack([banner, view]))
            suffix = f"_{group_index}" if len(groups) > 1 else ""
            write_image(folder / f"{stem}{suffix}.jpg", np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 90])


def run(args):
    import torch
    from sam3.model.device import default_device
    from sam3.model_builder import download_ckpt_from_hf

    device = str(default_device()) if args.device == "auto" else args.device
    if args.max_peak_mps_gib is not None and device != "mps":
        raise ValueError("--max-peak-mps-gib requires MPS")
    args.output.mkdir(parents=True, exist_ok=True)
    samples = extract_samples(args)
    if args.reference_masks:
        for sample in samples:
            image = cv2.imread(str(args.output / "samples" / "images" / f"{sample['frame_index']:06d}.jpg"))
            read_mask(args.reference_masks / sample["reference_name"], image.shape[:2])
    checkpoint = str(args.checkpoint) if args.checkpoint else download_ckpt_from_hf()
    context = mp.get_context("spawn")
    results, per_frame = [], []
    print(f"Testing {len(samples)} shared frames; excluding first {args.warmup} from timing.", flush=True)
    reference_dir = args.reference_masks or args.output / "1008" / "masks"
    configurations = getattr(args, "configurations", None) or [{"id": str(size), "size": size, **DEFAULTS} for size in args.sizes]
    if configurations[0]["id"] != "1008" or configurations[0]["size"] != 1008:
        raise ValueError("first configuration must be the 1008 float32 baseline")
    for config in configurations:
        size = config["size"]
        folder = args.output / config["id"]
        (folder / "masks").mkdir(parents=True)
        shutil.copytree(args.output / "samples" / "images", folder / "images", copy_function=shutil.copy2)
        worker_args = SimpleNamespace(output=folder, device=device, text=args.text, confidence=args.confidence, inference_size=size, no_overlays=True, alpha=0.45, timing_report=True, memory_report=True, check_finite=True, profile_stages=True,
                                     **{key: config.get(key, value) for key, value in DEFAULTS.items()})
        worker_args.mps_allocator_limit_gib = getattr(args, "allocator_limit", None)
        print(f"\n=== {config['id']} ({size}px) ===", flush=True)
        worker = context.Process(target=benchmark_worker, args=(worker_args, samples, checkpoint))
        start = time.perf_counter()
        worker.start()
        try:
            worker.join()
        except KeyboardInterrupt:
            worker.terminate()
            worker.join()
            raise
        metrics = json.loads((folder / "run.json").read_text()) if (folder / "run.json").exists() else {"error": f"worker exited {worker.exitcode} without metrics"}
        row = {"size": size, "config_id": config["id"], "configuration": config, "total_run_seconds": time.perf_counter() - start, **metrics}
        if worker.exitcode != 0:
            row["error"] = row.get("error") or f"worker exited {worker.exitcode}"
        if not row.get("error"):
            frame_metrics = [json.loads(line) for line in (folder / "metadata.jsonl").read_text().splitlines()][args.warmup:]
            times = [frame["frame_seconds"] for frame in frame_metrics]
            if len(times) != len(samples) - args.warmup:
                raise ValueError("worker did not label every selected frame")
            quality = []
            for sample in samples:
                stem = f"{sample['frame_index']:06d}"
                predicted = read_mask(folder / "masks" / f"{stem}.png")
                reference = read_mask(reference_dir / (sample["reference_name"] if args.reference_masks else f"{stem}.png"), predicted.shape)
                counts = {"size": size, "config_id": config["id"], "frame_index": sample["frame_index"], **mask_counts(predicted, reference)}
                quality.append(counts)
                per_frame.append(counts)
            row.update(summarize_quality(quality))
            row.update(mean_seconds=float(np.mean(times)), median_seconds=float(np.median(times)), p90_seconds=float(np.percentile(times, 90)), timed_frames=len(times))
            row["mean_inference_seconds"] = float(np.mean([f["inference_seconds"] for f in frame_metrics]))
            row["backbone_seconds"] = float(np.mean([f["backbone_seconds"] for f in frame_metrics]))
            row["grounding_seconds"] = float(np.mean([f["grounding_seconds"] for f in frame_metrics]))
        results.append(row)
        save_json(args.output / "partial_results.json", results)
        if config["id"] == "1008" and row.get("error"):
            raise RuntimeError("1008 baseline failed; see partial_results.json and 1008/run.json")
    baseline = results[0]
    for row in results:
        if not row.get("error"):
            row["speedup_vs_1008"] = baseline["mean_seconds"] / row["mean_seconds"]
            row["frame_time_reduction_pct"] = 100 * (1 - row["mean_seconds"] / baseline["mean_seconds"])
            baseline_peak = baseline["mps_driver_peak_gib"]
            row["mps_driver_peak_reduction_pct"] = 100 * (1 - row["mps_driver_peak_gib"] / baseline_peak) if baseline_peak else None
            row["mean_iou_loss"] = 1 - row["mean_iou"] if row["mean_iou"] is not None else None
    best = recommend(results, args.max_iou_loss, args.min_frame_iou, args.max_peak_mps_gib)
    report = {"source": str(args.input.resolve()), "prompt": args.text, "confidence": args.confidence, "device": device, "samples": samples, "warmup": args.warmup,
              "checkpoint": str(Path(checkpoint).resolve()), "pytorch_version": torch.__version__,
              "reference": str(args.reference_masks.resolve()) if args.reference_masks else "SAM3 1008px (agreement proxy, not ground truth)",
              "policy": {"max_iou_loss": args.max_iou_loss, "min_frame_iou": args.min_frame_iou, "max_peak_mps_gib": args.max_peak_mps_gib},
              "recommended_size": best["size"] if best else None,
              "recommended_config": best["configuration"] if best else None,
              "recommended_exporter_flags": exporter_flags(best["configuration"], args.confidence, device) if best else None,
              "results": results,
              "notes": ["Both-empty frames are excluded from mean IoU; one-empty disagreements score zero.", "Memory sampled every 100ms including setup; brief peaks may be missed. RSS and Metal allocations overlap: do not add them.", "Swap change is system-wide, not attributable to this worker alone.", "Sizes run sequentially; keep other applications and power conditions unchanged.", "Ranking uses complete frame processing: image read, preprocessing, forward pass, CPU mask transfer, PNG output and cache cleanup; excludes model load and first warmup samples. Inference-only timing is recorded separately.", "Selection chooses fastest measured mean time satisfying mean, aggregate, and worst-frame IoU limits and optional memory limit.", "Comparison colors: green overlap, red extra prediction, blue missed reference."]}
    save_json(args.output / "report.json", report)
    write_csv(args.output / "results.csv", results)
    write_csv(args.output / "agreement.csv", per_frame)
    comparison_previews(args, samples, results, reference_dir)
    print("\nConfiguration                      Mean s/frame   Speedup   Mean IoU   Peak MPS GiB   Eligible")
    for row in results:
        if row.get("error"):
            print(f"{row['config_id']}: FAILED: {row['error']}")
            continue
        iou = f"{row['mean_iou']:.3f}" if row["mean_iou"] is not None else "n/a"
        peak = f"{row['mps_driver_peak_gib']:.2f}" if row["mps_driver_peak_gib"] is not None else "n/a"
        print(f"{row['config_id']:34}   {row['mean_seconds']:12.2f}   {row['speedup_vs_1008']:6.2f}x   {iou:>8}   {peak:>12}   {row['eligible']}")
    print(f"\nRecommended: {best['config_id']}" if best else "\nNo recommendation meets the quality/memory requirements; see report.json.")
    if best:
        print("Exporter flags: " + " ".join(report["recommended_exporter_flags"]))
    if not args.reference_masks:
        print("Quality means agreement with 1008px SAM3, not verified segmentation accuracy.")
    print(f"Results and visual comparisons: {args.output}")
    return report


def main(argv=None):
    try:
        args = parse_args(argv)
        if args.full_suite:
            try:
                from .inference_suite import run_full_suite
            except ImportError:
                from inference_suite import run_full_suite
            run_full_suite(args, run)
        else:
            run(args)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted; completed files remain in the output folder.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

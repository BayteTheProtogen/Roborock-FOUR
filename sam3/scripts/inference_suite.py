"""Screen independent controls, combine compatible winners, then validate repeats."""

import copy
import itertools
import json
from pathlib import Path

import numpy as np

try:
    from .inference_options import DEFAULTS, exporter_flags
except ImportError:
    from inference_options import DEFAULTS, exporter_flags


def baseline():
    return {"id": "1008", "size": 1008, **DEFAULTS}


def screening_configs(sizes):
    configs = [baseline()]
    controls = {
        "precision": ["amp-fp16", "amp-bf16", "fp16"],
        "attention_chunk": [64, 128, 512, 1024],
        "attention": ["sdpa"],
        "attention_scope": ["all"],
        "mlp_chunk": [256, 512, 1024, 2048],
        "preprocess": ["cpu"],
        "postprocess": ["cpu"],
        "cache_policy": ["never", "always", "every-8"],
        "prefer_metal": [True],
        "fast_math": [True],
        "cpu_threads": [1, 4],
        "size": sorted(set(sizes) | set(range(336, 1009, 28)), reverse=True),
    }
    for key, values in controls.items():
        for value in values:
            if value == baseline().get(key):
                continue
            configs.append({**baseline(), "id": f"{key}-{value}".lower(), key: value})
    # Retest high-memory kernels with smaller activations: failure alone does
    # not prove that a mixed-precision / chunking combination is unusable.
    for precision, metal in itertools.product(("fp32", "amp-fp16", "amp-bf16", "fp16"), (False, True)):
        configs.append({**baseline(), "id": f"bounded-{precision}-metal-{metal}".lower(),
                        "precision": precision, "prefer_metal": metal,
                        "attention_scope": "all", "mlp_chunk": 512})
    return configs


def combined_configs(rows):
    """Cross the two most promising numerical/attention choices and ancillary winners."""
    passing = sorted([row for row in rows if row["eligible"]], key=lambda row: row["mean_seconds"])
    if not passing:
        return [baseline()]
    base = baseline()
    ancillary = ("preprocess", "postprocess", "cache_policy", "prefer_metal", "fast_math", "cpu_threads", "mlp_chunk")
    for key in ancillary:
        candidates = [row for row in passing if row["configuration"].get(key) != base[key]]
        if candidates and candidates[0]["mean_seconds"] < rows[0]["mean_seconds"]:
            base[key] = candidates[0]["configuration"][key]
    precisions = ["fp32"]
    sizes = [1008]
    attentions = [("chunked", 256, "global")]
    for row in passing:
        cfg = row["configuration"]
        if cfg["precision"] not in precisions and len(precisions) < 3:
            precisions.append(cfg["precision"])
        if cfg["size"] not in sizes and len(sizes) < 2:
            sizes.append(cfg["size"])
        attention = (cfg["attention"], cfg["attention_chunk"], cfg["attention_scope"])
        if attention not in attentions and len(attentions) < 3:
            attentions.append(attention)
    configs = [baseline()]
    seen = {json.dumps({k: v for k, v in baseline().items() if k != "id"}, sort_keys=True)}
    for precision, size, (attention, chunk, scope) in itertools.product(precisions, sizes, attentions):
        cfg = {**base, "precision": precision, "size": size, "attention": attention, "attention_chunk": chunk, "attention_scope": scope}
        fingerprint = json.dumps({k: v for k, v in cfg.items() if k != "id"}, sort_keys=True)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        cfg["id"] = f"combo-{len(configs):02d}-{precision}-{size}-{chunk}"
        configs.append(cfg)
    # Preserve successful single changes so combinations cannot displace a better single.
    for row in passing[:3]:
        cfg = row["configuration"]
        fingerprint = json.dumps({k: v for k, v in cfg.items() if k != "id"}, sort_keys=True)
        if fingerprint not in seen:
            configs.append(cfg)
            seen.add(fingerprint)
    return configs


def repeat_summary(reports):
    groups = {}
    for report in reports:
        for row in report["results"]:
            groups.setdefault(row["config_id"], []).append(row)
    rows = []
    for key, runs in groups.items():
        success = len(runs) == len(reports) and all(row.get("eligible") for row in runs)
        times = [row["mean_seconds"] for row in runs if not row.get("error")]
        rows.append({"config_id": key, "configuration": runs[0]["configuration"],
                     "eligible_every_repeat": success,
                     "mean_seconds": float(np.mean(times)) if times else None,
                     "repeat_mean_min": min(times) if times else None,
                     "repeat_mean_max": max(times) if times else None,
                     "worst_mean_iou": min((row["mean_iou"] for row in runs if not row.get("error") and row["mean_iou"] is not None), default=None),
                     "worst_frame_iou": min((row["worst_frame_iou"] for row in runs if not row.get("error") and row["worst_frame_iou"] is not None), default=None),
                     "max_driver_gib": max((row["mps_driver_peak_gib"] for row in runs if row.get("mps_driver_peak_gib") is not None), default=None)})
    return sorted(rows, key=lambda row: row["mean_seconds"] if row["mean_seconds"] is not None else float("inf"))


def run_full_suite(args, run_benchmark):
    from sam3.model.device import default_device

    device = str(default_device()) if args.device == "auto" else args.device
    if device != "mps":
        raise ValueError("the full optimization suite targets MPS; use the size benchmark on CPU/CUDA")
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    status = {"phase": "screening", "quality_reference": "human masks" if args.reference_masks else "1008 float32 SAM3; agreement is not ground-truth accuracy", "phases": []}

    def save():
        (root / "suite.json").write_text(json.dumps(status, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def phase(name, configurations, samples):
        status["phase"] = name
        save()
        a = copy.copy(args)
        a.full_suite = False
        a.output = root / name
        a.samples = samples
        a.device = device
        a.configurations = configurations
        a.allocator_limit = args.mps_allocator_limit_gib
        print(f"\n### {name}: {len(configurations)} configurations, {samples} samples ###", flush=True)
        report = run_benchmark(a)
        status["phases"].append({"name": name, "configurations": len(configurations), "report": str(a.output / "report.json")})
        save()
        return report

    screen = phase("screening", screening_configs(args.sizes), args.screen_samples)
    combinations = phase("combinations", combined_configs(screen["results"]), args.screen_samples)
    passing = sorted([row for row in combinations["results"] if row["eligible"]], key=lambda row: row["mean_seconds"])
    finalist_rows = []
    # Preserve a full-resolution alternative before faster reduced-size candidates.
    full_resolution = [row for row in passing if row["size"] == 1008 and row["config_id"] != "1008"]
    if full_resolution:
        finalist_rows.append(full_resolution[0])
    signatures = {(row["size"], row["configuration"]["precision"]) for row in finalist_rows}
    for row in passing:
        signature = (row["size"], row["configuration"]["precision"])
        if row["config_id"] != "1008" and signature not in signatures:
            finalist_rows.append(row)
            signatures.add(signature)
        if len(finalist_rows) == 3:
            break
    finalists = [baseline()] + [row["configuration"] for row in finalist_rows]
    repeats = []
    for index in range(args.repeats):
        # Baseline stays first for reference. Reverse candidate order on alternate runs.
        order = finalists if index % 2 == 0 else [finalists[0]] + list(reversed(finalists[1:]))
        repeats.append(phase(f"repeat-{index + 1}", order, args.samples))
    repeated = repeat_summary(repeats)
    status["repeated_results"] = repeated
    candidates = [row for row in repeated if row["eligible_every_repeat"]]
    validation_configs = [baseline()] + [row["configuration"] for row in candidates if row["config_id"] != "1008"][:2]
    validation = phase("validation", validation_configs, args.validation_samples)
    valid_rows = {row["config_id"]: row for row in validation["results"] if row["eligible"]}
    # Require passing repeats AND longer validation; rank on both sets' measured time.
    validated = []
    for row in candidates:
        if row["config_id"] in valid_rows:
            final = valid_rows[row["config_id"]]
            validated.append({**row, "validated_mean_seconds": (row["mean_seconds"] * args.repeats + final["mean_seconds"]) / (args.repeats + 1), "validation_mean_iou": final["mean_iou"], "validation_worst_iou": final["worst_frame_iou"]})
    winner = min(validated, key=lambda row: row["validated_mean_seconds"]) if validated else None
    status["validated_results"] = validated
    status["recommended_config"] = winner["configuration"] if winner else None
    status["recommended_exporter_flags"] = exporter_flags(winner["configuration"], args.confidence, device) if winner else None
    status["phase"] = "complete"
    status["scope"] = "Independent controls, selected passing combinations, repeated finalists, and longer validation; not an exhaustive Cartesian product of every possible setting."
    save()
    print("\nFULL SUITE COMPLETE", flush=True)
    print("Recommended exporter flags: " + " ".join(status["recommended_exporter_flags"]) if winner else "No configuration passed every quality/memory gate.", flush=True)
    print(f"Full results: {root / 'suite.json'}", flush=True)
    return status

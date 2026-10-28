"""Training annotations, portable upload archives, and image-size decisions."""

import json
import math
import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit

import cv2
import numpy as np


LARGE_IMAGE_BYTES = 50_000_000  # MB, not MiB; warn at the boundary too.


def add_training_arguments(parser):
    parser.add_argument("--export-format", choices=("masks", "ndjson", "yolo", "coco"),
                        default="masks", help="Image mode training format; masks keeps the original export")
    parser.add_argument("--export-task", choices=("segment", "detect"), default="segment",
                        help="Training labels: instance polygons or bounding boxes")
    parser.add_argument("--class-name", action="append",
                        help="Class name for each --text, in order; repeat once per prompt or omit all")
    parser.add_argument("--ndjson-image-url-base",
                        help="Public HTTP(S) base containing images/train|val/ for standalone cloud NDJSON")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Fraction of selected images reserved for validation (last frames)")
    parser.add_argument("--chunk-size", type=int, default=0, metavar="IMAGES",
                        help="Maximum images per training ZIP; 0 means no count limit")
    parser.add_argument("--chunk-max-mb", type=float, default=0, metavar="MB",
                        help="Maximum size of each training ZIP in decimal MB; 0 means no size limit")
    parser.add_argument("--large-image-action", choices=("ask", "continue", "compress", "abort"),
                        default="ask", help="Action for images >=50 MB; ask prompts in the parent process")


def class_definitions(args):
    return getattr(args, "classes", None) or [
        {"id": 0, "name": getattr(args, "class_name", None) or args.text, "text": args.text}
    ]


def prepare_class_arguments(parser, args):
    prompts = args.text if isinstance(args.text, list) else [args.text]
    names = args.class_name or []
    if isinstance(names, str):
        names = [names]
    if any(not text.strip() for text in prompts):
        parser.error("--text must not be empty")
    if len(prompts) > 1 and not args.image_mode:
        parser.error("multiple --text prompts require --image-mode")
    if names and len(names) != len(prompts):
        parser.error("repeat --class-name once per --text, or omit all class names")
    names = names or prompts
    if any(not name.strip() for name in names):
        parser.error("--class-name must not be empty")
    if len(set(name.strip() for name in names)) != len(names):
        parser.error("class names must be unique")
    args.classes = [{"id": index, "name": name.strip(), "text": text.strip()}
                    for index, (name, text) in enumerate(zip(names, prompts))]
    # Existing video/benchmark callers use the first prompt as a string.
    args.text = prompts[0]
    args.class_name = args.classes[0]["name"] if args.class_name else None


def validate_training_arguments(parser, args):
    if args.chunk_size < 0:
        parser.error("--chunk-size must be nonnegative")
    if not math.isfinite(args.chunk_max_mb) or args.chunk_max_mb < 0:
        parser.error("--chunk-max-mb must be a nonnegative finite number")
    if not math.isfinite(args.val_fraction) or not 0 <= args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 (inclusive) and 1 (exclusive)")
    if args.class_name is not None and not args.class_name.strip():
        parser.error("--class-name must not be empty")
    if args.export_format == "masks" and (args.chunk_size or args.chunk_max_mb):
        parser.error("chunking requires --export-format ndjson, yolo, or coco")
    if args.ndjson_image_url_base:
        if args.export_format != "ndjson":
            parser.error("--ndjson-image-url-base requires --export-format ndjson")
        parsed = urlsplit(args.ndjson_image_url_base)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.query or parsed.fragment:
            parser.error("--ndjson-image-url-base must be an HTTP(S) base URL without a query or fragment")
    if not args.image_mode and (
        args.export_format != "masks" or args.export_task != "segment"
        or args.class_name is not None or args.val_fraction != 0.2
        or args.chunk_size or args.chunk_max_mb or args.large_image_action != "ask"
        or args.ndjson_image_url_base
    ):
        parser.error("training export options require --image-mode")


def large_image_decision(paths, action="ask"):
    """Return the chosen action; no worker ever reads stdin."""
    large = [(path, path.stat().st_size) for path in paths
             if path.stat().st_size >= LARGE_IMAGE_BYTES]
    if not large:
        return "continue"
    print(f"Warning: {len(large)} image(s) are 50 MB or larger:", flush=True)
    for path, size in large:
        print(f"  {path}: {size / 1_000_000:.2f} MB", flush=True)
    if action == "ask":
        if not sys.stdin.isatty():
            raise ValueError("images >=50 MB require a decision; rerun interactively or set "
                             "--large-image-action continue|compress|abort")
        while True:
            try:
                answer = input("Continue, compress exported copies, or abort? [c/compress/a]: ").strip().lower()
            except EOFError:
                answer = "abort"
            choices = {"c": "continue", "continue": "continue", "compress": "compress",
                       "a": "abort", "abort": "abort"}
            if answer in choices:
                action = choices[answer]
                break
            print("Enter continue, compress, or abort.", flush=True)
    if action == "abort":
        raise ValueError("export aborted because an image is 50 MB or larger")
    return action


def compress_image(path):
    """JPEG quality reduction only: keep dimensions and source files intact."""
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"cannot read image to compress: {path}")
    for quality in (90, 80, 70, 60, 50, 40, 30, 20, 10):
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise OSError(f"could not compress image: {path}")
        if encoded.nbytes < LARGE_IMAGE_BYTES:
            path.write_bytes(encoded.tobytes())
            print(f"Compressed {path.name}: {encoded.nbytes / 1_000_000:.2f} MB (quality {quality})", flush=True)
            return quality
    raise ValueError(f"cannot compress {path} below 50 MB without resizing; export stopped")


def mask_instances(masks, scores, class_id=0):
    """Keep separate SAM instances, including disconnected components."""
    instances = []
    for mask, score in zip(masks, scores):
        binary = np.asarray(mask, dtype=np.uint8)
        points = cv2.findNonZero(binary)
        if points is None:
            continue
        x, y, width, height = cv2.boundingRect(points)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            if len(contour) < 3 or cv2.contourArea(contour) == 0:
                px, py, pw, ph = cv2.boundingRect(contour)
                polygon = [px, py, px + pw, py, px + pw, py + ph, px, py + ph]
            else:
                polygon = contour.reshape(-1).tolist()
            polygons.append(polygon)
        instances.append({"bbox": [x, y, width, height], "area": int(binary.sum()),
                          "segmentation": polygons, "score": float(score), "class_id": class_id})
    return instances


def joined_polygon(polygons):
    """Bridge components with a doubled edge so YOLO has one row per instance."""
    joined = np.array(polygons[0], dtype=float).reshape(-1, 2)
    for polygon in polygons[1:]:
        part = np.array(polygon, dtype=float).reshape(-1, 2)
        # Bound temporary distance arrays for detailed masks.
        best = (float("inf"), 0, 0)
        for start in range(0, len(joined), 256):
            distances = ((joined[start:start + 256, None] - part[None]) ** 2).sum(axis=2)
            i, j = np.unravel_index(distances.argmin(), distances.shape)
            candidate = (distances[i, j], start + i, j)
            if candidate[0] < best[0]:
                best = candidate
        _, i, j = best
        loop = np.concatenate((part[j:], part[:j + 1]))
        joined = np.concatenate((joined[:i + 1], loop, joined[i:]))
    return joined


def label_rows(row, task):
    if row.get("_label_task") == task:
        return row["_labels"]
    labels = []
    for instance in row["instances"]:
        if task == "detect":
            x, y, w, h = instance["bbox"]
            values = [(x + w / 2) / row["width"], (y + h / 2) / row["height"],
                      w / row["width"], h / row["height"]]
        else:
            polygon = joined_polygon(instance["segmentation"])
            values = (polygon / [row["width"], row["height"]]).reshape(-1).tolist()
        labels.append([instance.get("class_id", 0), *values])
    return labels


def yolo_entries(rows, task, class_names):
    splits = {row["split"] for row in rows}
    yaml = f"train: images/train\nval: {'images/val' if 'val' in splits else ''}\nnames:\n"
    yaml += "".join(f"  {index}: {json.dumps(name)}\n" for index, name in enumerate(class_names))
    entries = [("data.yaml", yaml.encode())]
    for row in rows:
        labels = label_rows(row, task)
        content = "".join(" ".join(str(v) for v in label) + "\n" for label in labels)
        entries.append((f"labels/{row['split']}/{Path(row['image']).stem}.txt", content.encode()))
    return entries


def package_entries(source, rows, export_format, task, class_names, image_url_base=None):
    """Entries for a self-contained dataset; filenames resolve after extraction."""
    entries = []
    if isinstance(class_names, str):
        class_names = [class_names]
    for row in rows:
        image = Path(row["image"])
        entries.append((f"images/{row['split']}/{image.name}", source / image))
    if export_format == "yolo":
        # Omitting path lets Ultralytics resolve relative to data.yaml on any machine.
        entries.extend(yolo_entries(rows, task, class_names))
    elif export_format == "ndjson":
        header = {"type": "dataset", "task": task, "name": source.name,
                  "class_names": {str(index): name for index, name in enumerate(class_names)}, "version": 0}
        if not image_url_base:
            header["path"] = "."
        records = [header]
        for row in rows:
            record = {"type": "image", "file": Path(row["image"]).name,
                      "width": row["width"], "height": row["height"], "split": row["split"],
                      "annotations": {"boxes" if task == "detect" else "segments": label_rows(row, task)}}
            if image_url_base:
                relative = f"images/{row['split']}/{Path(row['image']).name}"
                record["url"] = urljoin(image_url_base.rstrip("/") + "/", quote(relative))
            records.append(record)
        entries.append(("dataset.ndjson", ("".join(json.dumps(r) + "\n" for r in records)).encode()))
        if not image_url_base:
            # Cloud NDJSON needs reachable URLs; a local path is an SDK feature.
            # Platform prioritizes data.yaml, so companion labels import local pixels
            # without asking the cloud importer to fetch the path-only NDJSON rows.
            entries.extend(yolo_entries(rows, task, class_names))
    else:
        for split in sorted({row["split"] for row in rows}):
            images, annotations = [], []
            for row in rows:
                if row["split"] != split:
                    continue
                image_id = row["frame_index"] + 1
                images.append({"id": image_id, "file_name": f"images/{split}/{Path(row['image']).name}",
                               "width": row["width"], "height": row["height"]})
                for instance in row["instances"]:
                    annotation = {"id": instance.get("annotation_id", len(annotations) + 1), "image_id": image_id,
                                  "category_id": instance.get("class_id", 0) + 1,
                                  "bbox": instance["bbox"], "area": instance["area"], "iscrowd": 0}
                    if task == "segment":
                        annotation["segmentation"] = instance["segmentation"]
                    annotations.append(annotation)
            coco = {"images": images, "annotations": annotations,
                    "categories": [{"id": index + 1, "name": name, "supercategory": "object"}
                                   for index, name in enumerate(class_names)]}
            entries.append((f"annotations/instances_{split}.json", json.dumps(coco).encode()))
    return entries


def archive_size(entries):
    """Exact size for ZIP_STORED entries below ZIP64 limits (includes headers)."""
    return 22 + sum((value.stat().st_size if isinstance(value, Path) else len(value))
                    + 76 + 2 * len(name.encode("utf-8")) for name, value in entries)


def export_training_dataset(args, destination=None):
    source = args.output
    with (source / "metadata.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("cannot export an empty training dataset")
    if any("instances" not in row for row in rows):
        raise ValueError("dataset metadata has no instance annotations; label it again with a training export format")
    val_count = min(len(rows) - 1, max(1, round(len(rows) * args.val_fraction))) if args.val_fraction else 0
    for index, row in enumerate(rows):
        row["split"] = "val" if index >= len(rows) - val_count else "train"
    annotation_id = 0
    for row in rows:
        for instance in row["instances"]:
            annotation_id += 1
            instance["annotation_id"] = annotation_id
        if args.export_format != "coco":
            row["_labels"] = label_rows(row, args.export_task)
            row["_label_task"] = args.export_task
    if val_count == 0:
        print("Training export has no validation images; add a held-out set before training.", flush=True)
    class_names = [item["name"] for item in class_definitions(args)]
    image_url_base = getattr(args, "ndjson_image_url_base", None)
    def entries_for(group):
        return package_entries(source, group, args.export_format, args.export_task, class_names, image_url_base)

    destination = Path(destination) if destination is not None else source / "training"
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("training output must be a new or empty directory")
    destination.mkdir(parents=True, exist_ok=True)
    limit = args.chunk_max_mb * 1_000_000
    chunked = bool(args.chunk_size or limit)
    # Assign splits before chunking so images never change split or get duplicated.
    groups, group, estimated_bytes = [], [], 22
    for row in rows:
        # Charge each image its own metadata/header overhead. This is a conservative
        # upper bound for a combined ZIP and keeps partitioning linear in image count.
        row_bytes = archive_size(entries_for([row])) - 22 if limit else 0
        if group and ((args.chunk_size and len(group) >= args.chunk_size)
                      or (limit and estimated_bytes + row_bytes > limit)):
            groups.append(group)
            group = []
            estimated_bytes = 22
        group.append(row)
        estimated_bytes += row_bytes
        if limit and estimated_bytes > limit:
            raise ValueError(f"image {row['image']} and its annotations cannot fit --chunk-max-mb; increase the limit")
    groups.append(group)
    outputs = []
    for index, group in enumerate(groups, 1):
        entries = entries_for(group)
        archive = destination / (f"chunk-{index:04d}.zip" if chunked else "dataset.zip")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as handle:
            for name, value in entries:
                if isinstance(value, Path):
                    handle.write(value, name)
                else:
                    handle.writestr(name, value)
        if limit and archive.stat().st_size > limit:
            archive.unlink()
            raise ValueError("ZIP64 overhead exceeded --chunk-max-mb; increase the limit")
        if args.export_format == "ndjson" and image_url_base:
            ndjson_path = archive.with_suffix(".ndjson")
            ndjson_path.write_bytes(next(value for name, value in entries if name == "dataset.ndjson"))
        outputs.append({"file": archive.name, "images": len(group), "bytes": archive.stat().st_size,
                        "train": sum(r["split"] == "train" for r in group),
                        "val": sum(r["split"] == "val" for r in group)})
        print(f"Training export: {archive} ({len(group)} images, {archive.stat().st_size / 1_000_000:.2f} MB)", flush=True)
    manifest = {"format": args.export_format, "task": args.export_task, "class_names": class_names,
                "platform_import_format": "yolo" if args.export_format == "ndjson" and not image_url_base else args.export_format,
                "ndjson_image_url_base": image_url_base,
                "val_fraction": args.val_fraction, "chunks": outputs}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.export_format == "ndjson" and not image_url_base:
        print("Upload the ZIP to Platform (NDJSON + YOLO companion labels). "
              "Its embedded NDJSON uses local paths; standalone cloud NDJSON needs --ndjson-image-url-base.",
              flush=True)
    return manifest


def main(argv=None):
    """Repackage saved annotations without loading SAM or replacing old exports."""
    parser = argparse.ArgumentParser(description="Re-export a labeled dataset without running SAM again.")
    parser.add_argument("dataset", type=Path, help="Folder containing dataset.json, metadata.jsonl, and images/")
    parser.add_argument("destination", type=Path, help="New or empty training archive output folder")
    add_training_arguments(parser)
    parser.set_defaults(export_format="ndjson", export_task=None, val_fraction=None)
    args = parser.parse_args(argv)
    try:
        config = json.loads((args.dataset / "dataset.json").read_text(encoding="utf-8"))
        saved_classes = config.get("classes") or [
            {"text": config["text_prompt"], "name": config.get("class_name") or config["text_prompt"]}
        ]
        args.text = [item["text"] for item in saved_classes]
        if args.class_name is None:
            args.class_name = [item["name"] for item in saved_classes]
        args.image_mode = True
        args.export_task = args.export_task or config.get("export_task", "segment")
        if args.val_fraction is None:
            args.val_fraction = config.get("val_fraction", 0.2)
        prepare_class_arguments(parser, args)
        validate_training_arguments(parser, args)
        if args.export_format == "masks":
            parser.error("re-export requires ndjson, yolo, or coco")
        if args.destination.exists() and (not args.destination.is_dir() or any(args.destination.iterdir())):
            raise ValueError("training output must be a new or empty directory")
        with (args.dataset / "metadata.jsonl").open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        decision = large_image_decision([args.dataset / row["image"] for row in rows], args.large_image_action)
        args.output = args.dataset
        if decision == "compress":
            with tempfile.TemporaryDirectory(prefix="training-export-") as folder:
                staging = Path(folder) / args.dataset.name
                staging.mkdir()
                shutil.copy2(args.dataset / "metadata.jsonl", staging / "metadata.jsonl")
                for row in rows:
                    source = args.dataset / row["image"]
                    target = staging / row["image"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    if target.stat().st_size >= LARGE_IMAGE_BYTES:
                        compress_image(target)
                args.output = staging
                export_training_dataset(args, args.destination)
        else:
            export_training_dataset(args, args.destination)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

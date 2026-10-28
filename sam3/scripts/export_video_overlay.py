"""Export a SAM 3 text-prompted video or an image segmentation dataset.

Example:
    python scripts/export_video_overlay.py input.mp4 output.mp4 --text "a dog"
    python scripts/export_video_overlay.py input.mp4 dataset --image-mode --text "sidewalk"
    python scripts/export_video_overlay.py input.mp4 dataset --image-mode --text "sidewalk" --export-format yolo
"""

import argparse
import json
import math
import multiprocessing as mp
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
try:
    from .inference_options import DEFAULTS, add_inference_arguments, apply_environment, configure_model, precision_context
    from .training_export import (
        LARGE_IMAGE_BYTES, add_training_arguments, validate_training_arguments,
        large_image_decision, compress_image, mask_instances, export_training_dataset,
        prepare_class_arguments, class_definitions,
    )
except ImportError:
    from inference_options import DEFAULTS, add_inference_arguments, apply_environment, configure_model, precision_context
    from training_export import (
        LARGE_IMAGE_BYTES, add_training_arguments, validate_training_arguments,
        large_image_decision, compress_image, mask_instances, export_training_dataset,
        prepare_class_arguments, class_definitions,
    )


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_PRESETS = {
    "quality": {"inference_size": 1008, "precision": "fp16"},
    "fast": {"inference_size": 840, "precision": "fp32"},
    "full": {"inference_size": 1008, "precision": "fp32"},
}
PROMPT_BATCHING = {"high": 1.0, "medium": 0.5, "low": 0.25, "none": 0.0}
PALETTE_RGB = (
    (255, 99, 71),
    (64, 196, 255),
    (255, 205, 75),
    (137, 220, 119),
    (208, 131, 255),
    (255, 132, 190),
    (80, 224, 199),
    (255, 155, 90),
)


def prompt_batch_size(class_count, level="none"):
    """Fraction of class prompts per grounding call, rounded up to one or more."""
    if class_count < 1:
        raise ValueError("at least one class is required")
    return max(1, math.ceil(class_count * PROMPT_BATCHING[level]))


def parse_args(argv=None):
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument("--preset", choices=tuple(IMAGE_PRESETS))
    preset = preset_parser.parse_known_args(argv)[0].preset
    parser = argparse.ArgumentParser(
        description="Segment objects by text and export an overlay MP4 or image dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=tuple(IMAGE_PRESETS),
                        help="Image mode: quality=1008/FP16, fast=840/FP32, full=1008/FP32; explicit size/precision flags override the preset")
    parser.add_argument("input", type=Path, help="Input video, image folder, or video folder (image mode)")
    parser.add_argument("output", type=Path, help="Output .mp4 or image dataset folder")
    parser.add_argument("--text", required=True, action="append",
                        help="Object description; repeat for multiple classes in image mode")
    parser.add_argument("--checkpoint", type=Path, help="Local SAM 3 checkpoint")
    parser.add_argument(
        "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
    )
    parser.add_argument(
        "--prompt-frame", type=int, default=0, help="Frame receiving the text prompt"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.45, help="Mask opacity from 0 to 1"
    )
    parser.add_argument(
        "--fps", type=float, help="Output FPS; source FPS for video, 30 for a folder"
    )
    parser.add_argument(
        "--no-audio", action="store_true", help="Omit audio from a video input"
    )
    parser.add_argument(
        "--offload-video-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Store decoded frames on CPU (default on MPS)",
    )
    parser.add_argument(
        "--offload-state-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Store tracking state on CPU (default on MPS)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output MP4"
    )
    parser.add_argument(
        "--memory-report",
        action="store_true",
        help="Print process and MPS memory use during inference",
    )
    parser.add_argument(
        "--restart-every",
        type=int,
        default=None,
        metavar="FRAMES",
        help="Restart every N frames; 0 keeps one model loaded (image mode defaults to 3 on MPS)",
    )
    parser.add_argument(
        "--image-mode",
        action="store_true",
        help="Process frames independently and save images, masks, and overlays to a folder",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        metavar="N",
        help="In image mode, process every Nth input frame; restarts at frame 0 for each video",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        metavar="N",
        help="In image mode, stop after N selected frames across all inputs; 0 processes all",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="In image mode, minimum SAM 3 mask score",
    )
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="In image mode, save training images and masks without overlay previews",
    )
    parser.add_argument(
        "--drop-empty", action="store_true",
        help="In image mode, delete exported images with zero masks across all classes and exclude them from the dataset",
    )
    parser.add_argument(
        "--inference-size",
        type=int,
        choices=tuple(range(336, 1009, 28)),
        default=1008,
        help="Image mode internal size; smaller uses less activation memory but may lose detail",
    )
    add_inference_arguments(parser)
    add_training_arguments(parser)
    parser.add_argument(
        "--batching", choices=tuple(PROMPT_BATCHING), default="none",
        help="Image mode class prompts per call: high=100%%, medium=50%%, low=25%%, none=one; rounded up",
    )
    parser.add_argument(
        "--timing-report",
        action="store_true",
        help="In image mode, report model loading, per-frame inference, and total export time",
    )
    parser.set_defaults(**IMAGE_PRESETS.get(preset, {}))
    args = parser.parse_args(argv)
    prepare_class_arguments(parser, args)
    validate_training_arguments(parser, args)
    if args.drop_empty and not args.image_mode:
        parser.error("--drop-empty requires --image-mode")
    if not args.image_mode and args.batching != "none":
        parser.error("--batching requires --image-mode")
    if not args.text.strip():
        parser.error("--text must not be empty")
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be a positive finite number")
    if args.prompt_frame < 0:
        parser.error("--prompt-frame must be nonnegative")
    if args.restart_every is not None and args.restart_every < 0:
        parser.error("--restart-every must be nonnegative")
    if args.cpu_threads < 0:
        parser.error("--cpu-threads must be nonnegative")
    if args.sample_every < 1:
        parser.error("--sample-every must be positive")
    if args.max_samples < 0:
        parser.error("--max-samples must be nonnegative")
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    if args.image_mode:
        if args.output.exists() and (
            not args.output.is_dir() or any(args.output.iterdir())
        ):
            parser.error("image-mode output must be a new or empty directory")
        if args.fps is not None or args.no_audio or args.prompt_frame != 0:
            parser.error("--fps, --no-audio, and --prompt-frame are for video export")
        if args.offload_video_to_cpu is not None or args.offload_state_to_cpu is not None:
            parser.error("--offload-video-to-cpu and --offload-state-to-cpu are for video export")
        if args.overwrite:
            parser.error("--overwrite is for video export; choose a new image-mode folder")
    else:
        if args.preset is not None:
            parser.error("--preset requires --image-mode")
        if args.restart_every and (args.device != "mps" or not args.input.is_file()):
            parser.error("--restart-every requires --device mps and a video file")
        if args.restart_every and args.prompt_frame != 0:
            parser.error("--restart-every currently requires --prompt-frame 0")
        if args.sample_every != 1 or args.max_samples or args.confidence != 0.5:
            parser.error("--sample-every, --max-samples, and --confidence require --image-mode")
        if args.no_overlays or args.timing_report:
            parser.error("--no-overlays and --timing-report require --image-mode")
        if args.inference_size != 1008:
            parser.error("--inference-size requires --image-mode")
        if any(getattr(args, key) != value for key, value in DEFAULTS.items()):
            parser.error("image inference optimization flags require --image-mode")
        if args.output.suffix.lower() != ".mp4":
            parser.error("output must have a .mp4 extension")
    if not args.input.exists():
        parser.error(f"input does not exist: {args.input}")
    if args.input.is_file() and args.input.suffix.lower() not in VIDEO_SUFFIXES:
        parser.error("input must be a video file or a folder of images or videos")
    if args.input.is_dir():
        try:
            sources = image_dataset_sources(args.input)
        except ValueError as error:
            parser.error(str(error))
        if not args.image_mode and sources != [args.input]:
            parser.error("folders of videos require --image-mode")
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different paths")
    if not args.image_mode and args.output.exists() and not args.overwrite:
        parser.error("output exists; pass --overwrite to replace it")
    if not args.image_mode and args.output.is_dir():
        parser.error("output must be a file path")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if not args.image_mode and shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required to encode the output video")
    return args


def write_image(path, image, parameters=None):
    if not cv2.imwrite(str(path), image, parameters or []):
        raise OSError(f"could not write image: {path}")


def frame_paths(folder):
    paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not paths:
        raise ValueError(f"no image frames found in {folder}")
    try:
        return sorted(paths, key=lambda p: int(p.stem))
    except ValueError:
        return sorted(paths)


def image_dataset_sources(path):
    """Image folders stay one source; video folders yield ordered individual clips."""
    if not path.is_dir():
        return [path]
    files = [item for item in path.iterdir() if item.is_file()]
    images = [item for item in files if item.suffix.lower() in IMAGE_SUFFIXES]
    videos = sorted(item for item in files if item.suffix.lower() in VIDEO_SUFFIXES)
    if images and videos:
        raise ValueError("input folder mixes images and videos; use a folder containing only one media type")
    if images:
        return [path]
    if videos:
        return videos
    raise ValueError(f"no supported images or videos found directly in {path}")


class FrameReader:
    def __init__(self, path):
        self.path = path
        self.capture = None
        self.paths = None
        self.position = 0
        if path.is_dir():
            self.paths = frame_paths(path)
            self.num_frames = len(self.paths)
            self.fps = 30.0
        else:
            self.capture = cv2.VideoCapture(str(path))
            if not self.capture.isOpened():
                raise ValueError(f"cannot open video: {path}")
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            self.fps = fps if math.isfinite(fps) and fps > 0 else 30.0
            self.num_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self):
        if self.paths is not None:
            if self.position >= len(self.paths):
                return None
            path = self.paths[self.position]
            self.position += 1
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"cannot read frame: {path}")
            return frame
        ok, frame = self.capture.read()
        return frame if ok else None

    def close(self):
        if self.capture is not None:
            self.capture.release()


class VideoWriter:
    def __init__(self, output, width, height, fps, audio_source=None, overwrite=False):
        command = [
            "ffmpeg", "-loglevel", "error", "-y" if overwrite else "-n",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
        ]
        if audio_source is not None:
            command += ["-i", str(audio_source)]
        command += ["-map", "0:v:0"]
        if audio_source is not None:
            command += ["-map", "1:a?", "-c:a", "aac", "-b:a", "192k"]
        command += [
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ]
        self.output = output
        self.size = (height, width)
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def write(self, frame):
        if frame.shape[:2] != self.size:
            raise ValueError("input frames have different sizes")
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as error:
            message = self.process.stderr.read().decode(errors="replace")
            raise RuntimeError(f"ffmpeg stopped while encoding: {message}") from error

    def close(self):
        self.process.stdin.close()
        message = self.process.stderr.read().decode(errors="replace")
        if self.process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {message}")

    def abort(self):
        if not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait()


def overlay_masks(frame, outputs, alpha):
    if not outputs or alpha == 0:
        return frame
    masks = np.asarray(outputs.get("out_binary_masks", []))
    object_ids = np.asarray(outputs.get("out_obj_ids", []))
    if len(masks) != len(object_ids):
        raise ValueError("predictor returned different numbers of masks and object IDs")
    blended = frame.copy()
    for object_id, mask in zip(object_ids, masks):
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(
                mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        selected = mask.astype(bool)
        color_bgr = np.array(PALETTE_RGB[int(object_id) % len(PALETTE_RGB)][::-1])
        blended[selected] = (
            (1 - alpha) * blended[selected] + alpha * color_bgr
        ).astype(np.uint8)
    return blended


def report_memory(stage, device):
    import os
    import psutil

    rss_gib = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    message = f"memory at {stage}: process RSS {rss_gib:.2f} GiB"
    if device.type == "mps":
        import torch

        mps_gib = torch.mps.current_allocated_memory() / 1024**3
        driver_gib = torch.mps.driver_allocated_memory() / 1024**3
        message += f", MPS tensors {mps_gib:.2f} GiB, MPS driver {driver_gib:.2f} GiB"
    print(message, file=sys.stderr, flush=True)


def release_mps_cache(device, force=False):
    if device.type != "mps":
        return
    import torch

    torch.mps.synchronize()
    driver_bytes = torch.mps.driver_allocated_memory()
    unused_bytes = driver_bytes - torch.mps.current_allocated_memory()
    if force or (driver_bytes > 8 * 1024**3 and unused_bytes > 2 * 1024**3):
        torch.mps.empty_cache()


def infer_segment(
    args,
    start_frame,
    frame_count,
    result_path,
    checkpoint_path,
    offload_video_to_cpu,
    offload_state_to_cpu,
):
    """Run a short segment in a fresh process so MPS driver memory is released on exit."""
    from sam3.model_builder import build_sam3_video_predictor
    import torch

    device = torch.device(args.device)
    predictor = build_sam3_video_predictor(
        checkpoint_path=checkpoint_path,
        load_from_HF=checkpoint_path is not None,
        device=device,
        video_loader_type="cv2_lazy",
    )
    session_id = None
    try:
        if args.memory_report:
            report_memory(f"segment {start_frame} model load", device)
        response = predictor.handle_request(
            dict(
                type="start_session",
                resource_path=str(args.input),
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu,
            )
        )
        session_id = response["session_id"]
        predictor.handle_request(
            dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=start_frame,
                text=args.text,
            )
        )
        release_mps_cache(device, force=True)
        if args.memory_report:
            report_memory(f"segment {start_frame} prompt", device)

        if device.type == "mps":
            def after_inference_frame(frame_index, frame_output):
                release_mps_cache(device)
                if args.memory_report:
                    report_memory(f"segment {start_frame} frame {frame_index}", device)

            predictor.model.frame_inference_hook = after_inference_frame

        arrays = {}
        seen = set()
        for result in predictor.handle_stream_request(
            dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction="forward",
                start_frame_index=start_frame,
                max_frame_num_to_track=frame_count - 1,
                evict_cached_frame_outputs=True,
            )
        ):
            offset = result["frame_index"] - start_frame
            if not 0 <= offset < frame_count:
                raise ValueError("predictor returned a frame outside the segment")
            outputs = result["outputs"]
            arrays[f"ids_{offset}"] = np.asarray(outputs["out_obj_ids"])
            arrays[f"masks_{offset}"] = np.asarray(outputs["out_binary_masks"])
            seen.add(offset)
        if seen != set(range(frame_count)):
            raise RuntimeError("predictor did not return every frame in the segment")
        np.savez(result_path, **arrays)
    finally:
        try:
            if session_id is not None:
                predictor.handle_request(
                    dict(type="close_session", session_id=session_id)
                )
        finally:
            predictor.shutdown()


def export_video_in_segments(args):
    """Preserve the source video while restarting inference at segment boundaries."""
    from sam3.model_builder import download_ckpt_from_hf

    reader = FrameReader(args.input)
    writer = None
    try:
        frame = reader.read()
        if frame is None or reader.num_frames <= 0:
            raise ValueError("input contains no decodable frames")
        if (
            args.fps is not None
            and not args.no_audio
            and abs(args.fps - reader.fps) > 0.001
        ):
            raise ValueError("changing video FPS requires --no-audio to avoid audio drift")
        fps = args.fps or reader.fps
        checkpoint_path = (
            str(args.checkpoint) if args.checkpoint else download_ckpt_from_hf()
        )
        offload_video_to_cpu = (
            True if args.offload_video_to_cpu is None else args.offload_video_to_cpu
        )
        offload_state_to_cpu = (
            True if args.offload_state_to_cpu is None else args.offload_state_to_cpu
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = VideoWriter(
            args.output,
            frame.shape[1],
            frame.shape[0],
            fps,
            None if args.no_audio else args.input,
            overwrite=args.overwrite,
        )

        context = mp.get_context("spawn")
        with tempfile.TemporaryDirectory(prefix="sam3-overlay-") as temp_dir:
            for start in range(0, reader.num_frames, args.restart_every):
                count = min(args.restart_every, reader.num_frames - start)
                result_path = Path(temp_dir) / f"segment-{start}.npz"
                print(f"Segment {start // args.restart_every + 1}: frames {start}-{start + count - 1}", flush=True)
                worker = context.Process(
                    target=infer_segment,
                    args=(
                        args,
                        start,
                        count,
                        result_path,
                        checkpoint_path,
                        offload_video_to_cpu,
                        offload_state_to_cpu,
                    ),
                )
                worker.start()
                try:
                    worker.join()
                except KeyboardInterrupt:
                    worker.terminate()
                    worker.join()
                    raise
                if worker.exitcode != 0:
                    raise RuntimeError(
                        f"inference failed for frames {start}-{start + count - 1} "
                        f"(exit code {worker.exitcode})"
                    )
                with np.load(result_path) as results:
                    for offset in range(count):
                        if frame is None:
                            raise ValueError("video ended before its reported frame count")
                        outputs = {
                            "out_obj_ids": results[f"ids_{offset}"],
                            "out_binary_masks": results[f"masks_{offset}"],
                        }
                        writer.write(overlay_masks(frame, outputs, args.alpha))
                        frame = reader.read()
        if frame is not None:
            raise ValueError("video contains more frames than its reported frame count")
        writer.close()
        writer = None
        print(f"Exported {reader.num_frames} frames to {args.output}")
    finally:
        if writer is not None:
            writer.abort()
        reader.close()


def release_image_cache(args, sample_number, device):
    policy = getattr(args, "cache_policy", "auto")
    if policy == "auto":
        release_mps_cache(device)
    elif policy == "always" or (policy == "every-8" and sample_number % 8 == 0):
        release_mps_cache(device, force=True)


def infer_image_batch(args, samples, checkpoint_path):
    apply_environment(args)
    import torch

    if getattr(args, "cpu_threads", 0):
        torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    with precision_context(device, getattr(args, "precision", "fp32")):
        return _infer_image_batch(args, samples, checkpoint_path)


def _infer_image_batch(args, samples, checkpoint_path):
    """Label a small batch in its own process to release MPS driver memory."""
    import gc
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    import torch

    device = torch.device(args.device)
    load_started = time.perf_counter()
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
        device="cpu" if getattr(args, "precision", "fp32") == "fp16" else device,
        image_size=args.inference_size,
    )
    configure_model(model, args)
    if getattr(args, "precision", "fp32") == "fp16":
        model.to(device)
    processor = Sam3Processor(
        model, confidence_threshold=args.confidence,
        preprocess_on_cpu=getattr(args, "preprocess", "mps") == "cpu",
        postprocess_on_cpu=getattr(args, "postprocess", "mps") == "cpu",
        check_finite=getattr(args, "check_finite", False),
    )
    # Cache each class before releasing the text encoder. The image backbone is
    # evaluated once per frame and reused for every class's grounding pass.
    classes = class_definitions(args)
    group_size = prompt_batch_size(len(classes), getattr(args, "batching", "none"))
    prompt_groups = [classes[start:start + group_size] for start in range(0, len(classes), group_size)]
    text_features = [
        processor.encode_text_prompt(group[0]["text"]) if len(group) == 1
        else processor.encode_text_prompts([item["text"] for item in group])
        for group in prompt_groups
    ]
    print(f"Prompt batching {getattr(args, 'batching', 'none')}: "
          f"up to {group_size} of {len(classes)} classes per grounding call", flush=True)
    text_encoder_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.backbone.language_backbone.parameters()
    )
    model.backbone.language_backbone = None
    # The autocast weight cache must not keep the discarded text encoder alive.
    torch.clear_autocast_cache()
    gc.collect()
    release_mps_cache(device, force=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if args.memory_report:
        print(
            f"Released fixed-prompt text encoder: {text_encoder_bytes / 1024**3:.2f} GiB of weights",
            flush=True,
        )
    if args.timing_report:
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize(device)
        print(
            f"Model load and prompt encoding: {time.perf_counter() - load_started:.2f}s",
            flush=True,
        )
    if args.memory_report:
        report_memory("image model load", device)

    manifest_path = args.output / "metadata.jsonl"
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for sample_number, sample in enumerate(samples, 1):
            frame_index = sample["frame_index"]
            stem = f"{frame_index:06d}"
            image_path = args.output / "images" / f"{stem}.jpg"
            mask_path = args.output / "masks" / f"{stem}.png"
            overlay_path = args.output / "overlays" / f"{stem}.jpg"
            inference_started = time.perf_counter()
            frame = None
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
                if not args.no_overlays:
                    frame = cv2.cvtColor(np.asarray(rgb_image), cv2.COLOR_RGB2BGR)
                state = processor.set_image(rgb_image)
                del rgb_image
            if getattr(args, "profile_stages", False):
                if device.type == "mps":
                    torch.mps.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize(device)
            backbone_seconds = time.perf_counter() - inference_started
            height, width = state["original_height"], state["original_width"]
            binary_mask = np.zeros((height, width), dtype=np.uint8)
            scores, class_ids, class_counts, class_masks, preview_masks = [], [], [], {}, []
            instances = [] if getattr(args, "export_format", "masks") != "masks" else None
            for group, features in zip(prompt_groups, text_features):
                if len(group) == 1:
                    results = (processor.set_text_prompt(group[0]["text"], state, text_features=features),)
                else:
                    results = processor.set_text_prompts([item["text"] for item in group], state,
                                                         text_features=features)
                for item, result in zip(group, results):
                    masks = result["masks"]
                    class_mask = (masks[:, 0].any(dim=0).to("cpu").numpy().astype(np.uint8)
                                  if masks.shape[0] else np.zeros((height, width), dtype=np.uint8))
                    binary_mask |= class_mask
                    class_scores = result["scores"].detach().to("cpu").tolist()
                    scores.extend(class_scores)
                    class_ids.extend([item["id"]] * len(class_scores))
                    class_counts.append({"class_id": item["id"], "mask_count": len(class_scores)})
                    if instances is not None:
                        instances.extend(mask_instances(masks[:, 0].detach().to("cpu").numpy(),
                                                        class_scores, item["id"]))
                    if len(classes) > 1:
                        relative = f"masks/class-{item['id']:03d}/{stem}.png"
                        class_path = args.output / relative
                        class_path.parent.mkdir(parents=True, exist_ok=True)
                        write_image(class_path, class_mask * 255)
                        class_masks[str(item["id"])] = relative
                    if not args.no_overlays:
                        preview_masks.append(class_mask)
                    # Retain only image features between prompts, releasing
                    # full-size mask logits before postprocessing the next class.
                    if hasattr(processor, "reset_all_prompts"):
                        processor.reset_all_prompts(state)
                    else:
                        for key in ("masks", "masks_logits", "scores", "boxes", "geometric_prompt"):
                            state.pop(key, None)
                    del masks, class_mask, result
                del results
            if args.timing_report:
                # Empty outputs may not trigger a device-to-host synchronization.
                if device.type == "mps":
                    torch.mps.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize(device)
            inference_seconds = time.perf_counter() - inference_started
            if getattr(args, "drop_empty", False) and not scores:
                # These are generated copies only; leave input photos/videos intact.
                for path in (image_path, mask_path, overlay_path,
                             *(args.output / relative for relative in class_masks.values())):
                    path.unlink(missing_ok=True)
                del state, binary_mask, frame, preview_masks
                release_image_cache(args, sample_number, device)
                print(f"Dropped frame {frame_index}: 0 masks across all classes", flush=True)
                if args.memory_report:
                    report_memory(f"dropped image frame {frame_index}", device)
                continue
            writing_started = time.perf_counter()
            write_image(mask_path, binary_mask * 255)
            if not args.no_overlays:
                preview = overlay_masks(
                    frame,
                    {"out_binary_masks": np.asarray(preview_masks),
                     "out_obj_ids": [item["id"] for item in classes]},
                    args.alpha,
                )
                write_image(overlay_path, preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
                del preview
            row = {
                **sample,
                "image": f"images/{stem}.jpg",
                "mask": f"masks/{stem}.png",
                "overlay": None if args.no_overlays else f"overlays/{stem}.jpg",
                "mask_count": len(scores),
                "scores": scores,
                "class_ids": class_ids,
                "class_counts": class_counts,
                "class_masks": class_masks,
                "mask_fraction": float(binary_mask.mean()),
            }
            if instances is not None:
                row.update(width=width, height=height, instances=instances)
            if args.timing_report:
                row["inference_seconds"] = inference_seconds
                row["backbone_seconds"] = backbone_seconds
                row["grounding_seconds"] = inference_seconds - backbone_seconds
                row["output_seconds"] = time.perf_counter() - writing_started
            message = f"Labeled frame {frame_index}: {len(scores)} masks"
            if args.timing_report:
                message += (
                    f", inference {inference_seconds:.2f}s, "
                    f"output {time.perf_counter() - writing_started:.2f}s"
                )
            del state, binary_mask, frame, preview_masks
            # These tensors have no reference cycles; deleting state releases them
            # without scanning every Python object with gc.collect on every frame.
            release_image_cache(args, sample_number, device)
            if args.timing_report:
                row["frame_seconds"] = time.perf_counter() - inference_started
            manifest.write(json.dumps(row) + "\n")
            manifest.flush()
            print(message, flush=True)
            if args.memory_report:
                report_memory(f"image frame {frame_index}", device)


def extract_image_samples(args):
    """Extract one dataset, retaining each clip's frame numbers and timestamps."""
    sources = image_dataset_sources(args.input)
    samples, source_info = [], []
    action = args.large_image_action
    global_frame_index = 0
    for source in sources:
        reader = FrameReader(source)
        try:
            large_sources = set()
            if reader.paths is not None:
                selected = reader.paths[::args.sample_every]
                if args.max_samples:
                    selected = selected[:args.max_samples - len(samples)]
                large_sources = {path for path in selected if path.stat().st_size >= LARGE_IMAGE_BYTES}
                if large_sources:
                    action = large_image_decision(sorted(large_sources), action)
            folders = ("images", "masks") if args.no_overlays else ("images", "masks", "overlays")
            for name in folders:
                (args.output / name).mkdir(parents=True, exist_ok=True)
            is_video = reader.paths is None
            source_info.append({"source": str(source.resolve()),
                                "fps": reader.fps if is_video else None,
                                "num_frames": reader.num_frames})
            local_frame_index = 0
            decoded_any = False
            while (frame := reader.read()) is not None:
                decoded_any = True
                frame_index = global_frame_index
                global_frame_index += 1
                if local_frame_index % args.sample_every == 0:
                    image_path = args.output / "images" / f"{frame_index:06d}.jpg"
                    write_image(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    compression_quality = None
                    original = source if is_video else reader.paths[local_frame_index]
                    if original in large_sources and action == "compress":
                        compression_quality = compress_image(image_path)
                    samples.append({
                        "frame_index": frame_index,
                        "source": str(original.resolve()),
                        "source_frame_index": local_frame_index,
                        "source_fps": reader.fps if is_video else None,
                        "compression_quality": compression_quality,
                        "time_seconds": round(local_frame_index / reader.fps, 6) if is_video else None,
                    })
                    if args.max_samples and len(samples) >= args.max_samples:
                        break
                local_frame_index += 1
            if not decoded_any:
                raise ValueError(f"input contains no decodable frames: {source}")
        finally:
            reader.close()
        if args.max_samples and len(samples) >= args.max_samples:
            break
    if not samples:
        raise ValueError("input contains no decodable frames")

    # Check actual encoded frames too: video frames can exceed the upload limit.
    image_paths = [args.output / "images" / f"{sample['frame_index']:06d}.jpg" for sample in samples]
    decision = large_image_decision(image_paths, action)
    if decision == "compress":
        for sample, path in zip(samples, image_paths):
            if path.stat().st_size >= LARGE_IMAGE_BYTES:
                sample["compression_quality"] = compress_image(path)
    return samples, source_info


def export_image_dataset(args):
    export_started = time.perf_counter()
    samples, source_info = extract_image_samples(args)

    # Resolve/download model weights only after the parent has handled size decisions.
    from sam3.model.device import default_device
    from sam3.model_builder import download_ckpt_from_hf
    import torch
    device = default_device() if args.device == "auto" else torch.device(args.device)
    batch_size = args.restart_every
    if batch_size is None:
        batch_size = 3 if device.type == "mps" else 0
    checkpoint_path = str(args.checkpoint) if args.checkpoint else download_ckpt_from_hf()
    args.device = device.type

    config = {
        "source": str(args.input.resolve()),
        "text_prompt": args.text,
        "confidence_threshold": args.confidence,
        "inference_size": args.inference_size,
        "preset": args.preset,
        "inference_options": {key: getattr(args, key) for key in DEFAULTS},
        "sample_every": args.sample_every,
        "source_fps": source_info[0]["fps"] if args.input.is_file() else None,
        "sources": source_info,
        "num_samples": len(samples),
        "num_extracted": len(samples),
        "num_dropped_empty": 0,
        "drop_empty": args.drop_empty,
        "restart_every": batch_size,
        "overlay_previews": not args.no_overlays,
        "image_format": "JPEG",
        "mask_format": "8-bit PNG, 0 background and 255 foreground",
        "export_format": args.export_format,
        "export_task": args.export_task,
        "class_name": args.class_name or args.text,
        "classes": class_definitions(args),
        "batching": args.batching,
        "prompt_batch_size": prompt_batch_size(len(class_definitions(args)), args.batching),
        "ndjson_image_url_base": args.ndjson_image_url_base,
        "val_fraction": args.val_fraction,
        "chunk_size": args.chunk_size,
        "chunk_max_mb": args.chunk_max_mb,
        "large_image_action": args.large_image_action,
    }
    (args.output / "dataset.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Extracted {len(samples)} frames to {args.output / 'images'}", flush=True)

    context = mp.get_context("spawn")
    batch_size = batch_size or len(samples)
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        print(
            f"Labeling samples {start + 1}-{start + len(batch)} of {len(samples)}",
            flush=True,
        )
        worker = context.Process(
            target=infer_image_batch,
            args=(args, batch, checkpoint_path),
        )
        worker.start()
        try:
            worker.join()
        except KeyboardInterrupt:
            worker.terminate()
            worker.join()
            raise
        if worker.exitcode != 0:
            raise RuntimeError(
                f"image inference failed for samples {start + 1}-{start + len(batch)} "
                f"(exit code {worker.exitcode})"
            )
    retained = len(samples)
    if args.drop_empty:
        with (args.output / "metadata.jsonl").open(encoding="utf-8") as manifest:
            retained = sum(bool(line.strip()) for line in manifest)
        config.update(num_samples=retained, num_dropped_empty=len(samples) - retained)
        (args.output / "dataset.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"Discarded {len(samples) - retained} images with 0 masks; kept {retained}", flush=True)
    if args.export_format != "masks" and retained:
        export_training_dataset(args)
    elif args.export_format != "masks":
        print("No images with masks remain; training archive skipped.", flush=True)
    print(f"Exported {retained} image-mask pairs to {args.output}")
    if args.timing_report:
        elapsed = time.perf_counter() - export_started
        print(
            f"Total export: {elapsed:.2f}s, {elapsed / len(samples):.2f}s/sample "
            f"including setup, extraction, and worker restarts",
            flush=True,
        )


def export_video(args):
    if args.image_mode:
        return export_image_dataset(args)
    if args.restart_every:
        return export_video_in_segments(args)
    # Import after CLI validation so --help works without model dependencies.
    from sam3.model_builder import build_sam3_video_predictor
    from sam3.model.device import default_device
    import torch

    reader = FrameReader(args.input)
    predictor = None
    writer = None
    session_id = None
    try:
        frame = reader.read()
        if frame is None:
            raise ValueError("input contains no decodable frames")
        if (
            args.fps is not None
            and args.input.is_file()
            and not args.no_audio
            and abs(args.fps - reader.fps) > 0.001
        ):
            raise ValueError("changing video FPS requires --no-audio to avoid audio drift")
        fps = args.fps or reader.fps
        device = default_device() if args.device == "auto" else torch.device(args.device)
        offload_video_to_cpu = (
            args.offload_video_to_cpu
            if args.offload_video_to_cpu is not None
            else device.type == "mps"
        )
        offload_state_to_cpu = (
            args.offload_state_to_cpu
            if args.offload_state_to_cpu is not None
            else device.type == "mps"
        )
        predictor = build_sam3_video_predictor(
            checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
            device=device,
            video_loader_type="cv2_lazy" if args.input.is_file() else "cv2",
        )
        if device.type == "mps":
            def after_inference_frame(frame_index, frame_output):
                should_report = args.memory_report and (
                    frame_index < 10 or (frame_index + 1) % 30 == 0
                )
                if should_report:
                    print(
                        f"inference frame {frame_index + 1}: "
                        f"{len(frame_output['obj_id_to_mask'])} object masks",
                        file=sys.stderr,
                        flush=True,
                    )
                    report_memory(
                        f"inference frame {frame_index + 1} before release", device
                    )
                release_mps_cache(device)
                if should_report:
                    report_memory(
                        f"inference frame {frame_index + 1} after release", device
                    )

            predictor.model.frame_inference_hook = after_inference_frame
        if args.memory_report:
            report_memory("model load", device)
        response = predictor.handle_request(
            dict(
                type="start_session",
                resource_path=str(args.input),
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu,
            )
        )
        session_id = response["session_id"]
        if args.memory_report:
            report_memory("session start", device)
        predictor.handle_request(
            dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=args.prompt_frame,
                text=args.text,
            )
        )
        if args.memory_report:
            report_memory("prompt", device)
        release_mps_cache(device, force=True)
        if args.memory_report:
            report_memory("prompt after cache release", device)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        audio_source = args.input if args.input.is_file() and not args.no_audio else None
        writer = VideoWriter(
            args.output,
            frame.shape[1],
            frame.shape[0],
            fps,
            audio_source,
            overwrite=args.overwrite,
        )
        next_frame = 0
        for result in predictor.handle_stream_request(
            dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction="forward",
                start_frame_index=0,
                evict_cached_frame_outputs=True,
            )
        ):
            target = result["frame_index"]
            if target < next_frame:
                raise ValueError("predictor returned frames out of order")
            while next_frame < target:
                if frame is None:
                    raise ValueError("predictor returned a frame beyond the input")
                writer.write(frame)
                next_frame += 1
                frame = reader.read()
            if frame is None:
                raise ValueError("predictor returned a frame beyond the input")
            writer.write(overlay_masks(frame, result["outputs"], args.alpha))
            next_frame += 1
            frame = reader.read()
            if next_frame % 5 == 0:
                release_mps_cache(device)
            if args.memory_report and device.type != "mps" and (
                next_frame <= 10 or next_frame % 30 == 0
            ):
                report_memory(f"frame {next_frame}", device)
        while frame is not None:
            writer.write(frame)
            next_frame += 1
            frame = reader.read()
        writer.close()
        writer = None
        print(f"Exported {next_frame} frames to {args.output}")
    finally:
        if writer is not None:
            writer.abort()
        reader.close()
        if predictor is not None:
            try:
                if session_id is not None:
                    predictor.handle_request(
                        dict(type="close_session", session_id=session_id)
                    )
            finally:
                predictor.shutdown()


def main(argv=None):
    try:
        export_video(parse_args(argv))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

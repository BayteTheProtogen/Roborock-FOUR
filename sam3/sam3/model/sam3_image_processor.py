# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe
from typing import Dict, List

import numpy as np
import PIL
import torch
from sam3.model import box_ops
from sam3.model.data_misc import FindStage, interpolate
from torchvision.transforms import v2


class Sam3Processor:
    """ """

    def __init__(self, model, resolution=None, device=None, confidence_threshold=0.5,
                 preprocess_on_cpu=False, postprocess_on_cpu=False, check_finite=False):
        self.model = model
        model_resolution = getattr(model, "image_size", 1008)
        resolution = model_resolution if resolution is None else resolution
        if hasattr(model, "image_size") and resolution != model_resolution:
            raise ValueError("processor resolution must match the model's image_size")
        self.resolution = resolution
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.confidence_threshold = confidence_threshold
        self.preprocess_on_cpu = preprocess_on_cpu
        self.postprocess_on_cpu = postprocess_on_cpu
        self.check_finite = check_finite

        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=self.device, dtype=torch.long),
            text_ids=torch.tensor([0], device=self.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    @torch.inference_mode()
    def set_image(self, image, state=None):
        """Sets the image on which we want to do predictions."""
        if state is None:
            state = {}

        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image or a tensor")

        image = self._prepare_image(image).unsqueeze(0)

        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = self.model.backbone.forward_image(image)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_image_batch(self, images: List[np.ndarray], state=None):
        """Sets the image batch on which we want to do predictions."""
        if state is None:
            state = {}

        if not isinstance(images, list):
            raise ValueError("Images must be a list of PIL images or tensors")
        assert len(images) > 0, "Images list must not be empty"
        assert isinstance(images[0], PIL.Image.Image), (
            "Images must be a list of PIL images"
        )

        state["original_heights"] = [image.height for image in images]
        state["original_widths"] = [image.width for image in images]

        images = [self._prepare_image(image) for image in images]
        # pyrefly: ignore [bad-argument-type, bad-assignment]
        images = torch.stack(images, dim=0)
        state["backbone_out"] = self.model.backbone.forward_image(images)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    def _prepare_image(self, image):
        image = v2.functional.to_image(image)
        if self.preprocess_on_cpu:
            return self.transform(image.cpu()).to(self.device)
        return self.transform(image.to(self.device))

    @torch.inference_mode()
    def encode_text_prompt(self, prompt: str):
        """Encode a fixed prompt once for reuse across images with this model.

        Re-encode after changing model weights, device, or inference precision.
        """
        return self.model.backbone.forward_text([prompt], device=self.device)

    @torch.inference_mode()
    def encode_text_prompts(self, prompts: List[str]):
        """Encode a prompt group once for repeated batched grounding."""
        if not prompts:
            raise ValueError("prompts must not be empty")
        return self.model.backbone.forward_text(prompts, device=self.device)

    @torch.inference_mode()
    def set_text_prompts(self, prompts: List[str], state: Dict, text_features=None):
        """Ground all prompts in one call against one cached image.

        Returns an iterator of per-prompt results in input order. Full-resolution
        masks are postprocessed one prompt at a time to bound retained memory.
        Cached text_features must encode this exact prompt group. Image state and
        the single-prompt find_stage are left unchanged.
        """
        if not prompts:
            raise ValueError("prompts must not be empty")
        if "backbone_out" not in state or "original_height" not in state:
            raise ValueError("You must call set_image before set_text_prompts")
        count = len(prompts)
        text_outputs = self.encode_text_prompts(prompts) if text_features is None else text_features
        if text_outputs["language_features"].shape[1] != count:
            raise ValueError("cached text features must match the prompt group size")
        backbone_out = {**state["backbone_out"], **text_outputs}
        find_stage = FindStage(
            img_ids=torch.zeros(count, device=self.device, dtype=torch.long),
            text_ids=torch.arange(count, device=self.device, dtype=torch.long),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )
        outputs = self.model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_stage,
            geometric_prompt=self.model._get_dummy_prompt(num_prompts=count),
            find_target=None,
        )
        height, width = state["original_height"], state["original_width"]

        def results():
            for index in range(count):
                selected = {key: outputs[key][index:index + 1] for key in
                            ("pred_boxes", "pred_logits", "pred_masks", "presence_logit_dec")}
                yield self._postprocess_grounding(
                    selected, {"original_height": height, "original_width": width}
                )

        return results()

    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: Dict, text_features=None):
        """Set a prompt, optionally reusing features from encode_text_prompt.

        Cached features must encode this prompt with the same model and precision.
        """

        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        text_outputs = (
            self.encode_text_prompt(prompt) if text_features is None else text_features
        )
        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_geometric_prompt(self, box: List, label: bool, state: Dict):
        """Adds a box prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        The label is True for a positive box, False for a negative box.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        boxes = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([label], device=self.device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)

        return self._forward_grounding(state)

    def reset_all_prompts(self, state: Dict):
        """Removes all the prompts and results"""
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]
        for key in keys_to_del:
            if key in state:
                del state[key]

    @torch.inference_mode()
    def set_confidence_threshold(self, threshold: float, state=None):
        """Sets the confidence threshold for the masks"""
        self.confidence_threshold = threshold
        if state is not None and "boxes" in state:
            # we need to filter the boxes again
            # In principle we could do this more efficiently since we would only need
            # to rerun the heads. But this is simpler and not too inefficient
            return self._forward_grounding(state)
        return state

    @torch.inference_mode()
    def _forward_grounding(self, state: Dict):
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )
        return self._postprocess_grounding(outputs, state)

    @torch.inference_mode()
    def _postprocess_grounding(self, outputs: Dict, state: Dict):
        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        if self.check_finite:
            for name in ("pred_boxes", "pred_logits", "pred_masks", "presence_logit_dec"):
                if not torch.isfinite(outputs[name]).all().item():
                    raise RuntimeError(f"non-finite model output: {name}")
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

        if self.postprocess_on_cpu:
            out_masks = out_masks.float().cpu()
        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state

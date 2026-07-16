#!/usr/bin/env python3
"""Verify reference GenLIP and OpenCLIP produce the same image features.

This example loads one exported GenLIP ``hf_ckpt`` twice: once with the original
implementation and once through OpenCLIP's checkpoint converter. It feeds both models
the exact same deterministic image tensor and compares both per-patch encoder features
and the final mean-pooled image feature.

Example:

    python example.py /path/to/global_step_10000/hf_ckpt
"""

import argparse
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_model as load_safetensors_model

import open_clip
from open_clip.naflex_genlip_model import build_image_position_ids, build_patch_attn_mask


REFERENCE_MODEL_NAMES = {
    (1152, 24): "naflexgenlip_ref_l16",
    (1152, 27): "naflexgenlip_ref_so16",
    (1536, 40): "naflexgenlip_ref_g16",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path, help="GenLIP-exported hf_ckpt directory.")
    parser.add_argument(
        "--genlip-repo",
        type=Path,
        default=Path("/p/project1/laionize/cherti1/genlip"),
        help="Path to the original GenLIP repository.",
    )
    parser.add_argument(
        "--open-clip-model",
        choices=sorted(REFERENCE_MODEL_NAMES.values()),
        default=None,
        help="Override automatic L16/SO16/G16 architecture selection.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
        help="Use fp32 for strict conversion parity; bf16 may expose Conv2d/Linear kernel drift.",
    )
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    return parser.parse_args()


def find_single_weights_file(checkpoint_dir: Path) -> Path:
    preferred = (
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model.pth",
        "model.bin",
        "model.pt",
    )
    for filename in preferred:
        path = checkpoint_dir / filename
        if path.is_file():
            return path

    candidates = sorted(checkpoint_dir.glob("*.safetensors"))
    candidates += sorted(checkpoint_dir.glob("*.bin"))
    candidates += sorted(checkpoint_dir.glob("*.pt"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No model weights found in {checkpoint_dir}.")
    raise RuntimeError(
        f"Found multiple weight shards in {checkpoint_dir}. The example expects the single-file "
        "hf_ckpt export; consolidate the checkpoint first."
    )


def import_reference_genlip(genlip_repo: Path):
    """Import GenLIP modeling without executing unrelated VeOmni package initializers.

    The reference checkout's aggregate ``veomni.data`` initializer currently imports optional
    dataset builders that are absent in some revisions. The modeling file itself only needs
    specific submodules, so namespace packages keep this parity example independent of those
    training-data imports while still executing the original model implementation.
    """
    package_dirs = {
        "veomni": genlip_repo / "veomni",
        "veomni.models": genlip_repo / "veomni/models",
        "veomni.models.transformers": genlip_repo / "veomni/models/transformers",
        "veomni.models.transformers.genlip": genlip_repo / "veomni/models/transformers/genlip",
        "veomni.data": genlip_repo / "veomni/data",
        "veomni.distributed": genlip_repo / "veomni/distributed",
        "veomni.utils": genlip_repo / "veomni/utils",
    }
    for name, directory in package_dirs.items():
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(directory)]
            package.__package__ = name
            sys.modules[name] = package

    from veomni.models.transformers.genlip.genlip_modeling import GenLIPConfig, GenLIPModel
    return GenLIPConfig, GenLIPModel


def load_reference_genlip(
        checkpoint_dir: Path,
        weights_path: Path,
        genlip_repo: Path,
        device: torch.device,
        dtype: torch.dtype,
):
    if not genlip_repo.is_dir():
        raise FileNotFoundError(f"GenLIP repository not found: {genlip_repo}")
    GenLIPConfig, GenLIPModel = import_reference_genlip(genlip_repo)

    config = GenLIPConfig.from_pretrained(checkpoint_dir)
    # The public GenLIP Hub exports omit ``text_embed_dim`` from config.json even though
    # the language embeddings and vision/text projections retain the decoder width. Recover
    # the authoritative value from the checkpoint so the original implementation can load its
    # own export without treating the language weights as mismatched.
    if weights_path.suffix == ".safetensors":
        with safe_open(weights_path, framework="pt") as checkpoint:
            if "embeddings.weight" in checkpoint.keys():
                config.text_embed_dim = checkpoint.get_slice("embeddings.weight").get_shape()[1]
    # The parity path uses eager PyTorch SDPA and does not need the optional Liger loss kernel.
    config.use_liger_kernel = False
    config._attn_implementation = "sdpa"
    model = GenLIPModel.from_pretrained(
        checkpoint_dir,
        config=config,
        torch_dtype=dtype,
        local_files_only=True,
    )
    # Recent Transformers releases call GenLIPModel.initialize_weights() while finalizing
    # from_pretrained(), but GenLIP defines that method as a full-model reinitializer. Restore
    # the exported values directly after construction so this remains a checkpoint parity test.
    if weights_path.suffix == ".safetensors":
        load_safetensors_model(model, weights_path, strict=False)
    return model.to(device).eval(), config


def make_inputs(image_size: int, patch_size: int, device: torch.device, dtype: torch.dtype):
    # Zero in GenLIP's [-1, 1] normalized image space is a valid mid-gray image. It also makes
    # Conv2d and the converted Linear patch projection bit-identical, preventing their different
    # CUDA accumulation kernels from seeding drift in this unusually sensitive 27-layer checkpoint.
    pixels = torch.zeros(1, 3, image_size, image_size, device=device, dtype=dtype)
    patches = F.unfold(pixels.float(), kernel_size=patch_size, stride=patch_size)
    patches = patches.transpose(1, 2).to(dtype=dtype)
    grid_size = image_size // patch_size
    coords = torch.tensor(
        [(h, w) for h in range(grid_size) for w in range(grid_size)],
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)
    valid = torch.ones(1, grid_size * grid_size, device=device, dtype=torch.bool)
    return pixels, {"patches": patches, "patch_coord": coords, "patch_valid": valid}


@torch.no_grad()
def encode_reference(model, pixels: torch.Tensor, patch_coord: torch.Tensor):
    x = model.vision_embeddings(pixels)
    valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    position_ids = build_image_position_ids(patch_coord, valid)
    position_embeddings = model.rotary_emb(x, position_ids)
    # The reference SDPA implementation has an extra leading grouping dimension and was
    # trained with batch size one packed streams. A [1, 1, N, N] full-image mask broadcasts
    # to that layout and is equivalent to OpenCLIP's image-only key mask.
    attention_mask = torch.ones(
        1, 1, x.shape[1], x.shape[1], dtype=torch.bool, device=x.device)
    x, _ = model.visual(
        x,
        cu_seqlens=None,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
    )
    tokens = model.proj(model.ln_post(x))
    return tokens, tokens.mean(dim=1)


@torch.no_grad()
def encode_openclip(model, image):
    visual = model.visual
    x = visual.patch_embed(image["patches"])
    position_ids = build_image_position_ids(image["patch_coord"], image["patch_valid"])
    cos, sin = visual.rotary(x, position_ids)
    x = visual.trunk(x, build_patch_attn_mask(image["patch_valid"]), cos, sin)
    tokens = visual.proj(x)
    pooled = model.encode_image(image, normalize=False)
    return tokens, pooled


def main():
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    weights_path = find_single_weights_file(checkpoint_dir)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32

    reference, reference_cfg = load_reference_genlip(
        checkpoint_dir, weights_path, args.genlip_repo.resolve(), device, dtype)
    architecture = (reference_cfg.hidden_size, reference_cfg.num_hidden_layers)
    model_name = args.open_clip_model or REFERENCE_MODEL_NAMES.get(architecture)
    if model_name is None:
        raise ValueError(
            f"No reference OpenCLIP config for hidden_size/depth={architecture}; "
            "pass --open-clip-model if this is a supported variant."
        )

    converted = open_clip.create_model(
        model_name,
        pretrained=str(weights_path),
        precision=args.precision,
        device=device,
    ).eval()

    image_size = int(converted.vision_cfg.image_size)
    patch_size = int(converted.vision_cfg.patch_size)
    pixels, image = make_inputs(image_size, patch_size, device, dtype)
    reference_tokens, reference_pooled = encode_reference(reference, pixels, image["patch_coord"])
    converted_tokens, converted_pooled = encode_openclip(converted, image)

    # The reference implementation gives SDPA an extra singleton grouping dimension,
    # which can select a slightly different reduction kernel. Account for normal floating-point
    # accumulation differences while still making conversion mistakes fail decisively.
    atol = args.atol if args.atol is not None else (2e-2 if dtype == torch.bfloat16 else 5e-3)
    rtol = args.rtol if args.rtol is not None else (2e-2 if dtype == torch.bfloat16 else 5e-3)
    token_max_abs = (reference_tokens - converted_tokens).abs().max().item()
    pooled_max_abs = (reference_pooled - converted_pooled).abs().max().item()
    cosine = F.cosine_similarity(reference_pooled.float(), converted_pooled.float()).item()

    print(f"checkpoint:       {checkpoint_dir}")
    print(f"OpenCLIP model:   {model_name}")
    print(f"dtype/device:     {dtype} / {device}")
    print(f"token max |diff|: {token_max_abs:.8g}")
    print(f"pool max |diff|:  {pooled_max_abs:.8g}")
    print(f"pool cosine:      {cosine:.10f}")
    torch.testing.assert_close(converted_tokens, reference_tokens, atol=atol, rtol=rtol)
    torch.testing.assert_close(converted_pooled, reference_pooled, atol=atol, rtol=rtol)
    print("PASS: original GenLIP and converted OpenCLIP image features match.")


if __name__ == "__main__":
    main()

import json

import torch
import torch.nn.functional as F

import open_clip
from open_clip.convert import (
    convert_genlip_state_dict,
    convert_vision_state_dict,
    is_genlip_state_dict,
)
from open_clip.model import CLIPVisionCfg, _build_vision_tower
from open_clip.naflex_genlip_model import NaFlexGenLip
from open_clip.tokenizer import HFTokenizer


def _tiny_genlip_cfg():
    return {
        "embed_dim": 32,
        "vision_cfg": {
            "image_size": 32,
            "patch_size": 16,
            "in_chans": 3,
            "proj_bias": True,
        },
        "text_cfg": {
            "vocab_size": 31,
            "context_length": 12,
            "pad_id": 30,
            "bos_id": None,
            "eos_id": 29,
        },
        "genlip_cfg": {
            "width": 64,
            "depth": 2,
            "num_heads": 4,
            "intermediate_size": 96,
            "text_embed_dim": 32,
            "mrope_section": (2, 3, 3),
            "drop_path_rate": 0.0,
            "attention_bias": True,
            "mlp_bias": True,
        },
    }


def _reference_state_from_openclip(model):
    """Build the state-dict layout emitted by the reference GenLIP implementation."""
    state = model.state_dict()
    ref = {}
    patch_weight = state["patch_embed.proj.weight"]
    ref["vision_embeddings.patch_embedding.weight"] = patch_weight.reshape(
        patch_weight.shape[0], 3, 16, 16)
    ref["vision_embeddings.patch_embedding.bias"] = state["patch_embed.proj.bias"]
    for key, value in state.items():
        if key.startswith("trunk.layers."):
            ref["visual.layers." + key.removeprefix("trunk.layers.")] = value
        elif key.startswith("trunk.ln_post."):
            ref["ln_post." + key.removeprefix("trunk.ln_post.")] = value
    ref["embeddings.weight"] = state["text_embed.weight"]
    for prefix in ("in_proj.", "lm_head."):
        for key, value in state.items():
            if key.startswith(prefix):
                ref[key] = value
    for key, value in state.items():
        if key.startswith("out_proj."):
            ref["proj." + key.removeprefix("out_proj.")] = value
    return ref


def test_convert_reference_genlip_full_model_strict():
    model = NaFlexGenLip(**_tiny_genlip_cfg())
    reference = _reference_state_from_openclip(model)
    assert is_genlip_state_dict(reference)

    converted = convert_genlip_state_dict(model, reference)
    assert set(converted) == set(model.state_dict())
    model.load_state_dict(converted, strict=True)
    torch.testing.assert_close(
        model.visual.proj.weight,
        reference["proj.weight"],
    )


def test_convert_reference_genlip_vision_tower_strict():
    full_model = NaFlexGenLip(**_tiny_genlip_cfg())
    reference = _reference_state_from_openclip(full_model)
    cfg = _tiny_genlip_cfg()
    vision_cfg = CLIPVisionCfg(
        image_size=32,
        patch_size=16,
        in_chans=3,
        proj_bias=True,
        pool_type="avg",
        genlip_cfg=cfg["genlip_cfg"],
    )
    tower = _build_vision_tower(32, vision_cfg)

    converted, source_format = convert_vision_state_dict(tower, reference)
    assert source_format == "genlip"
    tower.load_state_dict(converted, strict=True)
    torch.testing.assert_close(tower.proj.weight, reference["proj.weight"])


def test_genlip_conv_patch_projection_matches_naflex_linear():
    model = NaFlexGenLip(**_tiny_genlip_cfg()).eval()
    reference = _reference_state_from_openclip(model)
    image = torch.randn(2, 3, 32, 32)

    conv = F.conv2d(
        image,
        reference["vision_embeddings.patch_embedding.weight"],
        reference["vision_embeddings.patch_embedding.bias"],
        stride=16,
    ).flatten(2).transpose(1, 2)
    patches = F.unfold(image, kernel_size=16, stride=16).transpose(1, 2)
    linear = model.patch_embed.proj(patches)
    torch.testing.assert_close(linear, conv)


def test_vision_conversion_allows_new_projection_dimension():
    full_model = NaFlexGenLip(**_tiny_genlip_cfg())
    reference = _reference_state_from_openclip(full_model)
    cfg = _tiny_genlip_cfg()
    tower = _build_vision_tower(
        48,
        CLIPVisionCfg(
            image_size=32,
            patch_size=16,
            pool_type="avg",
            genlip_cfg=cfg["genlip_cfg"],
        ),
    )
    converted, _ = convert_vision_state_dict(tower, reference)
    incompatible = tower.load_state_dict(converted, strict=False)
    assert incompatible.missing_keys == ["proj.weight", "proj.bias"]
    assert incompatible.unexpected_keys == []


def test_factory_loads_reference_genlip_full_checkpoint(tmp_path):
    cfg = _tiny_genlip_cfg()
    source_model = NaFlexGenLip(**cfg)
    checkpoint = tmp_path / "reference_genlip.pt"
    torch.save(_reference_state_from_openclip(source_model), checkpoint)
    config_path = tmp_path / "test_ref_genlip_full.json"
    config_path.write_text(json.dumps(cfg))
    open_clip.add_model_config(config_path)

    loaded = open_clip.create_model(config_path.stem, pretrained=str(checkpoint))
    torch.testing.assert_close(loaded.patch_embed.proj.weight, source_model.patch_embed.proj.weight)
    torch.testing.assert_close(loaded.out_proj.weight, source_model.out_proj.weight)
    torch.testing.assert_close(loaded.lm_head.weight, source_model.lm_head.weight)


def test_factory_loads_reference_genlip_as_clip_vision_tower(tmp_path):
    genlip_cfg = _tiny_genlip_cfg()
    source_model = NaFlexGenLip(**genlip_cfg)
    checkpoint = tmp_path / "reference_genlip_vision.pt"
    torch.save(_reference_state_from_openclip(source_model), checkpoint)
    clip_cfg = {
        "embed_dim": 32,
        "custom_text": True,
        "vision_cfg": {
            **genlip_cfg["vision_cfg"],
            "pool_type": "avg",
            "genlip_cfg": genlip_cfg["genlip_cfg"],
        },
        "text_cfg": {
            "context_length": 8,
            "vocab_size": 64,
            "width": 32,
            "heads": 4,
            "layers": 2,
        },
    }
    config_path = tmp_path / "test_ref_genlip_clip.json"
    config_path.write_text(json.dumps(clip_cfg))
    open_clip.add_model_config(config_path)

    loaded = open_clip.create_model(
        config_path.stem,
        pretrained_image_path=str(checkpoint),
        pretrained_text=False,
    )
    torch.testing.assert_close(loaded.visual.patch_embed.proj.weight, source_model.patch_embed.proj.weight)
    torch.testing.assert_close(loaded.visual.proj.weight, source_model.out_proj.weight)


def test_genlip_hf_tokenizer_appends_reference_suffix(monkeypatch):
    class DummyTokenizer:
        pad_token_id = 9
        eos_token_id = 8
        sep_token_id = None
        bos_token_id = None
        cls_token_id = None
        padding_side = "left"

        def __len__(self):
            return 10

        def encode(self, text, add_special_tokens=False):
            assert text == "<|im_end|>\n"
            return [7, 8]

        def __call__(self, texts, **kwargs):
            return {"input_ids": [[1, 2, 3] for _ in texts]}

    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )
    tokenizer = HFTokenizer(
        "dummy",
        context_length=6,
        tokenizer_mode="genlip",
        clean="identity",
    )
    variable = tokenizer(["caption"], pad=False)
    assert variable[0].tolist() == [1, 2, 3, 7, 8]
    padded, valid = tokenizer(["caption"], pad=True, output_mask=True)
    assert padded.tolist() == [[1, 2, 3, 7, 8, 9]]
    assert valid.tolist() == [[True, True, True, True, True, False]]

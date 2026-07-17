import json

import pytest
import torch
import torch.nn.functional as F

import open_clip
from open_clip.convert import (
    convert_genlip_state_dict,
    convert_vision_state_dict,
    is_genlip_state_dict,
)
from open_clip.model import CLIPVisionCfg, _build_vision_tower
from open_clip.naflex_genlip_model import GenLip, GenLipVisualAdapter, NaFlexGenLip
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


def test_fixed_genlip_full_model_loads_reference_checkpoint_and_matches_naflex():
    naflex = NaFlexGenLip(**_tiny_genlip_cfg()).eval()
    reference = _reference_state_from_openclip(naflex)
    fixed_cfg = _tiny_genlip_cfg()
    fixed_cfg["vision_cfg"]["genlip_naflex"] = False
    fixed = GenLip(**fixed_cfg).eval()

    fixed.load_state_dict(convert_genlip_state_dict(fixed, reference), strict=True)
    naflex.load_state_dict(convert_genlip_state_dict(naflex, reference), strict=True)

    image = torch.randn(2, 3, 32, 32)
    patches = F.unfold(image, kernel_size=16, stride=16).transpose(1, 2)
    coords = torch.tensor([(0, 0), (0, 1), (1, 0), (1, 1)]).expand(2, -1, -1)
    valid = torch.ones(2, 4, dtype=torch.bool)
    text = torch.tensor([[1, 2, 29], [3, 4, 29]])

    fixed_out = fixed(image, text)["logits"]
    naflex_out = naflex(
        {"patches": patches, "patch_coord": coords, "patch_valid": valid}, text)["logits"]
    torch.testing.assert_close(fixed_out, naflex_out, atol=2e-5, rtol=2e-5)


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


def test_factory_loads_reference_genlip_fixed_full_checkpoint(tmp_path):
    source_model = NaFlexGenLip(**_tiny_genlip_cfg())
    checkpoint = tmp_path / "reference_genlip_fixed_full.pt"
    torch.save(_reference_state_from_openclip(source_model), checkpoint)
    cfg = _tiny_genlip_cfg()
    cfg["vision_cfg"]["genlip_naflex"] = False
    config_path = tmp_path / "test_ref_genlip_fixed_full.json"
    config_path.write_text(json.dumps(cfg))
    open_clip.add_model_config(config_path)

    loaded = open_clip.create_model(config_path.stem, pretrained=str(checkpoint)).eval()
    assert isinstance(loaded, GenLip)
    torch.testing.assert_close(
        loaded.patch_embed.proj.weight,
        _reference_state_from_openclip(source_model)["vision_embeddings.patch_embedding.weight"],
    )
    text = torch.tensor([[1, 2, 29]])
    assert loaded(torch.randn(1, 3, 32, 32), text)["logits"].shape == (1, 7, 31)


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

    # CustomTextCLIP delegates --grad-checkpoint to its vision tower.
    loaded.set_grad_checkpointing(True, impl="inline")
    assert loaded.visual.trunk.grad_checkpointing is True


def test_factory_loads_reference_genlip_into_fixed_clip_vision_tower(tmp_path):
    genlip_cfg = _tiny_genlip_cfg()
    source_model = NaFlexGenLip(**genlip_cfg)
    checkpoint = tmp_path / "reference_genlip_fixed_vision.pt"
    torch.save(_reference_state_from_openclip(source_model), checkpoint)
    clip_cfg = {
        "embed_dim": 32,
        "custom_text": True,
        "vision_cfg": {
            **genlip_cfg["vision_cfg"],
            "pool_type": "avg",
            "genlip_naflex": False,
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
    config_path = tmp_path / "test_ref_genlip_fixed_clip.json"
    config_path.write_text(json.dumps(clip_cfg))
    open_clip.add_model_config(config_path)

    loaded = open_clip.create_model(
        config_path.stem,
        pretrained_image_path=str(checkpoint),
        pretrained_text=False,
    ).eval()
    assert isinstance(loaded.visual, GenLipVisualAdapter)
    assert loaded.visual.patch_embed.proj.weight.ndim == 4
    reference_weight = _reference_state_from_openclip(source_model)["vision_embeddings.patch_embedding.weight"]
    torch.testing.assert_close(loaded.visual.patch_embed.proj.weight, reference_weight)
    assert loaded.encode_image(torch.randn(2, 3, 32, 32)).shape == (2, 32)
    loaded.lock_image_tower(unlocked_groups=0, freeze_bn_stats=True)
    assert all(not p.requires_grad for p in loaded.visual.parameters())
    loaded.set_grad_checkpointing(False, impl="inline")
    assert loaded.visual.trunk.grad_checkpointing is False


@pytest.mark.parametrize("genlip_naflex", [False, True])
def test_genlip_visual_layer_groups_and_reentrant_lock(genlip_naflex):
    cfg = _tiny_genlip_cfg()
    tower = _build_vision_tower(
        32,
        CLIPVisionCfg(
            image_size=32,
            patch_size=16,
            in_chans=3,
            proj_bias=True,
            pool_type="avg",
            genlip_naflex=genlip_naflex,
            genlip_cfg=cfg["genlip_cfg"],
        ),
    )
    groups = tower.layer_groups()
    assert [name for name, _ in groups] == ["embeddings", "layer.0", "layer.1", "proj"]
    grouped = [p for _, members in groups for member in members for p in member.parameters()]
    assert {id(p) for p in grouped} == {id(p) for p in tower.parameters()}
    assert len(grouped) == len({id(p) for p in grouped})

    tower.lock(unlocked_groups=0, freeze_bn_stats=True)
    assert all(not p.requires_grad for p in tower.parameters())

    tower.lock(unlocked_groups=2)
    assert all(not p.requires_grad for p in tower.patch_embed.parameters())
    assert all(not p.requires_grad for p in tower.trunk.layers[0].parameters())
    assert any(p.requires_grad for p in tower.trunk.layers[-1].parameters())
    assert all(p.requires_grad for p in tower.trunk.ln_post.parameters())
    assert all(p.requires_grad for p in tower.proj.parameters())

    tower.lock(unlocked_groups=0)
    assert all(not p.requires_grad for p in tower.parameters())


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

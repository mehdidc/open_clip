import pytest

from open_clip_train.params import parse_args


def test_opt_kwargs_parse_timm_style_key_values():
    args = parse_args([
        "--opt-kwargs",
        "foreach=False",
        "amsgrad=True",
        "max_grad_norm=1.0",
        "mode=fast",
    ])

    assert args.opt_kwargs == {
        "foreach": False,
        "amsgrad": True,
        "max_grad_norm": 1.0,
        "mode": "fast",
    }


def test_val_retrieval_chunk_size_parse():
    args = parse_args(["--val-retrieval-chunk-size", "128"])

    assert args.val_retrieval_chunk_size == 128


def test_val_retrieval_precision_parse():
    args = parse_args(["--val-retrieval-precision", "model"])

    assert args.val_retrieval_precision == "model"


def test_caption_loss_options_parse():
    args = parse_args([
        "--caption-z-loss-weight", "1e-4",
        "--caption-loss-compute-dtype", "model",
        "--caption-loss-chunk-size", "512",
    ])

    assert args.caption_z_loss_weight == 1e-4
    assert args.caption_loss_compute_dtype == "model"
    assert args.caption_loss_chunk_size == 512


@pytest.mark.parametrize("option", [
    ["--caption-z-loss-weight=-1e-4"],
    ["--caption-loss-chunk-size", "0"],
])
def test_caption_loss_options_reject_invalid_values(option):
    with pytest.raises(ValueError):
        parse_args(option)


@pytest.mark.parametrize("model_name,is_generative", [
    ("genlip_ref_so16", True),
    ("clip_genlip_fixed_ref_so16", False),
])
def test_fixed_genlip_configs_do_not_enable_naflex(model_name, is_generative):
    args = parse_args(["--model", model_name])
    assert args.genlip is is_generative
    assert args.use_naflex is False


@pytest.mark.parametrize("model_name,is_generative", [
    ("naflexgenlip_ref_so16", True),
    ("clip_genlip_ref_so16", False),
])
def test_naflex_genlip_configs_still_enable_naflex(model_name, is_generative):
    args = parse_args(["--model", model_name])
    assert args.genlip is is_generative
    assert args.use_naflex is True

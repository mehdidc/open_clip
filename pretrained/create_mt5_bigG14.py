import torch
import open_clip


models = [
    # "mt5-xxl-ViT-BigG-14",
    # "coca_encoder-mt5-xxl_decoder-scratch_vis-ViT-BigG-14",
    "coca_encoder-mt5-large_decoder-scratch_vis-ViT-BigG-14",
    # "coca_encoder-mt5-xxl_decoder-mt5-xxl_vis-ViT-BigG-14",
    # "coca_encoder-mt5-xxl_decoder-mt5-xl_vis-ViT-BigG-14",
    # "coca_encoder-mt5-xxl_decoder-mt5-large_vis-ViT-BigG-14",
    # "coca_encoder-mt5-xxl_decoder-mt5-base_vis-ViT-BigG-14",
    # "coca_encoder-mt5-xl_decoder-mt5-xl_vis-ViT-BigG-14",
    # "coca_encoder-mt5-large_decoder-mt5-large_vis-ViT-BigG-14",
]

for name in models:
    # get text pretrained tower
    model_text, _, _ = open_clip.create_model_and_transforms(name)
    state_merged = model_text.state_dict()

    # get image pretrained tower
    model_visual, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained="laion2b_s39b_b160k")
    state_visual = model_visual.state_dict()

    # merge into state_merged
    visual_keys = [k for k in state_visual.keys() if 'visual' in k]
    for k in visual_keys:
        state_merged[k] = state_visual[k]

    # save
    with open(f"{name}.pt", "wb") as f:
      torch.save({"epoch": 0, "name": "go", "state_dict": state_merged}, f)


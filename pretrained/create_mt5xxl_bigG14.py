import torch
import open_clip

# get text pretrained tower
model_text, _, _ = open_clip.create_model_and_transforms('mt5-xxl-ViT-BigG-14')
state_merged = model_text.state_dict()

# get image pretrained tower
model_visual, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained="laion2b_s39b_b160k")
state_visual = model_visual.state_dict()

# merge into state_merged
visual_keys = [k for k in state_visual.keys() if 'visual' in k]
for k in visual_keys:
        state_merged[k] = state_visual[k]

# save
with open("mt5xxl_bigG14.pt", "wb") as f:
  torch.save({"epoch": 0, "name": "go", "state_dict": state_merged}, f)


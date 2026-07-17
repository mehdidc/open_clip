# Loading reference GenLIP checkpoints

OpenCLIP can initialize either its full generative GenLIP model or only the vision tower from
the Hugging Face-format checkpoints exported by the reference GenLIP repository. Use the
`hf_ckpt` weights produced by GenLIP's `save_hf_weights` flow, rather than its distributed
optimizer checkpoint.

Supported reference architectures are `l16`, `so16`, and `g16`.

## Fixed-resolution, non-NaFlex models

The built-in `genlip_ref_{l16,so16,g16}` models mirror the reference stage-1 input path: they
accept raw `[B, 3, H, W]` image tensors and apply the checkpoint's Conv2d patch embedding directly.
Use `clip_genlip_fixed_ref_{l16,so16,g16}` to put the same fixed/raw-pixel image encoder in a
contrastive CLIP.

```python
generative = open_clip.create_model(
    "genlip_ref_so16", pretrained="/path/to/hf_ckpt/model.safetensors")
contrastive = open_clip.create_model(
    "clip_genlip_fixed_ref_so16",
    pretrained_image_path="/path/to/hf_ckpt/model.safetensors",
)
```

Custom configs select the same implementation by setting `vision_cfg.genlip_naflex` to `false`.
Fixed models use the normal OpenCLIP image transform and do not enable the NaFlex batch scheduler.

Both fixed and NaFlex GenLIP vision towers support OpenCLIP's image locking options. With
`--lock-image`, the default `--lock-image-unlocked-groups 0` freezes the entire image tower;
`1` leaves only the final vision projection trainable, and `2` leaves the projection plus the
last transformer block/final norm trainable. Repeated calls can progressively unfreeze or re-lock
the tower. `--lock-image-freeze-bn-stats` is accepted but is a no-op because GenLIP has no batch
normalization layers.

## Continue generative GenLIP pretraining

```bash
python -m open_clip_train.main \
  --model naflexgenlip_ref_so16 \
  --pretrained /path/to/hf_ckpt/model.safetensors \
  --train-data '/path/to/shards/{000000..000999}.tar' \
  --dataset-type webdataset \
  --batch-size 64
```

The corresponding Python API is:

```python
model = open_clip.create_model(
    "naflexgenlip_ref_so16",
    pretrained="/path/to/hf_ckpt/model.safetensors",
)
```

This loads the patch embedding, unified image/text trunk, final norm, token embedding,
input/output projections, and LM head. It starts a new OpenCLIP optimizer; it does not import
the reference trainer's optimizer or scheduler state.

## Initialize a contrastive CLIP vision tower

```bash
python -m open_clip_train.main \
  --model clip_genlip_ref_so16 \
  --pretrained-image-path /path/to/hf_ckpt/model.safetensors \
  --train-data '/path/to/shards/{000000..000999}.tar' \
  --dataset-type webdataset \
  --batch-size 64
```

or:

```python
model = open_clip.create_model(
    "clip_genlip_ref_so16",
    pretrained_image_path="/path/to/hf_ckpt/model.safetensors",
)
```

This imports the reference patch embedding, transformer trunk, final norm, and learned
width-to-1024 projection. The contrastive text tower and logit scale are initialized by
OpenCLIP.

The converter automatically flattens the reference Conv2d patch kernel into OpenCLIP's
NaFlex linear patch projection and validates every transferred tensor shape. A mismatch in
the trunk is an error; only a deliberately different contrastive output dimension may leave
the final vision projection newly initialized.

# Loading reference GenLIP checkpoints

OpenCLIP can initialize either its full generative GenLIP model or only the vision tower from
the Hugging Face-format checkpoints exported by the reference GenLIP repository. Use the
`hf_ckpt` weights produced by GenLIP's `save_hf_weights` flow, rather than its distributed
optimizer checkpoint.

Supported reference architectures are `l16`, `so16`, and `g16`.

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

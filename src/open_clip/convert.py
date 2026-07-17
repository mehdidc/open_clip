""" Conversion functions for 3rd part state-dicts and non-torch native checkpoint formats.
"""
from typing import Union

import torch
import numpy as np

from .model import CLIP, CustomTextCLIP
from .transformer import TextTransformer, Transformer


@torch.no_grad()
def load_big_vision_weights(model: CustomTextCLIP, checkpoint_path: str):
    """ Load weights from .npz checkpoints for official Google big_vision image-text models

    Currently, the SigLIP source models are supported and a CustomTextCLIP destination model
    w/ timm image encoder.
    """
    from timm.layers import resample_patch_embed, resample_abs_pos_embed

    def _n2p(w, t=True, idx=None):
        if idx is not None:
            w = w[idx]
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    interpolation = 'bilinear'
    antialias = False

    def _linear_patch_embed_to_conv(embed_w, conv_weight_shape):
        out_channels, in_channels, patch_height, patch_width = conv_weight_shape
        if embed_w.shape != (out_channels, in_channels * patch_height * patch_width):
            return embed_w
        return embed_w.reshape(out_channels, patch_height, patch_width, in_channels).permute(0, 3, 1, 2)

    def _reshape_grid_pos_embed(pos_embed_w, target_shape):
        if pos_embed_w.ndim == 3:
            pos_embed_w = pos_embed_w.unsqueeze(0)
        if pos_embed_w.shape == target_shape:
            return pos_embed_w

        if pos_embed_w.ndim == 4 and len(target_shape) == 4:
            old_size = pos_embed_w.shape[1:3]
            new_size = target_shape[1:3]
            pos_embed_w = resample_abs_pos_embed(
                pos_embed_w.reshape(1, -1, pos_embed_w.shape[-1]),
                new_size=new_size,
                old_size=old_size,
                num_prefix_tokens=0,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
            pos_embed_w = pos_embed_w.reshape(target_shape)
        return pos_embed_w

    def _copy_map_head(module, prefix):
        if module.attn_pool is None:
            return

        block_prefix = f'{prefix}MAPHead_0/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_0/'
        module.attn_pool.latent.copy_(_n2p(w[f'{block_prefix}probe'], t=False))
        module.attn_pool.q.weight.copy_(_n2p(w[f'{mha_prefix}query/kernel'], t=False).flatten(1).T)
        module.attn_pool.q.bias.copy_(_n2p(w[f'{mha_prefix}query/bias'], t=False).reshape(-1))
        module.attn_pool.kv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('key', 'value')]))
        module.attn_pool.kv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('key', 'value')]))
        module.attn_pool.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        module.attn_pool.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        module.attn_pool.norm.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        module.attn_pool.norm.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        for r in range(2):
            getattr(module.attn_pool.mlp, f'fc{r + 1}').weight.copy_(
                _n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/kernel']))
            getattr(module.attn_pool.mlp, f'fc{r + 1}').bias.copy_(
                _n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/bias']))

    def _copy_timm_img_blocks(blocks, prefix):
        mha_sub, b_sub, ln1_sub = (0, 0, 1)
        for i, block in enumerate(blocks.children()):
            if f'{prefix}Transformer/encoderblock/LayerNorm_0/scale' in w:
                block_prefix = f'{prefix}Transformer/encoderblock/'
                idx = i
            else:
                block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
                idx = None
            mha_prefix = block_prefix + f'MultiHeadDotProductAttention_{mha_sub}/'
            block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale'], idx=idx))
            block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias'], idx=idx))
            block.attn.qkv.weight.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/kernel'], t=False, idx=idx).flatten(1).T for n in ('query', 'key', 'value')]))
            block.attn.qkv.bias.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/bias'], t=False, idx=idx).reshape(-1) for n in ('query', 'key', 'value')]))
            block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel'], idx=idx).flatten(1))
            block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias'], idx=idx))
            block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/scale'], idx=idx))
            block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/bias'], idx=idx))
            for r in range(2):
                getattr(block.mlp, f'fc{r + 1}').weight.copy_(
                    _n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel'], idx=idx))
                getattr(block.mlp, f'fc{r + 1}').bias.copy_(
                    _n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias'], idx=idx))

    def _convert_timm_img(module, prefix):
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
        if embed_conv_w.ndim == 2:
            embed_conv_w = _linear_patch_embed_to_conv(embed_conv_w, module.patch_embed.proj.weight.shape)
        if embed_conv_w.shape[-2:] != module.patch_embed.proj.weight.shape[-2:]:
            embed_conv_w = resample_patch_embed(
                embed_conv_w,
                module.patch_embed.proj.weight.shape[-2:],
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        module.patch_embed.proj.weight.copy_(embed_conv_w)
        module.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))

        if module.cls_token is not None:
            module.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))

        pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False)
        if pos_embed_w.ndim == 3:
            pos_embed_w = pos_embed_w.reshape(1, -1, pos_embed_w.shape[-1])
        if pos_embed_w.shape != module.pos_embed.shape:
            num_prefix_tokens = (
                0 if getattr(module, 'no_embed_class', False) else getattr(module, 'num_prefix_tokens', 1)
            )
            pos_embed_w = resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
                pos_embed_w,
                new_size=module.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        module.pos_embed.copy_(pos_embed_w)

        _copy_timm_img_blocks(module.blocks, prefix)

        module.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
        module.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))

        _copy_map_head(module, prefix)

    def _convert_naflex_timm_img(module, prefix):
        module.embeds.proj.weight.copy_(_n2p(w[f'{prefix}embedding/kernel']))
        if module.embeds.proj.bias is not None:
            module.embeds.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))

        if module.embeds.cls_token is not None and f'{prefix}cls' in w:
            module.embeds.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
        if module.embeds.pos_embed is not None:
            pos_embed_w = _reshape_grid_pos_embed(
                _n2p(w[f'{prefix}pos_embedding'], t=False),
                module.embeds.pos_embed.shape,
            )
            module.embeds.pos_embed.copy_(pos_embed_w)

        _copy_timm_img_blocks(module.blocks, prefix)
        if hasattr(module.norm, 'weight'):
            module.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
            module.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
        _copy_map_head(module, prefix)

    def _convert_openclip_transformer(module: Transformer, prefix):
        for i, block in enumerate(module.resblocks.children()):
            if f'{prefix}encoderblock/LayerNorm_0/scale' in w:
                block_prefix = f'{prefix}encoderblock/'
                idx = i
            else:
                block_prefix = f'{prefix}encoderblock_{i}/'
                idx = None
            mha_prefix = block_prefix + f'MultiHeadDotProductAttention_0/'
            block.ln_1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale'], idx=idx))
            block.ln_1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias'], idx=idx))
            block.attn.in_proj_weight.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/kernel'], t=False, idx=idx).flatten(1).T for n in ('query', 'key', 'value')]))
            block.attn.in_proj_bias.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/bias'], t=False, idx=idx).reshape(-1) for n in ('query', 'key', 'value')]))
            block.attn.out_proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel'], idx=idx).flatten(1))
            block.attn.out_proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias'], idx=idx))
            block.ln_2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_1/scale'], idx=idx))
            block.ln_2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_1/bias'], idx=idx))
            block.mlp.c_fc.weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_0/kernel'], idx=idx))
            block.mlp.c_fc.bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_0/bias'], idx=idx))
            block.mlp.c_proj.weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_1/kernel'], idx=idx))
            block.mlp.c_proj.bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_1/bias'], idx=idx))

    def _convert_openclip_txt(module: TextTransformer, prefix):
        module.token_embedding.weight.copy_(_n2p(w[f'{prefix}Embed_0/embedding'], t=False))
        pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False).squeeze(0)
        module.positional_embedding.copy_(pos_embed_w)
        _convert_openclip_transformer(module.transformer, prefix=prefix + 'Encoder_0/')
        module.ln_final.weight.copy_(_n2p(w[f'{prefix}Encoder_0/encoder_norm/scale']))
        module.ln_final.bias.copy_(_n2p(w[f'{prefix}Encoder_0/encoder_norm/bias']))
        if module.text_projection is not None:
            module.text_projection.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
            module.text_projection.bias.copy_(_n2p(w[f'{prefix}head/bias']))

    root_prefix = 'params/' if 'params/b' in w else ''
    if model.visual.trunk.__class__.__name__ == 'NaFlexVit':
        _convert_naflex_timm_img(model.visual.trunk, f'{root_prefix}img/')
    else:
        _convert_timm_img(model.visual.trunk, f'{root_prefix}img/')
    _convert_openclip_txt(model.text, f'{root_prefix}txt/')
    model.logit_bias.copy_(_n2p(w[f'{root_prefix}b'])[0])
    model.logit_scale.copy_(_n2p(w[f'{root_prefix}t'])[0])


@torch.no_grad()
def convert_mobile_clip_state_dict(model: CustomTextCLIP, state_dict, fastvit = True):

    def _convert_timm_img(state_dict):
        if fastvit:
            from timm.models.fastvit import checkpoint_filter_fn
        else:
            from timm.models.vision_transformer_hybrid import checkpoint_filter_fn
        timm_state_dict = checkpoint_filter_fn(state_dict, model.visual.trunk)
        timm_state_dict = {'visual.trunk.' + k: v for k, v in timm_state_dict.items()}
        return timm_state_dict

    def _convert_openclip_txt(state_dict, prefix='text_encoder.'):
        text_dict = {}
        for k, v in state_dict.items():
            if not k.startswith(prefix):
                continue
            k = k.replace(prefix, '')
            k = k.replace('projection_layer', 'text_projection')
            k = k.replace('embedding_layer', 'token_embedding')
            if k.startswith('positional_embedding.pos_embed.pos_embed'):
                k = k.replace('positional_embedding.pos_embed.pos_embed', 'positional_embedding')
                v = v.squeeze()
            k = k.replace('final_layer_norm', 'ln_final')
            k = k.replace('pre_norm_mha.0', 'ln_1')
            k = k.replace('pre_norm_mha.1', 'attn')
            k = k.replace('pre_norm_ffn.0', 'ln_2')
            k = k.replace('pre_norm_ffn.1', 'mlp.c_fc')
            k = k.replace('pre_norm_ffn.4', 'mlp.c_proj')
            k = k.replace('qkv_proj.weight', 'in_proj_weight')
            k = k.replace('qkv_proj.bias', 'in_proj_bias')
            k = k.replace('transformer.', 'transformer.resblocks.')
            text_dict['text.' + k] = v
        return text_dict

    image_dict = _convert_timm_img(state_dict)
    text_dict = _convert_openclip_txt(state_dict)
    out_dict = {**image_dict, **text_dict}
    out_dict['logit_scale'] = state_dict['logit_scale']
    return out_dict


def convert_mammut_state_dict(model, state_dict):
    """Convert a LAION open_clip_mammut fork checkpoint to the current MaMMUT layout.

    The fork repurposed the decoder's ``text_projection`` as the vocab head; here that
    weight lives in ``lm_head`` (a pure rename, [width, vocab_size], no transpose).
    All other keys align by construction. Only applicable when ``text.lm_head`` is
    absent -- current checkpoints can hold both ``text.lm_head`` and a real
    ``text.text_projection`` (classic decoder w/ proj_type='linear').
    """
    out_dict = dict(state_dict)
    out_dict['text.lm_head'] = out_dict.pop('text.text_projection')
    return out_dict


def is_genlip_state_dict(state_dict):
    """Return True for checkpoints exported by the reference GenLIP codebase."""
    return (
        'vision_embeddings.patch_embedding.weight' in state_dict
        and any(k.startswith('visual.layers.0.self_attn.') for k in state_dict)
    )


def convert_genlip_state_dict(model, state_dict):
    """Convert a reference GenLIP checkpoint to an OpenCLIP GenLIP layout.

    The reference implementation stores its image patch projection as a Conv2d and calls
    the shared transformer ``visual``. OpenCLIP consumes pre-patchified NaFlex inputs through
    a Linear and calls the same transformer ``trunk``. All transformer tensors otherwise have
    identical shapes and semantics.

    ``model`` may be a full :class:`NaFlexGenLip` or a standalone GenLIP vision tower. Using
    the destination state dict to select keys also takes care of the registered aliases under
    ``NaFlexGenLip.visual`` without hard-coding one target model shape.
    """
    target_state = model.state_dict()
    canonical = {}

    for key, value in state_dict.items():
        if key.startswith('vision_embeddings.patch_embedding.'):
            suffix = key.removeprefix('vision_embeddings.patch_embedding.')
            canonical[f'patch_embed.proj.{suffix}'] = value
        elif key.startswith('visual.layers.'):
            canonical['trunk.layers.' + key.removeprefix('visual.layers.')] = value
        elif key.startswith('ln_post.'):
            canonical['trunk.ln_post.' + key.removeprefix('ln_post.')] = value
        elif key == 'embeddings.weight':
            canonical['text_embed.weight'] = value
        elif key.startswith('in_proj.'):
            canonical[key] = value
        elif key.startswith('proj.'):
            canonical['out_proj.' + key.removeprefix('proj.')] = value
        elif key.startswith('lm_head.'):
            canonical[key] = value

    vision_only = not any(
        k.startswith(('text_embed.', 'lm_head.', 'in_proj.', 'out_proj.'))
        for k in target_state
    )
    converted = {}
    for target_key, target_value in target_state.items():
        source_key = target_key
        if target_key.startswith('visual.patch_embed.'):
            source_key = target_key.removeprefix('visual.')
        elif target_key.startswith('visual.trunk.'):
            source_key = target_key.removeprefix('visual.')
        elif target_key.startswith('visual.proj.'):
            source_key = 'out_proj.' + target_key.removeprefix('visual.proj.')
        elif target_key.startswith('proj.'):
            # Standalone GenLIP vision tower: initialize its contrastive projection from
            # the reference model's learned width -> text_embed_dim projection.
            source_key = 'out_proj.' + target_key.removeprefix('proj.')

        if source_key not in canonical:
            continue
        value = canonical[source_key]
        if source_key == 'patch_embed.proj.weight' and value.ndim == 4 and target_value.ndim == 2:
            # NaFlex consumes flattened patches through Linear; the fixed adapter retains Conv2d.
            value = value.flatten(1)
        if value.shape != target_value.shape:
            if vision_only and target_key.startswith('proj.'):
                # A contrastive model may deliberately choose an embedding dimension different
                # from GenLIP's text dimension. Keep that new projection randomly initialized.
                continue
            raise RuntimeError(
                f"GenLIP checkpoint tensor '{source_key}' has shape {tuple(value.shape)}, but "
                f"destination '{target_key}' expects {tuple(target_value.shape)}. Check that the "
                "OpenCLIP model config matches the reference GenLIP architecture."
            )
        converted[target_key] = value

    if not converted:
        raise RuntimeError(
            "Detected a reference GenLIP checkpoint, but the destination model has no compatible "
            "GenLIP parameters. Use a NaFlexGenLip model or a CLIP config with a GenLIP vision tower."
        )
    return converted


def convert_vision_state_dict(model, state_dict):
    """Convert a third-party checkpoint intended for a vision tower only."""
    if is_genlip_state_dict(state_dict):
        return convert_genlip_state_dict(model, state_dict), 'genlip'
    return state_dict, None


def convert_state_dict(model: Union[CustomTextCLIP, CLIP], state_dict):
    if is_genlip_state_dict(state_dict):
        state_dict = convert_genlip_state_dict(model, state_dict)
    if 'image_encoder.model.patch_embed.0.rbr_conv.0.conv.weight' in state_dict:
        # Apple MobileCLIP s1 & s2 state_dicts (s0 and b not currently supported)
        state_dict = convert_mobile_clip_state_dict(model, state_dict)
    if 'image_encoder.model.patch_emb.0.block.conv.weight' in state_dict:
        # convert b model
        state_dict = convert_mobile_clip_state_dict(model, state_dict, fastvit=False)
    if (
        'map_viz2txt_kv' in state_dict
        and 'text.text_projection' in state_dict
        and 'text.lm_head' not in state_dict
    ):
        # LAION open_clip_mammut fork MaMMUT checkpoint (text_projection is the repurposed vocab
        # head there; current checkpoints always carry text.lm_head, plus optionally a real
        # text.text_projection contrastive projection)
        state_dict = convert_mammut_state_dict(model, state_dict)
    return state_dict

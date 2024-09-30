from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower
from .coca_model import MultimodalCfg
from .transformer import QuickGELU, LayerNormFp32, LayerNorm, CapTransformer
#from .generation_utils import Generator



def _build_multimodal_decoder_tower(
        multimodal_cfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = (
        LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    )

    decoder = CapTransformer(
        context_length=multimodal_cfg.context_length,
        vocab_size=multimodal_cfg.vocab_size,
        width=multimodal_cfg.width,
        heads=multimodal_cfg.heads,
        layers=multimodal_cfg.layers,
        ls_init_value=multimodal_cfg.ls_init_value,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return decoder

class Cap(torch.nn.Module):
    def __init__(
        self,
        text_cfg: MultimodalCfg,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        pad_id: int = 0,
    ):
        super().__init__()
        multimodal_cfg = MultimodalCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        vision_cfg = (
            CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg
        )

        vocab_size = (
            self.text.config.vocab_size  # for hf models
            if multimodal_cfg.__dict__.get("hf_model_name", None) is not None
            else multimodal_cfg.vocab_size
        )

        self.text = _build_multimodal_decoder_tower(
            multimodal_cfg=multimodal_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.visual = _build_vision_tower(
            embed_dim=vision_cfg.width,
            vision_cfg=vision_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.context_length = multimodal_cfg.context_length
        self.map_viz2txt_kv = nn.Parameter(torch.randn(vision_cfg.width, multimodal_cfg.width))
        self.pad_id = pad_id
        self.use_contrastive = True

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_text(self, text, image_embs):
        token_logits = self.text(
            text=text,
            image_embs=image_embs,
        )
        return token_logits

    def encode_image(self, image, normalize: bool=True):
        image_latent, image_embs = self.visual(image)
        image_latent = F.normalize(image_latent, dim=-1) if normalize else image_latent
        return image_latent, image_embs

    def forward(self, image=None, text=None, image_embs=None, is_training=True):
        if (image_embs is None) and (image is not None):
            image_latent, image_embs = self.encode_image(image)
        out = {}
        out["labels"] = text[:, 1:]  # shift labels
        text = text[:, :-1] if is_training else text # drop last tok because it has no label
        image_embs = image_embs @ self.map_viz2txt_kv 
        out["logits"] = self.encode_text(text, image_embs=image_embs)
        return out

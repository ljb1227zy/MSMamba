import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model

from .base_segmenter import MSMambaBaseSegmenter
from .customized_model import MSMambaBackbone
from .utils import load_ckpt, update_mamba_config


class ConvBN(nn.Module):
    """2D convolution followed by BatchNorm and an optional activation layer."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bn_weight_init=1.0,
        norm_layer=nn.BatchNorm2d,
        act_layer=None,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bn = norm_layer(out_channels)
        self.act = act_layer() if act_layer is not None else nn.Identity()

        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0.0)
        self._init_conv()

    def _init_conv(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = (
                    module.kernel_size[0]
                    * module.kernel_size[1]
                    * module.out_channels
                    // module.groups
                )
                module.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvRefineBlock(nn.Module):
    """Lightweight convolutional refinement block used in the decoder."""

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.Hardswish,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels

        self.proj_in = ConvBN(in_channels, hidden_channels, act_layer=act_layer)
        self.dwconv = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=hidden_channels,
        )
        self.norm = norm_layer(hidden_channels)
        self.act = act_layer()
        self.proj_out = ConvBN(hidden_channels, out_channels)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv2d):
            fan_out = (
                module.kernel_size[0]
                * module.kernel_size[1]
                * module.out_channels
                // module.groups
            )
            module.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(self, x):
        x = self.proj_in(x)
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.proj_out(x)
        return x


class SoftFiLMLayer(nn.Module):
    """
    Text-conditioned FiLM modulation with a small learnable modulation scale.

    The layer keeps the initial modulation weak, which stabilizes training and
    prevents language features from overwhelming visual features in early epochs.
    """

    def __init__(self, channels, init_scale=0.1):
        super().__init__()
        self.channels = channels
        self.scale = nn.Parameter(torch.ones(1) * init_scale)

    def forward(self, x, gamma, beta):
        gamma = gamma.view(-1, self.channels, 1, 1)
        beta = beta.view(-1, self.channels, 1, 1)

        gamma = 1.0 + torch.tanh(gamma) * self.scale
        beta = beta * self.scale

        return gamma * x + beta


class MSFDDecoder(nn.Module):
    """
    Multi-scale semantic fusion decoder.

    The decoder progressively fuses the four-stage visual feature pyramid and
    injects sentence-level language guidance through Soft-FiLM modulation.
    """

    def __init__(
        self,
        in_dim,
        text_dim=768,
        hidden_dim=None,
        text_hidden_dim=256,
        **kwargs,
    ):
        super().__init__()

        self.text_dim = text_dim
        self.hidden_dim = hidden_dim or in_dim * 4

        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, text_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(text_hidden_dim, self.hidden_dim * 2),
        )
        self.film = SoftFiLMLayer(self.hidden_dim)

        self.stage4_refine = ConvRefineBlock(
            in_channels=in_dim * 8,
            out_channels=self.hidden_dim,
        )
        self.stage3_refine = ConvRefineBlock(
            in_channels=in_dim * 4 + self.hidden_dim,
            out_channels=self.hidden_dim,
        )
        self.stage2_refine = ConvRefineBlock(
            in_channels=in_dim * 2 + self.hidden_dim,
            out_channels=self.hidden_dim,
        )

        self.pred_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim + in_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(self.hidden_dim, 1, kernel_size=3, padding=1),
        )

    def _pool_text_features(self, text_feats, text_mask=None):
        """
        Convert token-level language features into one sentence-level embedding.

        Supported layouts:
            - (B, N, C): token-first layout
            - (B, C, N): channel-first layout
        """

        if text_feats.dim() != 3:
            raise ValueError(
                f"Expected text_feats to be a 3D tensor, but got shape {text_feats.shape}."
            )

        if text_feats.shape[-1] == self.text_dim:
            # Layout: (B, N, C)
            if text_mask is None:
                return text_feats.mean(dim=1)

            mask = text_mask.float().unsqueeze(-1)
            text_sum = (text_feats * mask).sum(dim=1)
            valid_tokens = mask.sum(dim=1).clamp(min=1.0)
            return text_sum / valid_tokens

        if text_feats.shape[1] == self.text_dim:
            # Layout: (B, C, N)
            if text_mask is None:
                return text_feats.mean(dim=-1)

            mask = text_mask.float().unsqueeze(1)
            text_sum = (text_feats * mask).sum(dim=-1)
            valid_tokens = mask.sum(dim=-1).clamp(min=1.0)
            return text_sum / valid_tokens

        raise ValueError(
            "Cannot infer the text feature layout. Expected either "
            f"(B, N, {self.text_dim}) or (B, {self.text_dim}, N), "
            f"but got shape {text_feats.shape}."
        )

    def _apply_text_film(self, x, text_embedding):
        gamma_beta = self.text_encoder(text_embedding)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return self.film(x, gamma, beta)

    def forward(self, features, text_feats, text_mask=None):
        stage4, stage3, stage2, stage1 = features

        text_embedding = self._pool_text_features(text_feats, text_mask)

        x = self.stage4_refine(stage4)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self._apply_text_film(x, text_embedding)

        x = self.stage3_refine(torch.cat([x, stage3], dim=1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self._apply_text_film(x, text_embedding)

        x = self.stage2_refine(torch.cat([x, stage2], dim=1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self._apply_text_film(x, text_embedding)

        x = self.pred_head(torch.cat([x, stage1], dim=1))
        return x


class MSMambaSegmenter(MSMambaBaseSegmenter):
    """MSMamba segmentation model with an MSFD decoder."""

    def __init__(self, backbone, img_size=480, patch_size=4, embed_dim=128, **kwargs):
        super().__init__(backbone)

        self.decoder = MSFDDecoder(
            in_dim=embed_dim,
            text_dim=768,
            hidden_dim=embed_dim * 4,
            text_hidden_dim=256,
            resolution=img_size // patch_size,
        )


@register_model
def MSMamba(img_size=480, model_size="base", pretrained=True, **kwargs):
    """
    Build the MSMamba model.

    Returns:
        model: MSMamba segmentation model.
        new_param: parameters or parameter names that are newly initialized
            after loading the pretrained backbone checkpoint. This return value
            is kept for compatibility with the existing optimizer builder.
    """

    config = update_mamba_config(model_size)
    backbone = MSMambaBackbone(**config)

    if pretrained:
        backbone, load_info = load_ckpt(backbone, model_size)
        new_param = load_info[0]
    else:
        new_param = []

    model = MSMambaSegmenter(
        backbone=backbone,
        img_size=img_size,
        embed_dim=config["dims"][0],
        patch_size=config["patch_size"],
    )

    return model, new_param

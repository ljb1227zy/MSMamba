import math

import einops
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath

from vmamba_model.vmamba import SS2D, VSSM, LayerNorm2d, Linear2d

from .utils import ImageTextCorr


DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def get_norm_layer(name):
    """Return the normalization layer used by VMamba-style blocks."""
    norm_layers = {
        "ln": nn.LayerNorm,
        "ln2d": LayerNorm2d,
        "bn": nn.BatchNorm2d,
    }
    layer = norm_layers.get(str(name).lower())
    if layer is None:
        raise ValueError(f"Unsupported norm layer: {name}")
    return layer


def get_activation_layer(name):
    """Return the activation layer used by the selective-scan block."""
    activation_layers = {
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
    }
    layer = activation_layers.get(str(name).lower())
    if layer is None:
        raise ValueError(f"Unsupported activation layer: {name}")
    return layer


class MultiSemanticSS2D(nn.Module):
    """
    Multi-semantic selective-scan fusion block.

    The block receives four spatial feature maps:
        1. visual feature;
        2. sentence-level text condition;
        3. local pixel-word condition;
        4. attribute-word condition.

    These features are concatenated along the channel dimension and then fused
    by an SS2D module.
    """

    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        forward_type="v4",
        channel_first=False,
        **kwargs,
    ):
        super().__init__()

        self.ss2d = SS2D(
            d_model,
            d_state,
            ssm_ratio,
            dt_rank,
            act_layer,
            d_conv,
            conv_bias,
            dropout,
            bias,
            dt_min,
            dt_max,
            dt_init,
            dt_scale,
            dt_init_floor,
            initialize,
            forward_type,
            channel_first,
            **kwargs,
        )

        # Four semantic branches are concatenated before SS2D.
        self.ss2d.in_proj = Linear2d(
            d_model * 4,
            self.ss2d.in_proj.weight.shape[0],
            bias=bias,
        )

    def forward(self, inputs):
        image_feat, sentence_cond, local_cond, attribute_cond = inputs

        if sentence_cond is None:
            raise ValueError("sentence_cond should not be None in MultiSemanticSS2D.")
        if local_cond is None:
            raise ValueError("local_cond should not be None in MultiSemanticSS2D.")
        if attribute_cond is None:
            raise ValueError("attribute_cond should not be None in MultiSemanticSS2D.")

        fused_input = torch.cat(
            [image_feat, sentence_cond, attribute_cond, local_cond],
            dim=1,
        )
        fused_image = self.ss2d(fused_input)

        return [fused_image, sentence_cond, local_cond, attribute_cond]


class MultiSemanticVSSBlock(nn.Module):
    """A VSS block that supports image-text multi-semantic fusion."""

    def __init__(self, forward_core="MultiSemanticSS2D", **kwargs):
        super().__init__()

        dim = kwargs["dim"]
        drop_path = kwargs["drop_path"]
        norm_layer = kwargs["norm_layer"]

        self.norm = norm_layer(dim)
        self.forward_core = forward_core

        if forward_core == "SS2D":
            self.self_attention = SS2D(
                d_model=dim,
                d_state=kwargs["ssm_d_state"],
                dt_rank=kwargs["ssm_dt_rank"],
                act_layer=kwargs["ssm_act_layer"],
                d_conv=kwargs["ssm_conv"],
                conv_bias=kwargs["ssm_conv_bias"],
                dropout=kwargs["ssm_drop_rate"],
                initialize=kwargs["ssm_init"],
                **kwargs,
            )
        elif forward_core == "MultiSemanticSS2D":
            self.self_attention = MultiSemanticSS2D(
                d_model=dim,
                d_state=kwargs["ssm_d_state"],
                ssm_ratio=kwargs["ssm_ratio"],
                dt_rank=kwargs["ssm_dt_rank"],
                act_layer=kwargs["ssm_act_layer"],
                d_conv=kwargs["ssm_conv"],
                conv_bias=kwargs["ssm_conv_bias"],
                dropout=kwargs["ssm_drop_rate"],
                initialize=kwargs["ssm_init"],
                forward_type=kwargs["forward_type"],
                channel_first=kwargs["channel_first"],
            )
        else:
            raise ValueError(f"Unsupported forward core: {forward_core}")

        self.drop_path = DropPath(drop_path)

    def forward(self, inputs):
        if isinstance(inputs, torch.Tensor):
            residual = inputs
            x = self.norm(inputs)
            x = self.self_attention(x)
            return residual + self.drop_path(x)

        normalized_inputs = [
            self.norm(item) if item is not None else None
            for item in inputs
        ]
        outputs = self.self_attention(normalized_inputs)

        # Keep the same residual behavior as the original implementation.
        return [
            residual + self.drop_path(output) if residual is not None else None
            for residual, output in zip(inputs, outputs)
        ]


class MultiSemanticVSSLayer(nn.Module):
    """A stack of multi-semantic VSS blocks for one feature stage."""

    def __init__(
        self,
        depth,
        downsample=None,
        forward_core="MultiSemanticSS2D",
        **kwargs,
    ):
        super().__init__()

        dim = kwargs["dim"]
        use_checkpoint = kwargs["use_checkpoint"]
        norm_layer = kwargs["norm_layer"]

        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                MultiSemanticVSSBlock(
                    forward_core=forward_core,
                    drop_path=0.0,
                    **kwargs,
                )
                for _ in range(depth)
            ]
        )

        if downsample is not None:
            self.downsample = downsample(
                dim=dim,
                norm_layer=norm_layer,
                channel_first=kwargs["channel_first"],
            )
        else:
            self.downsample = None

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        for name, param in module.named_parameters():
            if name == "out_proj.weight":
                param = param.clone().detach_()
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))

    def forward(self, x, *args, **kwargs):
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        inner = x
        if self.downsample is not None:
            x = self.downsample(x)

        return x, inner


class MSMambaBackbone(VSSM):
    """
    MSMamba backbone for referring remote sensing image segmentation.

    The backbone extends a VMamba visual encoder with three language-guided
    semantic conditions:
        - sentence-level global guidance;
        - local pixel-word correlation;
        - attribute-word guidance.

    Each stage first extracts visual features using the original VMamba layer,
    then injects the language conditions through a multi-semantic SS2D block.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.classifier = None
        self.dims = kwargs["dims"]
        self.use_checkpoint = kwargs["use_checkpoint"]

        norm_layer = get_norm_layer(kwargs["norm_layer"])
        ssm_act_layer = get_activation_layer(kwargs["ssm_act_layer"])

        self.sentence_projection = nn.ModuleList()
        self.local_text_fusion = nn.ModuleList()
        self.attribute_projection = nn.ModuleList()
        self.sentence_cross_attention = nn.ModuleList()
        self.multi_semantic_layers = nn.ModuleList()

        for stage_idx in range(self.num_layers):
            dim = self.dims[stage_idx]

            self.multi_semantic_layers.append(
                MultiSemanticVSSLayer(
                    dim=dim,
                    depth=2,
                    use_checkpoint=self.use_checkpoint,
                    norm_layer=norm_layer,
                    ssm_act_layer=ssm_act_layer,
                    downsample=None,
                    channel_first=self.channel_first,
                    ssm_d_state=kwargs["ssm_d_state"],
                    ssm_ratio=kwargs["ssm_ratio"],
                    ssm_dt_rank=kwargs["ssm_dt_rank"],
                    ssm_conv=kwargs["ssm_conv"],
                    ssm_conv_bias=kwargs["ssm_conv_bias"],
                    ssm_drop_rate=kwargs["ssm_drop_rate"],
                    ssm_init=kwargs["ssm_init"],
                    forward_type=kwargs["forward_type"],
                    mlp_ratio=kwargs["mlp_ratio"],
                    mlp_act_layer=kwargs["mlp_act_layer"],
                    mlp_drop_rate=kwargs["mlp_drop_rate"],
                    gmlp=kwargs["gmlp"],
                    forward_core="MultiSemanticSS2D",
                )
            )

            self.sentence_projection.append(
                nn.Sequential(
                    nn.Linear(768, dim),
                    nn.ReLU(inplace=True),
                )
            )

            self.local_text_fusion.append(
                ImageTextCorr(
                    visual_dim=dim,
                    text_dim=768,
                    hidden_dim=512,
                    out_dim=dim,
                )
            )

            self.attribute_projection.append(
                nn.Sequential(
                    nn.Linear(768, dim),
                    nn.ReLU(inplace=True),
                )
            )

            self.sentence_cross_attention.append(
                nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=8,
                    batch_first=False,
                )
            )

    @staticmethod
    def _forward_visual_layer(x, layer):
        """Forward one original VMamba visual stage and return inner features."""
        inner = layer.blocks(x)
        out = layer.downsample(inner)
        return out, inner

    @staticmethod
    def _masked_average(features, mask):
        """Average valid token features according to a binary mask."""
        if features is None or mask is None:
            return None

        mask = mask.to(device=features.device).float()
        while mask.dim() < features.dim():
            mask = mask.unsqueeze(-1)

        pooled = (features * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return pooled / denom

    def _build_sentence_condition(self, stage_idx, image_feat, language_feat, pooler_out):
        """Build sentence-level spatial guidance with cross-attention."""
        _, _, height, width = image_feat.shape

        if pooler_out is None:
            sentence_feat = language_feat[..., 0]
        else:
            sentence_feat = pooler_out

        sentence_query = self.sentence_projection[stage_idx](sentence_feat)
        query = sentence_query.unsqueeze(0)
        key_value = image_feat.flatten(2).permute(2, 0, 1)

        attended_sentence, _ = self.sentence_cross_attention[stage_idx](
            query,
            key_value,
            key_value,
        )
        attended_sentence = attended_sentence.squeeze(0)

        return einops.repeat(
            attended_sentence,
            "b c -> b c h w",
            h=height,
            w=width,
        )

    def _build_local_condition(self, stage_idx, image_feat, language_feat, language_mask):
        """Build local pixel-word correlation features."""
        height = image_feat.shape[-2]
        local_condition = self.local_text_fusion[stage_idx](
            image_feat,
            language_feat,
            language_mask,
        )
        return einops.rearrange(
            local_condition,
            "b h w c -> b c h w",
            h=height,
        )

    def _build_attribute_condition(self, stage_idx, image_feat, attribute_feat, attribute_mask):
        """Build attribute-word spatial guidance."""
        batch_size, _, height, width = image_feat.shape

        pooled_attribute = self._masked_average(attribute_feat, attribute_mask)
        if pooled_attribute is None:
            pooled_attribute = torch.zeros(
                batch_size,
                768,
                device=image_feat.device,
                dtype=image_feat.dtype,
            )

        attribute_vector = self.attribute_projection[stage_idx](pooled_attribute)
        return einops.repeat(
            attribute_vector,
            "b c -> b c h w",
            h=height,
            w=width,
        )

    def forward(
        self,
        x,
        l_feat,
        l_mask,
        pooler_out=None,
        attributeword_features=None,
        attributeword_mask=None,
        **kwargs,
    ):
        x = self.patch_embed(x)
        outputs = []

        for stage_idx, layer in enumerate(self.layers):
            x, image_feat = self._forward_visual_layer(x, layer)

            sentence_condition = self._build_sentence_condition(
                stage_idx,
                image_feat,
                l_feat,
                pooler_out,
            )
            local_condition = self._build_local_condition(
                stage_idx,
                image_feat,
                l_feat,
                l_mask,
            )
            attribute_condition = self._build_attribute_condition(
                stage_idx,
                image_feat,
                attributeword_features,
                attributeword_mask,
            )

            multi_semantic_input = (
                image_feat,
                sentence_condition,
                local_condition,
                attribute_condition,
            )
            fused_features, _ = self.multi_semantic_layers[stage_idx](
                multi_semantic_input,
                None,
                None,
            )

            image_feat = fused_features[0]
            if layer.downsample is not None:
                x = layer.downsample(image_feat)
            else:
                x = image_feat

            outputs.append(image_feat)

        return outputs

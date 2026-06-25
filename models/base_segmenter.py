import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

from .attributeword_extractor import AttributewordExtractor
from .cross_scale_interaction import CIM
from .utils import dice_loss, sigmoid_focal_loss


class MSMambaBaseSegmenter(nn.Module):
    """
    Base segmentation wrapper for MSMamba.

    This module is responsible for:
        1. encoding text instructions with BERT;
        2. extracting attribute-word features used by the multimodal backbone;
        3. forwarding image and language features through the backbone;
        4. enhancing multi-scale visual features with CIM;
        5. decoding the final referring segmentation mask.
    """

    def __init__(
        self,
        backbone,
        bert_model_name="bert-base-uncased",
        max_text_len=20,
        local_files_only=False,
        use_attribute_words=True,
        spacy_model="en_core_web_sm",
        max_attribute_words=5,
        cim_channels=(128, 256, 512, 1024),
        **kwargs,
    ):
        super().__init__()

        self.backbone = backbone
        self.decoder = None

        self.max_text_len = max_text_len
        self.use_attribute_words = use_attribute_words

        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model_name,
            local_files_only=local_files_only,
        )
        self.text_encoder = BertModel.from_pretrained(
            bert_model_name,
            local_files_only=local_files_only,
        )

        self.cim = CIM(dim=sum(cim_channels))

        if self.use_attribute_words:
            self.attributeword_extractor = AttributewordExtractor(
                tokenizer=self.tokenizer,
                spacy_model=spacy_model,
                max_attributewords=max_attribute_words,
            )
        else:
            self.attributeword_extractor = None

    @staticmethod
    def _normalize_text(text):
        """Normalize one text expression into a plain string."""
        if isinstance(text, (list, tuple)):
            text = " ".join(str(item) for item in text)
        return " ".join(str(text).strip().split())

    def _normalize_text_batch(self, text, batch_size):
        """Convert text input into a list whose length matches the image batch size."""
        if isinstance(text, (list, tuple)):
            if len(text) == batch_size:
                return [self._normalize_text(item) for item in text]
            if len(text) == 1:
                return [self._normalize_text(text[0])] * batch_size

        return [self._normalize_text(text)] * batch_size

    def _encode_text(self, text_list, device):
        """Tokenize text and extract BERT token features."""
        encoded = self.tokenizer(
            text_list,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        text_ids = encoded["input_ids"].to(device, non_blocking=True)
        attention_mask = encoded["attention_mask"].to(device, non_blocking=True)

        bert_outputs = self.text_encoder(text_ids, attention_mask=attention_mask)
        token_features = bert_outputs["last_hidden_state"]
        pooled_features = bert_outputs.get("pooler_output", None)

        return text_ids, attention_mask, token_features, pooled_features

    def _extract_attribute_features(self, text_list, text_ids, attention_mask, token_features):
        """Extract attribute-word features from BERT token features."""
        if not self.use_attribute_words:
            return None, None

        attribute_positions, attribute_mask = (
            self.attributeword_extractor.extract_attributeword_positions(
                text_list,
                text_ids,
                attention_mask,
            )
        )

        attribute_features = (
            self.attributeword_extractor.extract_attributeword_features(
                token_features,
                attribute_positions,
                attribute_mask,
            )
        )

        return attribute_features, attribute_mask

    @staticmethod
    def _prepare_target_mask(mask):
        """Convert a segmentation mask to binary float format."""
        if mask is None:
            raise ValueError("Training requires a ground-truth mask.")

        mask = mask.float()
        if mask.max() > 1.0:
            mask = mask / 255.0
        return (mask > 0.5).float()

    def forward(self, image, text, mask=None, **kwargs):
        input_shape = image.shape[-2:]
        text_list = self._normalize_text_batch(text, image.size(0))

        text_ids, attention_mask, token_features, pooled_features = self._encode_text(
            text_list,
            image.device,
        )

        attribute_features, attribute_mask = self._extract_attribute_features(
            text_list,
            text_ids,
            attention_mask,
            token_features,
        )

        # Backbone expects language features in channel-first layout.
        language_features = token_features.permute(0, 2, 1).contiguous()
        language_mask = attention_mask.unsqueeze(-1).contiguous()

        features = self.backbone(
            image,
            language_features,
            language_mask,
            pooler_out=pooled_features,
            attributeword_features=attribute_features,
            attributeword_mask=attribute_mask,
        )

        c1, c2, c3, c4 = features
        c1, c2, c3, c4 = self.cim([c1, c2, c3, c4])

        pred = self.decoder([c4, c3, c2, c1], language_features, language_mask)
        pred = F.interpolate(
            pred,
            size=input_shape,
            mode="bilinear",
            align_corners=True,
        )

        if self.training:
            target = self._prepare_target_mask(mask)
            epoch = kwargs.get("epoch", None)
            loss = dice_loss(pred, target, epoch=epoch) + sigmoid_focal_loss(
                pred,
                target,
                epoch=epoch,
                alpha=-1,
                gamma=0,
            )
            return pred.detach(), target, loss

        return pred.detach()


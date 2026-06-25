import string
from typing import List, Optional, Tuple

import spacy
import torch


class AttributewordExtractor:
    """
    Extract attribute words from referring expressions and map them to BERT tokens.

    Attribute words include visual properties such as color, size, shape, number,
    state, and spatial relation words. The extracted token features are used by
    MSMamba as attribute-level language guidance.
    """

    SPATIAL_WORDS = {
        "left", "right", "front", "back", "behind", "above", "below", "under",
        "over", "top", "bottom", "middle", "center", "inside", "outside",
        "near", "far", "close", "distant", "upper", "lower", "inner", "outer",
        "north", "south", "east", "west", "northern", "southern", "eastern",
        "western",
    }

    COMMON_ATTRIBUTE_WORDS = {
        # Colors and brightness
        "red", "blue", "green", "yellow", "black", "white", "brown", "pink",
        "purple", "orange", "gray", "grey", "silver", "gold", "golden",
        "dark", "light", "bright", "pale",

        # Size and scale
        "big", "small", "large", "tiny", "huge", "massive", "little", "giant",
        "mini", "enormous", "vast", "compact", "medium",

        # Shape and geometry
        "round", "square", "long", "short", "tall", "wide", "narrow", "thick",
        "thin", "flat", "curved", "straight", "circular", "rectangular",
        "triangular",

        # Condition and material-like attributes
        "new", "old", "clean", "dirty", "fresh", "dry", "wet", "soft", "hard",
        "smooth", "rough", "strong", "weak", "heavy", "empty", "full",

        # Number and quantity
        "many", "few", "several", "multiple", "single", "double", "triple",

        # Spatial words
        *SPATIAL_WORDS,

        # Common vertical/depth attributes
        "high", "low", "deep", "shallow",
    }

    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "this", "that", "these", "those",
    }

    ADJECTIVE_SUFFIXES = (
        "ful", "less", "able", "ible", "ed", "ing", "ous", "ious", "eous",
        "al", "ial", "ic", "tic", "ive", "ative", "itive", "ly", "y", "ish",
        "ern", "en", "ese", "ian", "ist",
    )

    NOUN_SUFFIXES = (
        "tion", "sion", "ness", "ment", "ship", "hood", "ity", "ty", "er",
        "or", "ist",
    )

    def __init__(
        self,
        tokenizer,
        spacy_model: str = "en_core_web_sm",
        max_attributewords: int = 5,
    ):
        """
        Args:
            tokenizer: BERT tokenizer used by the text encoder.
            spacy_model: spaCy model name used for part-of-speech parsing.
            max_attributewords: Maximum number of attribute words retained per text.
        """

        self.tokenizer = tokenizer
        self.max_attributewords = max_attributewords

        try:
            self.nlp = spacy.load(spacy_model)
            self.use_spacy = True
        except OSError:
            print(
                f"[AttributewordExtractor] spaCy model '{spacy_model}' was not found. "
                f"Falling back to a heuristic extractor. To enable spaCy parsing, run: "
                f"python -m spacy download {spacy_model}"
            )
            self.nlp = None
            self.use_spacy = False

    def extract_attributeword_positions(
        self,
        texts: List[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract attribute-word positions after BERT tokenization.

        Args:
            texts: Original referring expressions with length B.
            input_ids: BERT token ids with shape (B, L).
            attention_mask: BERT attention mask with shape (B, L).

        Returns:
            attributeword_positions: Token positions with shape
                (B, max_attributewords).
            attributeword_mask: Valid-position mask with shape
                (B, max_attributewords).
        """

        batch_size = len(texts)
        device = input_ids.device

        attributeword_positions = torch.zeros(
            batch_size,
            self.max_attributewords,
            dtype=torch.long,
            device=device,
        )
        attributeword_mask = torch.zeros(
            batch_size,
            self.max_attributewords,
            dtype=torch.bool,
            device=device,
        )

        for batch_idx, text in enumerate(texts):
            if self.use_spacy:
                attributewords = self._extract_with_spacy(text)
            else:
                attributewords = self._extract_with_heuristics(text)

            positions = self._map_words_to_token_positions(
                attributewords=attributewords,
                input_ids=input_ids[batch_idx],
                attention_mask=attention_mask[batch_idx],
            )

            num_positions = min(len(positions), self.max_attributewords)
            if num_positions > 0:
                attributeword_positions[batch_idx, :num_positions] = torch.tensor(
                    positions[:num_positions],
                    dtype=torch.long,
                    device=device,
                )
                attributeword_mask[batch_idx, :num_positions] = True

        return attributeword_positions, attributeword_mask

    def _extract_with_spacy(self, text: str) -> List[str]:
        """Extract attribute words using spaCy POS tags and dependency labels."""

        doc = self.nlp(str(text))
        attributewords = []

        for token in doc:
            token_text = token.text.strip()
            token_lower = token_text.lower()

            if token.pos_ == "ADJ":
                attributewords.append(token_text)
            elif token.dep_ == "amod":
                attributewords.append(token_text)
            elif token.dep_ == "nummod":
                attributewords.append(token_text)
            elif token.dep_ == "advmod" and token.head.pos_ == "ADJ":
                attributewords.append(token_text)
            elif token.pos_ in {"ADJ", "NUM"} and token.dep_ in {"amod", "nummod", "compound"}:
                attributewords.append(token_text)
            elif token.tag_ in {"VBN", "VBG"} and token.dep_ == "amod":
                attributewords.append(token_text)
            elif token_lower in self.SPATIAL_WORDS:
                attributewords.append(token_text)
            elif token.dep_ == "pobj" and token.head.pos_ == "ADP" and token_lower in self.SPATIAL_WORDS:
                attributewords.append(token_text)
            elif token.dep_ == "compound" and token_lower in self.SPATIAL_WORDS:
                attributewords.append(token_text)

        return self._deduplicate(attributewords)[: self.max_attributewords]

    def _extract_with_heuristics(self, text: str) -> List[str]:
        """
        Extract attribute words with a lightweight rule-based fallback.

        This fallback is used when the requested spaCy model is unavailable.
        """

        words = str(text).lower().split()
        attributewords = []

        for idx, word in enumerate(words):
            clean_word = self._clean_word(word)

            if not clean_word or clean_word in self.STOP_WORDS:
                continue

            if clean_word in self.COMMON_ATTRIBUTE_WORDS:
                attributewords.append(clean_word)
                continue

            if any(clean_word.endswith(suffix) for suffix in self.ADJECTIVE_SUFFIXES):
                attributewords.append(clean_word)
                continue

            if clean_word.isdigit():
                attributewords.append(clean_word)
                continue

            if idx < len(words) - 1:
                next_word = self._clean_word(words[idx + 1])
                if (
                    len(clean_word) > 3
                    and any(next_word.endswith(suffix) for suffix in self.NOUN_SUFFIXES)
                ):
                    attributewords.append(clean_word)

        return self._deduplicate(attributewords)[: self.max_attributewords]

    def _map_words_to_token_positions(
        self,
        attributewords: List[str],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[int]:
        """Map extracted attribute words to their first matching BERT token index."""

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.detach().cpu().tolist())

        valid_length = len(tokens)
        if attention_mask is not None:
            valid_length = int(attention_mask.detach().cpu().sum().item())
        tokens = tokens[:valid_length]

        positions = []
        for word in attributewords:
            word_tokens = self.tokenizer.tokenize(word.lower())
            if not word_tokens:
                continue

            position = self._find_token_position(tokens, word_tokens, word)
            if position is not None:
                positions.append(position)

        return self._deduplicate_positions(positions)

    @staticmethod
    def _find_token_position(
        tokens: List[str],
        word_tokens: List[str],
        original_word: str,
    ) -> Optional[int]:
        """Find the first token position that matches a word or its WordPiece form."""

        for idx in range(len(tokens) - len(word_tokens) + 1):
            if tokens[idx : idx + len(word_tokens)] == word_tokens:
                return idx

        if len(word_tokens) == 1:
            target_token = word_tokens[0]
            for idx, token in enumerate(tokens):
                if target_token in token or token in target_token:
                    return idx

        original_lower = original_word.lower()
        for idx, token in enumerate(tokens):
            clean_token = token.replace("##", "").lower()
            if (
                clean_token in original_lower
                or original_lower in clean_token
                or (
                    len(clean_token) > 3
                    and len(original_lower) > 3
                    and clean_token[:3] == original_lower[:3]
                )
            ):
                return idx

        return None

    def extract_attributeword_features(
        self,
        text_features: torch.Tensor,
        attributeword_positions: torch.Tensor,
        attributeword_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather BERT features corresponding to extracted attribute-word positions.

        Args:
            text_features: BERT token features with shape (B, L, C).
            attributeword_positions: Token positions with shape
                (B, max_attributewords).
            attributeword_mask: Valid-position mask with shape
                (B, max_attributewords).

        Returns:
            Attribute-word features with shape (B, max_attributewords, C).
            Invalid positions are filled with zeros.
        """

        batch_size, seq_len, hidden_dim = text_features.shape
        device = text_features.device

        safe_positions = attributeword_positions.clamp(min=0, max=seq_len - 1)
        gather_index = safe_positions.unsqueeze(-1).expand(-1, -1, hidden_dim)
        features = torch.gather(text_features, dim=1, index=gather_index)

        valid_mask = attributeword_mask.to(device=device).unsqueeze(-1)
        features = features * valid_mask.to(dtype=features.dtype)

        return features

    @staticmethod
    def _clean_word(word: str) -> str:
        """Remove punctuation and keep alphabetic or numeric characters."""
        return word.strip(string.punctuation).lower()

    @staticmethod
    def _deduplicate(words: List[str]) -> List[str]:
        """Deduplicate words while preserving their original order."""
        seen = set()
        unique_words = []

        for word in words:
            key = word.lower()
            if key not in seen:
                seen.add(key)
                unique_words.append(word)

        return unique_words

    @staticmethod
    def _deduplicate_positions(positions: List[int]) -> List[int]:
        """Deduplicate token positions while preserving order."""
        seen = set()
        unique_positions = []

        for position in positions:
            if position not in seen:
                seen.add(position)
                unique_positions.append(position)

        return unique_positions

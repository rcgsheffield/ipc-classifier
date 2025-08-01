from typing import Dict, Optional

import torch
from torch import nn
from transformers import AutoModel


class HierarchicalIPCClassifier(nn.Module):
    """Hierarchical classifier with parent-aware predictions for section and class only."""

    def __init__(
            self,
            model_name: str,
            n_section: int,
            n_class: int,
            dropout: float = 0.3,
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Shared feature extractor
        self.dropout = nn.Dropout(dropout)

        # Level-specific classifiers with proper initialization
        self.section_classifier = nn.Linear(hidden_size, n_section)
        nn.init.xavier_uniform_(self.section_classifier.weight)
        nn.init.constant_(
            self.section_classifier.bias, -2.0
        )  # Negative bias for multi-label

        self.class_classifier = nn.Linear(hidden_size, n_class)
        nn.init.xavier_uniform_(self.class_classifier.weight)
        nn.init.constant_(self.class_classifier.bias, -2.0)

        self.n_section = n_section
        self.n_class = n_class

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            parent_constraints: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional parent constraints.

        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            parent_constraints: Dict with boolean masks for valid children at each level

        Returns:
            Dict with logits for each level
        """
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # Section predictions (no constraints)
        section_logits = self.section_classifier(pooled_output)

        # Class predictions (optionally constrained by section)
        class_logits = self.class_classifier(pooled_output)
        if parent_constraints and "class_mask" in parent_constraints:
            # Apply mask by setting invalid classes to very negative value
            class_logits = class_logits.masked_fill(
                ~parent_constraints["class_mask"], -10000.0
            )

        return {
            "section_logits": section_logits,
            "class_logits": class_logits,
        }

    def predict_hierarchical(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            processor: "IPCDataProcessor",
            confidence_threshold: float = 0.3,
            use_soft_masking: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Make hierarchical predictions with automatic parent-based constraints.

        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            processor: Data processor with hierarchy mappings
            confidence_threshold: Minimum confidence for parent predictions
            use_soft_masking: If True, use soft masking (0.1 for invalid). If False, hard masking (-10000)

        Returns:
            Dict with constrained logits for each level
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Get BERT embeddings once
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)

        # Get section predictions (unconstrained)
        section_logits = self.section_classifier(pooled_output)
        section_probs = torch.sigmoid(
            section_logits
        ).detach()  # Detach to prevent gradient flow through constraints

        # Create class constraints based on confident section predictions
        class_masks = torch.ones(
            (batch_size, self.n_class), dtype=torch.bool, device=device
        )

        for i in range(batch_size):
            # Get sections with confidence above threshold
            confident_sections_mask = section_probs[i] > confidence_threshold

            if confident_sections_mask.any():
                confident_sections = []
                for j in confident_sections_mask.nonzero().squeeze(-1):
                    section_code = processor.encoders["section"].classes_[j.item()]
                    confident_sections.append(section_code)

                # Get valid classes for these sections
                valid_classes = processor.get_valid_children(
                    "section", confident_sections
                )

                if valid_classes:  # Only apply mask if we have valid classes
                    class_masks[i] = False  # Reset to False
                    for j, class_code in enumerate(
                            processor.encoders["class"].classes_
                    ):
                        if class_code in valid_classes:
                            class_masks[i, j] = True
            else:
                class_masks[i] = (
                    True  # if no valid classes from sections, allow all classes
                )

        # Get class predictions with constraints
        class_logits = self.class_classifier(pooled_output)
        if use_soft_masking:
            soft_mask = torch.where(class_masks, 1.0, 0.1)
            class_logits = class_logits * soft_mask
        else:
            class_logits = class_logits.masked_fill(~class_masks, -10000.0)

        return {
            "section_logits": section_logits,
            "class_logits": class_logits,
        }
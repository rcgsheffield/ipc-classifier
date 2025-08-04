from typing import Dict, List

import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset


class IPCDataProcessor:
    """Simple processor for hierarchical IPC data with hint support."""

    def __init__(self, ipc_metadata_path: str):
        # Load only hierarchy_level == 2 data
        self.df_ipc = pd.read_csv(ipc_metadata_path)
        self.df_ipc = self.df_ipc[self.df_ipc['hierarchy_level'] == 2].copy()

        self.hierarchy_mappings = self._build_hierarchy_mappings()
        self.hierarchy_texts = self._build_hierarchy_texts()
        self.encoders = {}

    def _build_hierarchy_mappings(self) -> Dict[str, List[str]]:
        """Build parent-to-child mappings from IPC metadata."""
        mappings = {"section_to_class": {}}

        for _, row in self.df_ipc.iterrows():
            section = row["section_code"] if pd.notna(row["section_code"]) else None
            class_code = row["class_code"] if pd.notna(row["class_code"]) else None

            if section and class_code:
                if section not in mappings["section_to_class"]:
                    mappings["section_to_class"][section] = []
                if class_code not in mappings["section_to_class"][section]:
                    mappings["section_to_class"][section].append(class_code)

        return mappings

    def _build_hierarchy_texts(self) -> Dict[str, Dict[str, str]]:
        """Extract IPC descriptions from full_hierarchical_description."""
        texts = {"section": {}, "class": {}}

        for _, row in self.df_ipc.iterrows():
            desc = row.get("full_hierarchical_description", "")
            if not desc or pd.isna(desc):
                continue

            parts = desc.split(" | ")

            section_code = row["section_code"]
            class_code = row["class_code"]

            if len(parts) >= 1 and section_code:
                section_part = parts[0]
                if f"[{section_code}]" in section_part:
                    section_desc = section_part.split(f"[{section_code}]", 1)[1].strip()
                    texts["section"][section_code] = section_desc

            if len(parts) >= 2 and class_code:
                cleaned_parts = []
                for part in parts:
                    if "]" in part:
                        cleaned_part = part.split("]", 1)[1].strip()
                        cleaned_parts.append(cleaned_part)
                    else:
                        cleaned_parts.append(part)
                texts["class"][class_code] = " | ".join(cleaned_parts)

        return texts

    @staticmethod
    def parse_labels(value):
        """Parse string representation of list to actual list and deduplicate."""
        if isinstance(value, str):
            value = value.strip("[]")
            if value:
                items = [item.strip().strip("'\"") for item in value.split(",")]
                return list(set(items))
        elif isinstance(value, list):
            return list(set(value))
        return []

    def get_all_level_hints(self, section_codes: List[str], class_codes: List[str]) -> str:
        """Get combined hints from all hierarchy levels."""
        hints = []

        for code in section_codes:
            if code in self.hierarchy_texts["section"]:
                hints.append(self.hierarchy_texts['section'][code])

        for code in class_codes:
            if code in self.hierarchy_texts["class"]:
                hints.append(self.hierarchy_texts["class"][code])

        return " | ".join(hints) if hints else ""

    def process_dataframe(self, df: pd.DataFrame, add_hints: bool = False) -> pd.DataFrame:
        """Process dataframe to extract and clean labels, optionally add hints."""
        df = df.copy()

        for col in ["section", "class"]:
            if col in df.columns:
                df[col] = df[col].apply(self.parse_labels)
                df[col] = df[col].apply(lambda x: list(set(x)) if x else [])

        if add_hints and "section" in df.columns and "class" in df.columns:
            text_columns = ['appln_abstract', 'abstract', 'full_description', 'description', 'text', 'title']
            text_col = None
            for col in text_columns:
                if col in df.columns:
                    text_col = col
                    break

            if text_col:
                df[f"{text_col}_with_hint"] = df.apply(
                    lambda row: (
                        str(row[text_col]) + " [SEP] " +
                        self.get_all_level_hints(row["section"], row["class"])
                    ).strip(),
                    axis=1
                )

        return df

    def fit_encoders(self, train_df: pd.DataFrame) -> Dict[str, MultiLabelBinarizer]:
        """Fit label encoders on training data."""
        self.encoders = {}

        for level in ["section", "class"]:
            self.encoders[level] = MultiLabelBinarizer()
            self.encoders[level].fit(train_df[level])
            print(f"{level}: {len(self.encoders[level].classes_)} unique labels")

        return self.encoders

    def get_valid_children(self, parent_level: str, parent_codes: List[str]) -> set:
        """Get all valid children codes given parent predictions."""
        if parent_level == "section":
            mapping = self.hierarchy_mappings["section_to_class"]
        else:
            return set()

        valid_children = set()
        for parent_code in parent_codes:
            if parent_code in mapping:
                valid_children.update(mapping[parent_code])

        return valid_children


class IPCDataset(Dataset):
    """Dataset for hierarchical IPC classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        encoders: Dict,
        max_length: int = 512,
        use_hints: bool = False
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.encoders = encoders
        self.max_length = max_length
        self.use_hints = use_hints

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = None
        text_columns = ['appln_abstract', 'abstract', 'full_description', 'description', 'text', 'title']
        found_column = None

        for col in text_columns:
            if col in self.df.columns:
                try:
                    val = row[col]
                    if pd.notna(val) and str(val).strip():
                        text = val
                        found_column = col
                        break
                except KeyError:
                    continue

        if text is None:
            available_cols = list(self.df.columns)
            raise KeyError(
                f"No text column found at index {idx}. "
                f"Looked for: {text_columns}. "
                f"Available columns: {available_cols}"
            )

        if self.use_hints and found_column:
            hint_column = f"{found_column}_with_hint"
            if hint_column in self.df.columns:
                try:
                    hint_val = row[hint_column]
                    if pd.notna(hint_val):
                        text = hint_val
                except KeyError:
                    pass

        encoding = self.tokenizer(
            str(text),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = {}
        for level in ["section", "class"]:
            if level in self.df.columns:
                try:
                    label_val = row[level]
                    if pd.notna(label_val):
                        if isinstance(label_val, str):
                            label_list = [label_val]
                        else:
                            label_list = label_val

                        labels[f"{level}_labels"] = torch.FloatTensor(
                            self.encoders[level].transform([label_list])[0]
                        )
                    else:
                        labels[f"{level}_labels"] = torch.zeros(
                            len(self.encoders[level].classes_),
                            dtype=torch.float
                        )
                except (KeyError, Exception):
                    labels[f"{level}_labels"] = torch.zeros(
                        len(self.encoders[level].classes_),
                        dtype=torch.float
                    )
            else:
                labels[f"{level}_labels"] = torch.zeros(
                    len(self.encoders[level].classes_),
                    dtype=torch.float
                )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            **labels,
        }

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

    def _build_hierarchy_mappings(self) -> Dict[str, Dict[str, List[str]]]:
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
                
            # Parse the description format: [A] HUMAN NECESSITIES | [A01] AGRICULTURE...
            parts = desc.split(" | ")
            
            section_code = row["section_code"]
            class_code = row["class_code"]
            
            # Extract section description
            if len(parts) >= 1 and section_code:
                section_part = parts[0]
                # Extract text after the code in brackets
                if f"[{section_code}]" in section_part:
                    section_desc = section_part.split(f"[{section_code}]", 1)[1].strip()
                    texts["section"][section_code] = section_desc
            
            # For class, combine section and class descriptions
            if len(parts) >= 2 and class_code:
                # Full description includes both section and class
                texts["class"][class_code] = desc
                    
        return texts

    @staticmethod
    def parse_labels(value):
        """Parse string representation of list to actual list and deduplicate."""
        if isinstance(value, str):
            value = value.strip("[]")
            if value:
                items = [item.strip().strip("'\"") for item in value.split(",")]
                return list(set(items))  # Remove duplicates
        elif isinstance(value, list):
            return list(set(value))  # Remove duplicates from existing lists
        return []

    def get_all_level_hints(self, section_codes: List[str], class_codes: List[str]) -> str:
        """Get combined hints from all hierarchy levels."""
        hints = []
        
        # Add section hints
        for code in section_codes:
            if code in self.hierarchy_texts["section"]:
                hints.append(f"[{code}] {self.hierarchy_texts['section'][code]}")
        
        # Add class hints (which already include full hierarchy)
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
                # Remove duplicates and ensure list
                df[col] = df[col].apply(lambda x: list(set(x)) if x else [])

        # Add hints if requested
        if add_hints and "section" in df.columns and "class" in df.columns:
            df["appln_abstract_with_hint"] = df.apply(
                lambda row: (
                    row["appln_abstract"] + " [SEP] " + 
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
        
        # Use hint-enhanced text if available and requested
        if self.use_hints and "appln_abstract_with_hint" in row:
            text = row["appln_abstract_with_hint"]
        else:
            text = row["appln_abstract"]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Encode labels
        labels = {}
        for level in ["section", "class"]:
            labels[f"{level}_labels"] = torch.FloatTensor(
                self.encoders[level].transform([row[level]])[0]
            )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            **labels,
        }

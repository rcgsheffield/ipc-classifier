import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Dict, List, Tuple


class IPCDataProcessor:
    """Simple processor for hierarchical IPC data."""
    
    def __init__(self, ipc_metadata_path: str):
        self.df_ipc = pd.read_csv(ipc_metadata_path)
        self.hierarchy_mappings = self._build_hierarchy_mappings()
        self.encoders = {}
        
    def _build_hierarchy_mappings(self) -> Dict[str, Dict[str, List[str]]]:
        """Build parent-to-child mappings from IPC metadata."""
        mappings = {
            'section_to_class': {},
            'class_to_subclass': {}
        }
        
        for _, row in self.df_ipc.iterrows():
            section = row['section_code'] if pd.notna(row['section_code']) else None
            class_code = row['class_code'] if pd.notna(row['class_code']) else None
            subclass_code = row['subclass_code'] if pd.notna(row['subclass_code']) else None
            
            if section and class_code:
                if section not in mappings['section_to_class']:
                    mappings['section_to_class'][section] = []
                if class_code not in mappings['section_to_class'][section]:
                    mappings['section_to_class'][section].append(class_code)
                    
            if class_code and subclass_code:
                if class_code not in mappings['class_to_subclass']:
                    mappings['class_to_subclass'][class_code] = []
                if subclass_code not in mappings['class_to_subclass'][class_code]:
                    mappings['class_to_subclass'][class_code].append(subclass_code)
                    
        return mappings
    
    @staticmethod
    def parse_labels(value):
        """Parse string representation of list to actual list."""
        if isinstance(value, str):
            value = value.strip('[]')
            if value:
                items = [item.strip().strip("'\"") for item in value.split(',')]
                return list(set(items))  # Remove duplicates here too
        elif isinstance(value, list):
            return list(set(value))  # Remove duplicates from existing lists
        return []
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process dataframe to extract and clean labels."""
        df = df.copy()
        
        for col in ['section', 'class', 'subclass']:
            if col in df.columns:
                df[col] = df[col].apply(self.parse_labels)
                # Remove duplicates and ensure list
                df[col] = df[col].apply(lambda x: list(set(x)) if x else [])
                
        return df
    
    def fit_encoders(self, train_df: pd.DataFrame) -> Dict[str, MultiLabelBinarizer]:
        """Fit label encoders on training data."""
        self.encoders = {}
        
        for level in ['section', 'class', 'subclass']:
            self.encoders[level] = MultiLabelBinarizer()
            self.encoders[level].fit(train_df[level])
            print(f"{level}: {len(self.encoders[level].classes_)} unique labels")
            
        return self.encoders
    
    def get_valid_children(self, parent_level: str, parent_codes: List[str]) -> set:
        """Get all valid children codes given parent predictions."""
        if parent_level == 'section':
            mapping = self.hierarchy_mappings['section_to_class']
        elif parent_level == 'class':
            mapping = self.hierarchy_mappings['class_to_subclass']
        else:
            return set()
        
        valid_children = set()
        for parent_code in parent_codes:
            if parent_code in mapping:
                valid_children.update(mapping[parent_code])
                
        return valid_children


class IPCDataset(Dataset):
    """Simple dataset for hierarchical IPC classification."""
    
    def __init__(self, df: pd.DataFrame, tokenizer, encoders: Dict, max_length: int = 512):
        self.df = df
        self.tokenizer = tokenizer
        self.encoders = encoders
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['appln_abstract']
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Encode labels
        labels = {}
        for level in ['section', 'class', 'subclass']:
            labels[f'{level}_labels'] = torch.FloatTensor(
                self.encoders[level].transform([row[level]])[0]
            )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            **labels
        }

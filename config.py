from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Config:
    # Model settings
    model_name: str = 'allenai/scibert_scivocab_uncased'
    max_length: int = 512
    dropout: float = 0.3
    
    # Training settings
    batch_size: int = 32
    learning_rate: float = 1e-5
    epochs: int = 20
    warmup_ratio: float = 0.1
    use_fp16: bool = True  # Enable mixed precision training for faster training (set True if no NaN issues)
    
    # Data paths
    train_path: str = 'data/train.csv'
    val_path: str = 'data/val.csv'
    test_path: str = 'data/test.csv'
    ipc_metadata_path: str = 'data/full_ipc_combined.csv'
    
    # Output settings
    checkpoint_dir: str = './checkpoints'
    output_path: str = 'predictions.csv'
    
    # Hierarchical settings
    parent_confidence_threshold: float = 0.2  # Min confidence to consider parent prediction
    prediction_threshold: Dict[str, float] = None

    # Hint settings:
    use_hints: bool = False

    def __post_init__(self):
        if self.prediction_threshold is None:
            self.prediction_threshold = {
                'section': 0.7,
                'class': 0.6,
            }

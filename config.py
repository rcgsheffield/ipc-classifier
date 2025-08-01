from dataclasses import dataclass
from typing import Dict


@dataclass
class Config:
    # Model settings
    model_name: str = "allenai/scibert_scivocab_uncased"  # Uncomment to try another model
    # model_name: str = "sentence-transformers/all-mpnet-base-v2"
    max_length: int = 512
    dropout: float = 0.3

    # Training settings
    batch_size: int = 32
    learning_rate: float = 1e-5
    epochs: int = 20
    warmup_ratio: float = 0.1
    use_fp16: bool = True  # Enable mixed precision training for faster training
    use_class_weights: bool = True  # Using inverse frequency weights for imbalanced classes

    # Data paths
    train_path: str = "data/train.csv"
    val_path: str = "data/val.csv"
    test_path: str = "data/test.csv"
    ipc_metadata_path: str = "data/full_ipc_combined.csv"

    # Output settings
    checkpoint_dir: str = "./checkpoints"
    output_path: str = "predictions.csv" # Base name for output files

    # Hierarchical settings
    parent_confidence_threshold: float = 0.2
    prediction_threshold: Dict[str, float] = None

    # Hint settings
    use_hints: bool = False

    # Threshold optimization settings
    optimize_thresholds: bool = True
    threshold_search_range: tuple = (0.1, 0.7)
    threshold_search_step: float = 0.1

    # Top-k settings
    use_top_k: bool = True  # Enable top-k predictions alongside threshold
    top_k: Dict[str, int] = None  # Will be set based on training data

    def __post_init__(self):
        if self.prediction_threshold is None:
            self.prediction_threshold = {"section": 0.5, "class": 0.2}

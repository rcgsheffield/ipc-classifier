# Hierarchical IPC Patent Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![isort](https://img.shields.io/badge/imports-isort-%231674b1)](https://pycqa.github.io/isort/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

This repository contains a PyTorch implementation of hierarchical multi-label patent classification by fine-tuning sentence transformers
(BERT-based models). This codebase classifies patent abstracts into the International Patent Classification (IPC) hierarchy with
parent-child constraints between classification levels.

## Features

- Hierarchical Classification: Three-level classification (Section → Class → Subclass) with automatic parent-child constraint enforcement
- Multi-label Support: Patents can belong to multiple categories at each level
- SciBERT Integration: Uses domain-specific SciBERT model for better patent text understanding
- Mixed Precision Training: FP16 support for faster training on compatible GPUs
- Soft/Hard Masking: Flexible constraint application during training and inference
- valuation: Multi-level metrics including F1, precision, and recall

## Architecture

The model uses a hierarchical approach where:

1. Section predictions are made independently
2. Class predictions are constrained by confident section predictions
3. Subclass predictions are constrained by confident class predictions

This ensures taxonomically valid predictions that respect the IPC hierarchy.

## Strucutre

```
.
├── config.py                 # Configuration settings
├── data_processing.py        # Data loading and preprocessing
├── model.py                  # Hierarchical classifier architecture
├── train.py                  # Training script
├── predict.py                # Inference script
├── training_job.sh           # SLURM job script for training
├── prediction_job.sh         # SLURM job script for prediction
├── pyproject.toml            # Project metadata and dependencies
├── conda-env.yaml            # Conda environment specification
├── .pre-commit-config.yaml   # Pre-commit hooks configuration
└── data/
    ├── train.csv            # Training data
    ├── val.csv              # Validation data
    ├── test.csv             # Test data
    └── full_ipc_combined.csv # IPC hierarchy metadata
 ```

 ## Getting Started

 ### Prerequisites

 - Python 3.10+
 - PyTorch 2.0+
 - Transformers 4.53+
 - CUDA-capable GPU

 ## Installation

 1. Clone the repository

 ```
git clone https://github.com/rcgsheffield/ipc-classifier.git
cd ipc-classifier
 ```

 2. Create a conda environment

 ```
conda env create -f conda-env.yaml
conda activate ai-innovation
```

3. Install additional dependencies

```
pip install transformers datasets
```

4. Install optional developer dependencies:

```
pip install -e .[dev]
pre-commit install
```

## Data Format
- The CSV files (train.csv, test.csv, val.csv) should have the following columns:

- appln_abstract: Patent abstract text
- section: List of section codes (e.g., "['A', 'B']")
- class: List of class codes (e.g., "['A01', 'B32']")
- subclass: List of subclass codes (e.g., "['A01B', 'B32B']")

The IPC metadata file (full_ipc_combined.csv) should contain:

- section_code: Section identifier
- class_code: Class identifier
- subclass_code: Subclass identifier

## Training

### Local training:

```
python train.py
```
### Submit a job on the HPC

```
sbatch training_job.sh
```

### Configuration
Key parameters in config.py:

- model_name: Pre-trained model (default: "allenai/scibert_scivocab_uncased")
- batch_size: Training batch size (default: 32)
- learning_rate: Learning rate (default: 1e-5)
- epochs: Number of training epochs (default: 20)
- parent_confidence_threshold: Minimum confidence for parent constraints (default: 0.2)
- prediction_threshold: Level-specific prediction thresholds
- use_fp16: Enable mixed precision training (default: True)

## Inference

```
python predict.py
```

### Submit a job on the HPC

```
sbatch prediction_job.sh
```

The script will:

1. Load the best saved model from `./checkpoints/`
2. Process test data with hierarchical constraints
3. Save predictions to predictions.csv

### Output Format
The predictions CSV contains:

- index: Sample index
- abstract: Patent abstract text
- For each level (section/class/subclass):

    - true_{level}: Ground truth labels
    - {level}_predicted: Predicted labels
    - {level}_probabilities: Prediction confidence scores


## Code Quality
- This project uses pre-commit hooks for code quality:

- Black: Code formatting
- isort: Import sorting
-Pylint: Code linting

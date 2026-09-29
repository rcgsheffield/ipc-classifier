# Hierarchical IPC Patent Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![isort](https://img.shields.io/badge/imports-isort-%231674b1)](https://pycqa.github.io/isort/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)
[![DOI](https://img.shields.io/badge/doi-10.15131%2Fshef.data.34014453.v1-blue.svg?style=flat&labelColor=whitesmoke&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAB8AAAAfCAYAAAAfrhY5AAAJsklEQVR42qWXd1DTaRrHf%2BiB2Hdt5zhrAUKz4IKEYu9IGiGFFJJQ0gkJCAKiWFDWBRdFhCQUF3UVdeVcRQEBxUI3yY9iEnQHb3bdW1fPubnyz%2F11M7lvEHfOQee2ZOYzPyDv%2B3yf9%2Fk95YX4fx%2BltfUt08GcFEuPR4U9hDDZ%2FVngIlhb%2FSiI6InkTgLzgDcgfvtnovhH4BzoVlrbwr55QnhCtBW4QHXnFrZbPBaQoBh4%2FSYH2EnpBEtqcDMVzB93wA%2F8AFwa23XFGcc8CkT3mxz%2BfXWtq9T9IQlLIXYEuHojudb%2BCM7Hgdq8ydi%2FAHiBXyY%2BLjwFlAEnS6Jnar%2FvnQVhvdzasad0eKvWZKe8hvDB2ofLZ%2FZEcWsh%2BhyIuyO5Bxs2iZIE4nRv7NWAb0EO8AC%2FWPxjYAWuOEX2MSXZVgPxzmRL3xKz3ScGpx6p6QnOx4mDIFqO0w6Q4fEhO5IzwxlSwyD2FYHzwAW%2BAZ4fEsf74gCumykwNHskLM7taQxLYjjIyy8MUtraGhTWdkfhkFJqtvuVl%2F9l2ZquDfEyrH8B0W06nnpH3JtIyRGpH1iJ6SfxDIHjRXHJmdQjLpfHeN54gnfFx4W9QRnovx%2FN20aXZeTD2J84hn3%2BqoF2Tqr14VqTPUCIcP%2B5%2Fly4qC%2BUL3sYxSvNj1NwsVYPsWdMUfomsdkYm3Tj0nbV0N1wRKwFe1MgKACDIBdMAhPE%2FwicwNWxll8Ag40w%2BFfhibJkGHmutjYeQ8gVlaN%2BjO51nDysa9TwNUFMqaGbKdRJZFfOJSp6mkRKsv0rRIpEVWjAvyFkxNOEpwvcAVPfEe%2Bl8ojeNTx3nXLBcWRrYGxSRjDEk0VlpxYrbe1ZmaQ5xuT0u3r%2B2qe5j0J5uytiZPGsRL2Jm32AldpxPUNJ3jmmsN4x62z1cXrbedXBQf2yvIFCeZrtyicZZG2U2nrrBJzYorI2EXLrvTfCSB43s41PKEvbZDEfQby6L4JTj%2FfIwam%2B4%2BwucBu%2BDgNK05Nle1rSt9HvR%2FKPC4U6LTfvUIaip1mjIa8fPzykii23h2eanT57zQ7fsyYH5QjywwlooAUcAdOh5QumgTHx6aAO7%2FL52eaQNEShrxfhL6albEDmfhGflrsT4tps8gTHNOJbeDeBlt0WJWDHSgxs6cW6lQqyg1FpD5ZVDfhn1HYFF1y4Eiaqa18pQf3zzYMBhcanlBjYfgWNayAf%2FASOgklu8bmgD7hADrk4cRlOL7NSOewEcbqSmaivT33QuFdHXj5sdvjlN5yMDrAECmdgDWG2L8P%2BAKLs9ZLZ7dJda%2BB4Xl84t7QvnKfvpXJv9obz2KgK8dXyqISyV0sXGZ0U47hOA%2FAiigbEMECJxC9aoKp86re5O5prxOlHkcksutSQJzxZRlPZmrOKhsQBF5zEZKybUC0vVjG8PqOnhOq46qyDTDnj5gZBriWCk4DvXrudQnXQmnXblebhAC2cCB6zIbM4PYgGl0elPSgIf3iFEA21aLdHYLHUQuVkpgi02SxFdrG862Y8ymYGMvXDzUmiX8DS5vKZyZlGmsSgQqfLub5RyLNS4zfDiZc9Edzh%2FtCE%2BX8j9k%2FqWB071rcZyMImne1SLkL4GRw4UPHMV3jjwEYpPG5uW5fAEot0aTSJnsGAwHJi2nvF1Y5OIqWziVCQd5NT7t6Q8guOSpgS%2Fa1dSRn8JGGaCD3BPXDyQRG4Bqhu8XrgAp0yy8DMSvvyVXDgJcJTcr1wQ2BvFKf65jqhvmxXUuDpGBlRvV36XvGjQzLi8KAKT2lYOnmxQPGorURSV0NhyTIuIyqOmKTMhQ%2BieEsgOgpc4KBbfDM4B3SIgFljvfHF6cef7qpyLBXAiQcXvg5l3Iunp%2FWv4dH6qFziO%2BL9PbrimQ9RY6MQphEfGUpOmma7KkGzuS8sPUFnCtIYcKCaI9EXo4HlQLgGrBjbiK5EqMj2AKWt9QWcIFMtnVvQVDQV9lXJJqdPVtUQpbh6gCI2Ov1nvZts7yYdsnvRgxiWFOtNJcOMVLn1vgptVi6qrNiFOfEjHCDB3J%2BHDLqUB77YgQGwX%2Fb1eYna3hGKdlqJKIyiE4nSbV8VFgxmxR4b5mVkkeUhMgs5YTi4ja2XZ009xJRHdkfwMi%2BfocaancuO7h%2FMlcLOa0V%2FSw6Dq47CumRQAKhgbOP8t%2BMTjuxjJGhXCY6XpmDDFqWlVYbQ1aDJ5Cptdw4oLbf3Ck%2BdWkVP0LpH7s9XLPXI%2FQX8ws%2Bj2In63IcRvOOo%2BTTjiN%2BlssfRsanW%2B3REVKoavBOAPTXABW4AL7e4NygHdpAKBscmlDh9Jysp4wxbnUNna3L3xBvyE1jyrGIkUHaqQMuxhHElV6oj1picvgL1QEuS5PyZTEaivqh5vUCKJqOuIgPFGESns8kyFk7%2FDxyima3cYxi%2FYOQCj%2F%2B9Ms2Ll%2Bhn4FmKnl7JkGXQGDKDAz9rUGL1TIlBpuJr9Be2JjK6qPzyDg495UxXYF7JY1qKimw9jWjF0iV6DRIqE%2B%2FeWG0J2ofmZTk0mLYVd4GLiFCOoKR0Cg727tWq981InYynvCuKW43aXgEjofVbxIqrm0VL76zlH3gQzWP3R3Bv9oXxclrlO7VVtgBRpSP4hMFWJ8BrUSBCJXC07l40X4jWuvtc42ofNCxtlX2JH6bdeojXgTh5TxOBKEyY5wvBE%2BACh8BtOPNPkApjoxi5h%2B%2FFMQQNpWvZaMH7MKFu5Ax8HoCQdmGkJrtnOiLHwD3uS5y8%2F2xTSDrE%2F4PT1yqtt6vGe8ldMBVMEPd6KwqiYECHDlfbvzphcWP%2BJiZuL5swoWQYlS%2Br7Yu5mNUiGD2retxBi9fl6RDGn4Ti9B1oyYy%2BMP5G87D%2FCpRlvdnuy0PY6RC8BzTA40NXqckQ9TaOUDywkYsudxJzPgyDoAWn%2BB6nEFbaVxxC6UXjJiuDkW9TWq7uRBOJocky9iMfUhGpv%2FdQuVVIuGjYqACbXf8aa%2BPeYNIHZsM7l4s5gAQuUAzRUoT51hnH3EWofXf2vkD5HJJ33vwE%2FaEWp36GHr6GpMaH4AAPuqM5eabH%2FhfG9zcCz4nN6cPinuAw6IHwtvyB%2FdO1toZciBaPh25U0ducR2PI3Zl7mokyLWKkSnEDOg1x5fCsJE9EKhH7HwFNhWMGMS7%2BqxyYsbHHRUDUH4I%2FAheQY7wujJNnFUH4KdCju83riuQeHU9WEqNzjsJFuF%2FdTDAZ%2FK7%2F1WaAU%2BAWymT59pVMT4g2AxcwNa0XEBDdBDpAPvgDIH73R25teeuAF5ime2Ul0OUIiG4GpSAEJeYW9wDTf43wfwHgHLKJoPznkwAAAABJRU5ErkJggg%3D%3D)](https://doi.org/10.15131/shef.data.34014453.v1)

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

import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import Config
from data_processing import IPCDataProcessor, IPCDataset
from model import HierarchicalIPCClassifier


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, level: str) -> dict:
    """Compute flat metrics for a single level."""
    return {
        f"{level}_f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        f"{level}_f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{level}_precision_micro": precision_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        f"{level}_precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        f"{level}_recall_micro": recall_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        f"{level}_recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


def compute_hierarchical_metrics(
        y_true_section: np.ndarray,
        y_true_class: np.ndarray,
        y_pred_section: np.ndarray,
        y_pred_class: np.ndarray,
        processor: IPCDataProcessor,
        partial_credit: float = 0.5
) -> dict:
    """
    Compute hierarchical metrics that give partial credit for correct parent predictions.

    Args:
        y_true_section: True section labels (binary matrix)
        y_true_class: True class labels (binary matrix)
        y_pred_section: Predicted section labels (binary matrix)
        y_pred_class: Predicted class labels (binary matrix)
        processor: Data processor with hierarchy mappings
        partial_credit: Credit given for correct section when class is wrong
    """
    n_samples = y_true_section.shape[0]

    # Initialize scores
    h_precision_scores = []
    h_recall_scores = []
    h_f1_scores = []

    for i in range(n_samples):
        # Get indices of true and predicted labels
        true_sections = np.where(y_true_section[i])[0]
        true_classes = np.where(y_true_class[i])[0]
        pred_sections = np.where(y_pred_section[i])[0]
        pred_classes = np.where(y_pred_class[i])[0]

        # Convert indices to actual codes
        true_section_codes = [processor.encoders["section"].classes_[idx] for idx in true_sections]
        true_class_codes = [processor.encoders["class"].classes_[idx] for idx in true_classes]
        pred_section_codes = [processor.encoders["section"].classes_[idx] for idx in pred_sections]
        pred_class_codes = [processor.encoders["class"].classes_[idx] for idx in pred_classes]

        # Calculate hierarchical precision
        if len(pred_section_codes) + len(pred_class_codes) > 0:
            precision_score_sum = 0

            # Full credit for correct class predictions
            for pc in pred_class_codes:
                if pc in true_class_codes:
                    precision_score_sum += 1.0
                else:
                    # Check if parent section is correct (partial credit)
                    parent_section = pc[:1]  # First character is section
                    if parent_section in true_section_codes:
                        precision_score_sum += partial_credit

            # Credit for section predictions not covered by class predictions
            for ps in pred_section_codes:
                # Only count if we haven't already counted this through class predictions
                if not any(pc.startswith(ps) for pc in pred_class_codes):
                    if ps in true_section_codes:
                        precision_score_sum += partial_credit

            h_precision = precision_score_sum / (len(pred_section_codes) + len(pred_class_codes))
        else:
            h_precision = 0.0

        # Calculate hierarchical recall
        if len(true_section_codes) + len(true_class_codes) > 0:
            recall_score_sum = 0

            # Full credit for recalled class labels
            for tc in true_class_codes:
                if tc in pred_class_codes:
                    recall_score_sum += 1.0
                else:
                    # Check if parent section was at least predicted (partial credit)
                    parent_section = tc[:1]
                    if parent_section in pred_section_codes:
                        recall_score_sum += partial_credit

            # Credit for section recalls not covered by class recalls
            for ts in true_section_codes:
                if not any(tc.startswith(ts) for tc in true_class_codes):
                    if ts in pred_section_codes:
                        recall_score_sum += partial_credit

            h_recall = recall_score_sum / (len(true_section_codes) + len(true_class_codes))
        else:
            h_recall = 0.0

        # Calculate hierarchical F1
        if h_precision + h_recall > 0:
            h_f1 = 2 * (h_precision * h_recall) / (h_precision + h_recall)
        else:
            h_f1 = 0.0

        h_precision_scores.append(h_precision)
        h_recall_scores.append(h_recall)
        h_f1_scores.append(h_f1)

    return {
        "hierarchical_precision": np.mean(h_precision_scores),
        "hierarchical_recall": np.mean(h_recall_scores),
        "hierarchical_f1": np.mean(h_f1_scores),
    }


def find_optimal_thresholds(
        y_true: dict,
        y_probs: dict,
        processor: IPCDataProcessor,
        config: Config
) -> dict:
    """
    Find optimal thresholds for each level that maximize hierarchical F1 score.

    Args:
        y_true: Dict with true labels for each level
        y_probs: Dict with predicted probabilities for each level
        processor: Data processor for hierarchy information
        config: Configuration with search parameters

    Returns:
        Dict with optimal thresholds for each level
    """
    start, end = config.threshold_search_range
    step = config.threshold_search_step

    best_thresholds = {"section": config.prediction_threshold["section"],
                       "class": config.prediction_threshold["class"]}
    best_h_f1 = 0.0

    # Grid search over threshold combinations
    for section_thresh in np.arange(start, end + step, step):
        for class_thresh in np.arange(start, end + step, step):
            # Apply thresholds
            y_pred_section = (y_probs["section"] >= section_thresh).astype(int)
            y_pred_class = (y_probs["class"] >= class_thresh).astype(int)

            # Compute hierarchical metrics
            h_metrics = compute_hierarchical_metrics(
                y_true["section"], y_true["class"],
                y_pred_section, y_pred_class,
                processor
            )

            # Update if better
            if h_metrics["hierarchical_f1"] > best_h_f1:
                best_h_f1 = h_metrics["hierarchical_f1"]
                best_thresholds = {"section": section_thresh, "class": class_thresh}

    return best_thresholds, best_h_f1


def calculate_average_labels(train_df: pd.DataFrame) -> dict:
    """Calculate average number of labels per level in training data."""
    avg_labels = {}
    for level in ["section", "class"]:
        label_counts = train_df[level].apply(len)
        avg_labels[level] = int(np.round(label_counts.mean()))
    return avg_labels


def calculate_class_weights(train_df: pd.DataFrame, encoders: dict) -> dict:
    """Calculate inverse frequency weights for each class to handle imbalance."""
    class_weights = {}

    for level in ["section", "class"]:
        # Count frequency of each label
        label_counts = np.zeros(len(encoders[level].classes_))

        for labels in train_df[level]:
            indices = encoders[level].transform([labels])[0]
            label_counts += indices

        # Calculate inverse frequency weights
        # Add smoothing to avoid division by zero
        total_samples = len(train_df)
        weights = total_samples / (label_counts + 1.0)

        # Normalize weights to have mean of 1.0
        weights = weights / weights.mean()

        # Convert to tensor
        class_weights[level] = torch.FloatTensor(weights)

        print(f"\n{level} class weights (sample):")
        for i in range(min(5, len(weights))):
            print(f"  {encoders[level].classes_[i]}: {weights[i]:.3f}")
        print(f"  ... (total {len(weights)} classes)")
        print(f"  Weight range: [{weights.min():.3f}, {weights.max():.3f}]")

    return class_weights


def train_epoch(
        model, dataloader, optimizer, scheduler, processor, config, device, scaler=None, class_weights=None
):
    """Train for one epoch with class-weighted loss."""
    model.train()
    total_loss = 0

    # Create weighted BCE loss functions
    if class_weights:
        section_criterion = nn.BCEWithLogitsLoss(
            pos_weight=class_weights["section"].to(device)
        )
        class_criterion = nn.BCEWithLogitsLoss(
            pos_weight=class_weights["class"].to(device)
        )
    else:
        section_criterion = nn.BCEWithLogitsLoss()
        class_criterion = nn.BCEWithLogitsLoss()

    progress_bar = tqdm(dataloader, desc="Training")

    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        section_labels = batch["section_labels"].to(device)
        class_labels = batch["class_labels"].to(device)

        # Zero gradients BEFORE forward pass
        optimizer.zero_grad()

        # Forward pass with soft hierarchical constraints
        if config.use_fp16 and scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model.predict_hierarchical(
                    input_ids,
                    attention_mask,
                    processor,
                    confidence_threshold=config.parent_confidence_threshold,
                    use_soft_masking=True,  # Soft masking for stable training
                )

                # Compute losses using class-weighted criteria
                section_loss = section_criterion(outputs["section_logits"], section_labels)
                class_loss = class_criterion(outputs["class_logits"], class_labels)

                # Check for extreme values
                if class_loss > 100:
                    print(
                        f"Warning: Extreme loss at batch {batch_idx} - class: {class_loss.item():.2f}"
                    )
                    print(
                        f"Class logits range: [{outputs['class_logits'].min().item():.2f}, "
                        f"{outputs['class_logits'].max().item():.2f}]"
                    )

                # Weighted sum of losses
                loss = 0.3 * section_loss + 0.7 * class_loss

            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model.predict_hierarchical(
                input_ids,
                attention_mask,
                processor,
                confidence_threshold=config.parent_confidence_threshold,
                use_soft_masking=True,  # Soft masking for stable training
            )

            # Compute losses using class-weighted criteria
            section_loss = section_criterion(outputs["section_logits"], section_labels)
            class_loss = class_criterion(outputs["class_logits"], class_labels)

            # Weighted sum of losses
            loss = 0.3 * section_loss + 0.7 * class_loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Step scheduler AFTER optimizer step
        scheduler.step()

        total_loss += loss.item()

        # Update progress bar with current loss values
        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "sec": f"{section_loss.item():.4f}",
                "cls": f"{class_loss.item():.4f}",
            }
        )

    return total_loss / len(dataloader)


def evaluate(model, dataloader, processor, config, device):
    """Evaluate model with both flat and hierarchical metrics."""
    model.eval()

    all_preds = {level: [] for level in ["section", "class"]}
    all_labels = {level: [] for level in ["section", "class"]}
    all_probs = {level: [] for level in ["section", "class"]}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get predictions with HARD constraints for evaluation
            outputs = model.predict_hierarchical(
                input_ids,
                attention_mask,
                processor,
                confidence_threshold=config.parent_confidence_threshold,
                use_soft_masking=False,  # Hard masking for evaluation
            )

            # Store probabilities and apply thresholds
            for level in ["section", "class"]:
                probs = torch.sigmoid(outputs[f"{level}_logits"])
                preds = (probs > config.prediction_threshold[level]).float()

                all_probs[level].append(probs.cpu().numpy())
                all_preds[level].append(preds.cpu().numpy())
                all_labels[level].append(batch[f"{level}_labels"].numpy())

    # Concatenate all batches
    for level in ["section", "class"]:
        all_probs[level] = np.vstack(all_probs[level])
        all_preds[level] = np.vstack(all_preds[level])
        all_labels[level] = np.vstack(all_labels[level])

    # Compute flat metrics
    metrics = {}
    for level in ["section", "class"]:
        metrics.update(compute_metrics(all_labels[level], all_preds[level], level))

    # Compute hierarchical metrics
    h_metrics = compute_hierarchical_metrics(
        all_labels["section"], all_labels["class"],
        all_preds["section"], all_preds["class"],
        processor
    )
    metrics.update(h_metrics)

    # Optimize thresholds if enabled
    optimal_thresholds = None
    if config.optimize_thresholds:
        optimal_thresholds, optimal_h_f1 = find_optimal_thresholds(
            all_labels, all_probs, processor, config
        )
        metrics["optimal_thresholds"] = optimal_thresholds
        metrics["optimal_hierarchical_f1"] = optimal_h_f1

    return metrics, all_probs, all_labels


def main():
    # Load config
    config = Config()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize GradScaler for fp16 if enabled and using cuda
    scaler = (
        torch.amp.GradScaler("cuda")
        if config.use_fp16 and device.type == "cuda"
        else None
    )
    if scaler:
        print("Using mixed precision training (fp16)")

    # Initialize processor
    print("Loading data processor...")
    processor = IPCDataProcessor(config.ipc_metadata_path)

    # Load and process data
    print("Loading datasets...")
    train_df = pd.read_csv(config.train_path)
    val_df = pd.read_csv(config.val_path)

    # Process with hints if enabled
    train_df = processor.process_dataframe(train_df, add_hints=config.use_hints)
    val_df = processor.process_dataframe(val_df, add_hints=False)  # No hints for validation

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Using hints: {config.use_hints}")

    # Calculate average labels for top-k
    avg_labels = calculate_average_labels(train_df)
    config.top_k = avg_labels
    print(f"Average labels per level: {avg_labels}")

    # Fit encoders
    encoders = processor.fit_encoders(train_df)

    # Calculate class weights for handling imbalance
    class_weights = None
    if config.use_class_weights:
        print("\nCalculating class weights for handling imbalance...")
        class_weights = calculate_class_weights(train_df, encoders)

    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Create datasets
    train_dataset = IPCDataset(
        train_df, tokenizer, encoders, config.max_length, use_hints=config.use_hints
    )
    val_dataset = IPCDataset(
        val_df, tokenizer, encoders, config.max_length, use_hints=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=1
    )

    # Initialize model
    print("Initializing model...")
    model = HierarchicalIPCClassifier(
        model_name=config.model_name,
        n_section=len(encoders["section"].classes_),
        n_class=len(encoders["class"].classes_),
        dropout=config.dropout,
    ).to(device)

    # Setup optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=0.01
    )

    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(config.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    best_h_f1 = 0
    best_config = config.__dict__.copy()
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, processor, config, device, scaler, class_weights
        )
        print(f"Train loss: {train_loss:.4f}")

        # Evaluate
        metrics, val_probs, val_labels = evaluate(model, val_loader, processor, config, device)

        # Print flat metrics
        print("\nFlat Metrics:")
        for level in ["section", "class"]:
            print(
                f"{level} - F1 micro: {metrics[f'{level}_f1_micro']:.4f}, "
                f"F1 macro: {metrics[f'{level}_f1_macro']:.4f}, "
                f"Precision micro: {metrics[f'{level}_precision_micro']:.4f}, "
                f"Recall micro: {metrics[f'{level}_recall_micro']:.4f}"
            )

        # Print hierarchical metrics
        print("\nHierarchical Metrics:")
        print(
            f"H-Precision: {metrics['hierarchical_precision']:.4f}, "
            f"H-Recall: {metrics['hierarchical_recall']:.4f}, "
            f"H-F1: {metrics['hierarchical_f1']:.4f}"
        )

        # Print optimal thresholds if found
        if "optimal_thresholds" in metrics:
            print(f"\nOptimal thresholds: {metrics['optimal_thresholds']}")
            print(f"Optimal H-F1: {metrics['optimal_hierarchical_f1']:.4f}")

        # Use hierarchical F1 for model selection
        current_h_f1 = metrics['hierarchical_f1']

        # Update config with optimal thresholds if better
        if "optimal_thresholds" in metrics and metrics["optimal_hierarchical_f1"] > current_h_f1:
            # Only update if optimized thresholds are higher than defaults
            update_config = False
            for level in ["section", "class"]:
                if metrics["optimal_thresholds"][level] > config.prediction_threshold[level]:
                    update_config = True
                    config.prediction_threshold[level] = metrics["optimal_thresholds"][level]

            if update_config:
                current_h_f1 = metrics["optimal_hierarchical_f1"]
                print(f"Updated thresholds to: {config.prediction_threshold}")

        # Save best model based on hierarchical F1
        if current_h_f1 > best_h_f1:
            best_h_f1 = current_h_f1
            best_config = config.__dict__.copy()

            # Save model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "config": best_config,
                },
                os.path.join(config.checkpoint_dir, "best_model.pt"),
            )

            # Save processor with hierarchy texts if using hints
            save_data = {
                "encoders": processor.encoders,
                "hierarchy_mappings": processor.hierarchy_mappings,
            }
            if hasattr(processor, 'hierarchy_texts'):
                save_data["hierarchy_texts"] = processor.hierarchy_texts

            with open(os.path.join(config.checkpoint_dir, "processor.pkl"), "wb") as f:
                pickle.dump(save_data, f)

            print(f"New best model saved! (H-F1: {best_h_f1:.4f})")

    print(f"\nTraining complete! Best Hierarchical F1: {best_h_f1:.4f}")
    print(f"Best configuration: {best_config}")


if __name__ == "__main__":
    main()
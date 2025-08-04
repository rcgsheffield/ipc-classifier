import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from config import Config
from data_processing import IPCDataProcessor, IPCDataset
from model import HierarchicalIPCClassifier


def predict_batch(model, dataloader, processor, config, device):
    """Make predictions for a batch of data with both threshold and top-k approaches."""
    model.eval()

    all_predictions_threshold = []
    all_predictions_topk = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model.predict_hierarchical(
                input_ids,
                attention_mask,
                processor,
                confidence_threshold=config.parent_confidence_threshold,
            )

            batch_size = input_ids.size(0)

            for i in range(batch_size):
                # Threshold-based predictions
                sample_preds_threshold = {}

                for level in ["section", "class"]:
                    probs = torch.sigmoid(outputs[f"{level}_logits"][i]).cpu().numpy()
                    threshold = config.prediction_threshold[level]

                    pred_indices = np.where(probs > threshold)[0]
                    pred_probs = probs[pred_indices]

                    sorted_idx = np.argsort(pred_probs)[::-1]
                    pred_indices = pred_indices[sorted_idx]
                    pred_probs = pred_probs[sorted_idx]

                    pred_labels = [
                        processor.encoders[level].classes_[idx] for idx in pred_indices
                    ]

                    sample_preds_threshold[level] = list(zip(pred_labels, pred_probs))

                all_predictions_threshold.append(sample_preds_threshold)

                # Top-k predictions
                if config.use_top_k and config.top_k:
                    sample_preds_topk = {}

                    for level in ["section", "class"]:
                        probs = torch.sigmoid(outputs[f"{level}_logits"][i]).cpu().numpy()
                        k = config.top_k.get(level, 1)

                        # Get top-k indices
                        if k >= len(probs):
                            top_k_indices = np.argsort(probs)[::-1]
                        else:
                            top_k_indices = np.argpartition(probs, -k)[-k:]
                            top_k_indices = top_k_indices[np.argsort(probs[top_k_indices])[::-1]]

                        top_k_probs = probs[top_k_indices]

                        # Filter out very low confidence predictions even in top-k
                        min_confidence = 0.1
                        valid_mask = top_k_probs > min_confidence
                        top_k_indices = top_k_indices[valid_mask]
                        top_k_probs = top_k_probs[valid_mask]

                        pred_labels = [
                            processor.encoders[level].classes_[idx] for idx in top_k_indices
                        ]

                        sample_preds_topk[level] = list(zip(pred_labels, top_k_probs))

                    all_predictions_topk.append(sample_preds_topk)

    return all_predictions_threshold, all_predictions_topk


def format_predictions_for_csv(predictions, true_labels, project_ids, has_labels=True):
    """Format predictions for CSV output in the specified format."""
    rows = []

    for i in range(len(predictions)):
        row = {
            "project_id": project_ids[i]
        }

        # Add true labels only if they exist
        if has_labels:
            for level in ["section", "class"]:
                true_codes = true_labels[i][level]
                # Format as comma-separated string without brackets
                row[f"{level}_label"] = ', '.join(true_codes) if true_codes else ""

        # Add predictions
        pred = predictions[i]
        for level in ["section", "class"]:
            level_preds = pred[level]

            if level_preds:
                labels = [label for label, _ in level_preds]
                probs = [prob for _, prob in level_preds]

                # Format as comma-separated strings
                row[f"{level}_predicted"] = ', '.join(labels)
                row[f"{level}_probability"] = ', '.join([f"{p:.3f}" for p in probs])
            else:
                row[f"{level}_predicted"] = ""
                row[f"{level}_probability"] = ""

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Predict IPC classifications')
    parser.add_argument('--input', type=str,
                        help='Path to input CSV file (if not specified, uses test_path from config)')
    parser.add_argument('--output', type=str,
                        help='Base name for output files (if not specified, uses output_path from config)')
    args = parser.parse_args()

    config = Config()

    # Override config paths if provided
    if args.input:
        input_path = args.input
        print(f"Using input file: {input_path}")
    else:
        input_path = config.test_path
        print(f"Using test file from config: {input_path}")

    if args.output:
        config.output_path = args.output

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading processor...")
    with open(os.path.join(config.checkpoint_dir, "scibert_hf1_0.389_processor.pkl"), "rb") as f:
        saved_data = pickle.load(f)

    processor = IPCDataProcessor(config.ipc_metadata_path)
    processor.encoders = saved_data["encoders"]
    processor.hierarchy_mappings = saved_data["hierarchy_mappings"]
    processor.hierarchy_texts = saved_data.get("hierarchy_texts", {})

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    print("Loading model...")
    checkpoint = torch.load(
        os.path.join(config.checkpoint_dir, "scibert_hf1_0.389_best_model.pt"), map_location=device, weights_only=False
    )

    # Load saved configuration
    model_config = checkpoint.get("config", {})
    use_hints = model_config.get("use_hints", False)

    # Update config with saved thresholds and top-k values
    if "prediction_threshold" in model_config:
        config.prediction_threshold = model_config["prediction_threshold"]
        print(f"Using saved thresholds: {config.prediction_threshold}")

    if "top_k" in model_config:
        config.top_k = model_config["top_k"]
        print(f"Using saved top-k values: {config.top_k}")

    print(f"Model was trained with hints: {use_hints}")

    model = HierarchicalIPCClassifier(
        model_name=config.model_name,
        n_section=len(processor.encoders["section"].classes_),
        n_class=len(processor.encoders["class"].classes_),
        dropout=config.dropout,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    print("Loading data...")
    test_df = pd.read_csv(input_path)

    # Remove rows with null/empty full_description
    test_df = test_df.dropna(subset=['full_description'])
    test_df = test_df[test_df['full_description'].str.strip() != '']
    test_df = test_df.reset_index(drop=True)

    print(f"Filtered dataset size: {len(test_df)} (removed empty descriptions)")

    # Extract project IDs (either appln_id or gtr_proj_id)
    if 'appln_id' in test_df.columns:
        project_ids = test_df['appln_id'].values
    elif 'gtr_proj_id' in test_df.columns:
        project_ids = test_df['gtr_proj_id'].values
    else:
        # Use first column as project ID
        project_ids = test_df.iloc[:, 0].values
        print(f"Using first column '{test_df.columns[0]}' as project ID")

    # Process without hints for test data (realistic scenario)
    test_df = processor.process_dataframe(test_df, add_hints=False)

    test_dataset = IPCDataset(
        test_df, tokenizer, processor.encoders, config.max_length, use_hints=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=0
    )

    print("Making predictions...")
    predictions_threshold, predictions_topk = predict_batch(
        model, test_loader, processor, config, device
    )

    # Prepare true labels if they exist
    true_labels = []
    has_labels = "section" in test_df.columns and "class" in test_df.columns

    if has_labels:
        for _, row in test_df.iterrows():
            true_labels.append(
                {
                    "section": row["section"],
                    "class": row["class"],
                }
            )
    else:
        print("No true labels found in data - running inference only mode")
        # Create empty label structure
        for _ in range(len(test_df)):
            true_labels.append(
                {
                    "section": [],
                    "class": [],
                }
            )

    # Prepare true labels if they exist
    true_labels = []
    has_labels = "section" in test_df.columns and "class" in test_df.columns

    if has_labels:
        for _, row in test_df.iterrows():
            true_labels.append(
                {
                    "section": row["section"],
                    "class": row["class"],
                }
            )
    else:
        print("No true labels found in data - running inference only mode")
        # Create empty label structure
        for _ in range(len(test_df)):
            true_labels.append(
                {
                    "section": [],
                    "class": [],
                }
            )

    # Format results for threshold-based predictions
    results_threshold_df = format_predictions_for_csv(
        predictions_threshold, true_labels, project_ids, has_labels
    )

    # Define column order based on whether we have labels
    if has_labels:
        columns = [
            "project_id",
            "section_label", "section_predicted", "section_probability",
            "class_label", "class_predicted", "class_probability"
        ]
    else:
        columns = [
            "project_id",
            "section_predicted", "section_probability",
            "class_predicted", "class_probability"
        ]

    results_threshold_df = results_threshold_df[columns]

    # Save threshold-based predictions
    threshold_output_path = config.output_path.replace('.csv', '_threshold.csv')
    results_threshold_df.to_csv(threshold_output_path, index=False)
    print(f"Threshold-based predictions saved to {threshold_output_path}")

    # Save top-k predictions if available
    if predictions_topk:
        results_topk_df = format_predictions_for_csv(
            predictions_topk, true_labels, project_ids, has_labels
        )
        results_topk_df = results_topk_df[columns]

        topk_output_path = config.output_path.replace('.csv', '_topk.csv')
        results_topk_df.to_csv(topk_output_path, index=False)
        print(f"Top-k predictions saved to {topk_output_path}")


if __name__ == "__main__":
    main()

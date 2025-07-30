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
    """Make predictions for a batch of data."""
    model.eval()

    all_predictions = []

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

            batch_preds = {}
            for level in ["section", "class"]:
                probs = torch.sigmoid(outputs[f"{level}_logits"]).cpu().numpy()
                batch_preds[level] = probs

            for i in range(len(probs)):
                sample_preds = {}
                for level in ["section", "class"]:
                    level_probs = batch_preds[level][i]
                    threshold = config.prediction_threshold[level]

                    pred_indices = np.where(level_probs > threshold)[0]
                    pred_probs = level_probs[pred_indices]

                    sorted_idx = np.argsort(pred_probs)[::-1]
                    pred_indices = pred_indices[sorted_idx]
                    pred_probs = pred_probs[sorted_idx]

                    pred_labels = [
                        processor.encoders[level].classes_[idx] for idx in pred_indices
                    ]

                    sample_preds[level] = list(zip(pred_labels, pred_probs))

                all_predictions.append(sample_preds)

    return all_predictions


def format_predictions_for_csv(predictions, true_labels=None):
    """Format predictions for CSV output with separate columns for labels and probabilities."""
    rows = []

    for i, pred in enumerate(predictions):
        row = {"index": i}

        if true_labels:
            for level in ["section", "class"]:
                true_codes = true_labels[i][level]
                row[f"true_{level}"] = (
                    f"[{', '.join(true_codes)}]"
                    if len(true_codes) > 1
                    else (true_codes[0] if true_codes else "")
                )

        for level in ["section", "class"]:
            level_preds = pred[level]

            if level_preds:
                labels = [label for label, _ in level_preds]
                probs = [f"{prob:.3f}" for _, prob in level_preds]

                row[f"{level}_predicted"] = (
                    f"[{', '.join(labels)}]"
                    if len(labels) > 1
                    else (labels[0] if labels else "")
                )
                row[f"{level}_probabilities"] = (
                    f"[{', '.join(probs)}]"
                    if len(probs) > 1
                    else (probs[0] if probs else "")
                )
            else:
                row[f"{level}_predicted"] = ""
                row[f"{level}_probabilities"] = ""

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading processor...")
    with open(os.path.join(config.checkpoint_dir, "processor.pkl"), "rb") as f:
        saved_data = pickle.load(f)

    processor = IPCDataProcessor(config.ipc_metadata_path)
    processor.encoders = saved_data["encoders"]
    processor.hierarchy_mappings = saved_data["hierarchy_mappings"]
    processor.hierarchy_texts = saved_data.get("hierarchy_texts", {})

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    print("Loading model...")
    checkpoint = torch.load(
        os.path.join(config.checkpoint_dir, "best_model.pt"), map_location=device
    )

    # Check if model was trained with hints
    model_config = checkpoint.get("config", {})
    use_hints = model_config.get("use_hints", False)
    print(f"Model was trained with hints: {use_hints}")

    model = HierarchicalIPCClassifier(
        model_name=config.model_name,
        n_section=len(processor.encoders["section"].classes_),
        n_class=len(processor.encoders["class"].classes_),
        dropout=config.dropout,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    print("Loading test data...")
    test_df = pd.read_csv(config.test_path)
    
    # Process with hints if model was trained with them
    test_df = processor.process_dataframe(test_df, add_hints=use_hints)

    test_dataset = IPCDataset(
        test_df, tokenizer, processor.encoders, config.max_length, use_hints=use_hints
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=4
    )

    print("Making predictions...")
    predictions = predict_batch(model, test_loader, processor, config, device)

    true_labels = []
    for _, row in test_df.iterrows():
        true_labels.append(
            {
                "section": row["section"],
                "class": row["class"],
            }
        )

    results_df = format_predictions_for_csv(predictions, true_labels)
    results_df["abstract"] = test_df["appln_abstract"].values

    columns = ["index", "abstract"]
    for level in ["section", "class"]:
        columns.extend(
            [f"true_{level}", f"{level}_predicted", f"{level}_probabilities"]
        )

    results_df = results_df[columns]
    results_df.to_csv(config.output_path, index=False)
    print(f"Predictions saved to {config.output_path}")

    print("\nSample predictions:")
    print("=" * 80)
    for i in range(min(3, len(results_df))):
        print(f"\nSample {i+1}:")
        print(f"Abstract: {results_df.iloc[i]['abstract'][:100]}...")
        for level in ["section", "class"]:
            true = results_df.iloc[i][f"true_{level}"]
            pred = results_df.iloc[i][f"{level}_predicted"]
            probs = results_df.iloc[i][f"{level}_probabilities"]
            print(f"\n{level.upper()}:")
            print(f"  True: {true}")
            print(f"  Pred: {pred}")
            print(f"  Prob: {probs}")


if __name__ == "__main__":
    main()

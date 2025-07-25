def format_predictions_for_csv(predictions, true_labels=None):
    """Format predictions for CSV output with separate columns per level."""
    rows = []
    
    for i, pred in enumerate(predictions):
        row = {'index': i}
        
        # Add true labels if available
        if true_labels:
            for level in ['section', 'class', 'subclass']:
                # Create separate columns for each true label
                for j, label in enumerate(true_labels[i][level]):
                    row[f'true_{level}_{j+1}'] = label
        
        # Add predictions - top 3 for each level
        for level in ['section', 'class', 'subclass']:
            level_preds = pred[level]
            import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import pickle
import os

from config import Config
from data_processing import IPCDataProcessor, IPCDataset
from model import HierarchicalIPCClassifier


def predict_batch(model, dataloader, processor, config, device):
    """Make predictions for a batch of data."""
    model.eval()
    
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get predictions with hierarchical constraints
            outputs = model.predict_hierarchical(
                input_ids, attention_mask, processor,
                confidence_threshold=config.parent_confidence_threshold,
                use_soft_masking=False  # Use hard masking for inference
            )
            
            # Convert to probabilities
            batch_preds = {}
            for level in ['section', 'class', 'subclass']:
                probs = torch.sigmoid(outputs[f'{level}_logits']).cpu().numpy()
                batch_preds[level] = probs
            
            # Process each sample in batch
            for i in range(len(probs)):
                sample_preds = {}
                
                for level in ['section', 'class', 'subclass']:
                    # Get predictions above threshold
                    level_probs = batch_preds[level][i]
                    threshold = config.prediction_threshold[level]
                    
                    pred_indices = np.where(level_probs > threshold)[0]
                    pred_probs = level_probs[pred_indices]
                    
                    # Sort by probability
                    sorted_idx = np.argsort(pred_probs)[::-1]
                    pred_indices = pred_indices[sorted_idx]
                    pred_probs = pred_probs[sorted_idx]
                    
                    # Get label names
                    pred_labels = [processor.encoders[level].classes_[idx] for idx in pred_indices]
                    
                    sample_preds[level] = list(zip(pred_labels, pred_probs))
                
                all_predictions.append(sample_preds)
    
    return all_predictions


def format_predictions_for_csv(predictions, true_labels=None):
    """Format predictions for CSV output with separate columns for labels and probabilities."""
    rows = []
    
    for i, pred in enumerate(predictions):
        row = {'index': i}
        
        # Add true labels if available
        if true_labels:
            for level in ['section', 'class', 'subclass']:
                true_codes = true_labels[i][level]
                row[f'true_{level}'] = f"[{', '.join(true_codes)}]" if len(true_codes) > 1 else (true_codes[0] if true_codes else '')
        
        # Add predictions with separate columns for labels and probabilities
        for level in ['section', 'class', 'subclass']:
            level_preds = pred[level]
            
            if level_preds:
                labels = [label for label, _ in level_preds]
                probs = [f"{prob:.3f}" for _, prob in level_preds]
                
                # Format with brackets if multiple, otherwise single value
                row[f'{level}_predicted'] = f"[{', '.join(labels)}]" if len(labels) > 1 else (labels[0] if labels else '')
                row[f'{level}_probabilities'] = f"[{', '.join(probs)}]" if len(probs) > 1 else (probs[0] if probs else '')
            else:
                row[f'{level}_predicted'] = ''
                row[f'{level}_probabilities'] = ''
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def main():
    # Load config
    config = Config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load saved processor
    print("Loading processor...")
    with open(os.path.join(config.checkpoint_dir, 'processor.pkl'), 'rb') as f:
        saved_data = pickle.load(f)
    
    processor = IPCDataProcessor(config.ipc_metadata_path)
    processor.encoders = saved_data['encoders']
    processor.hierarchy_mappings = saved_data['hierarchy_mappings']
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(os.path.join(config.checkpoint_dir, 'best_model.pt'), map_location=device)
    
    model = HierarchicalIPCClassifier(
        model_name=config.model_name,
        n_section=len(processor.encoders['section'].classes_),
        n_class=len(processor.encoders['class'].classes_),
        n_subclass=len(processor.encoders['subclass'].classes_),
        dropout=config.dropout
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(config.test_path)
    test_df = processor.process_dataframe(test_df)
    
    # Create dataset and dataloader
    test_dataset = IPCDataset(test_df, tokenizer, processor.encoders, config.max_length)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=1)
    
    # Make predictions
    print("Making predictions...")
    predictions = predict_batch(model, test_loader, processor, config, device)
    
    # Extract true labels for comparison
    true_labels = []
    for _, row in test_df.iterrows():
        true_labels.append({
            'section': row['section'],
            'class': row['class'],
            'subclass': row['subclass']
        })
    
    # Format as DataFrame
    results_df = format_predictions_for_csv(predictions, true_labels)
    
    # Add abstracts
    results_df['abstract'] = test_df['appln_abstract'].values
    
    # Reorder columns
    columns = ['index', 'abstract']
    for level in ['section', 'class', 'subclass']:
        columns.extend([f'true_{level}', f'{level}_predicted', f'{level}_probabilities'])
    
    results_df = results_df[columns]
    
    # Save results
    results_df.to_csv(config.output_path, index=False)
    print(f"Predictions saved to {config.output_path}")
    
    # Print sample results
    print("\nSample predictions:")
    print("=" * 80)
    
    for i in range(min(3, len(results_df))):
        print(f"\nSample {i+1}:")
        print(f"Abstract: {results_df.iloc[i]['abstract'][:100]}...")
        
        for level in ['section', 'class', 'subclass']:
            true = results_df.iloc[i][f'true_{level}']
            pred = results_df.iloc[i][f'predicted_{level}']
            probs = results_df.iloc[i][f'{level}_probabilities']
            
            print(f"\n{level.upper()}:")
            print(f"  True: {true}")
            print(f"  Pred: {pred}")
            print(f"  Prob: {probs}")


if __name__ == "__main__":
    main()

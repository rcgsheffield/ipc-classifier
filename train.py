import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import os
import json
import pickle

from config import Config
from data_processing import IPCDataProcessor, IPCDataset
from model import HierarchicalIPCClassifier


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, level: str) -> dict:
    """Compute metrics for a single level."""
    return {
        f'{level}_f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        f'{level}_f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        f'{level}_precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        f'{level}_precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        f'{level}_recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        f'{level}_recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0)
    }


def train_epoch(model, dataloader, optimizer, scheduler, processor, config, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        section_labels = batch['section_labels'].to(device)
        class_labels = batch['class_labels'].to(device)
        subclass_labels = batch['subclass_labels'].to(device)
        
        # Zero gradients BEFORE forward pass
        optimizer.zero_grad()
        
        # Forward pass - NOW WITH SOFT HIERARCHICAL CONSTRAINTS!
        if config.use_fp16 and scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model.predict_hierarchical(
                    input_ids, attention_mask, processor, 
                    confidence_threshold=config.parent_confidence_threshold,
                    use_soft_masking=True  # Soft masking for stable training
                )
                
                # Compute losses
                section_loss = criterion(outputs['section_logits'], section_labels)
                class_loss = criterion(outputs['class_logits'], class_labels)
                subclass_loss = criterion(outputs['subclass_logits'], subclass_labels)
                
                # Check for extreme values
                if class_loss > 100 or subclass_loss > 100:
                    print(f"Warning: Extreme loss at batch {batch_idx} - class: {class_loss.item():.2f}, subclass: {subclass_loss.item():.2f}")
                    print(f"Class logits range: [{outputs['class_logits'].min().item():.2f}, {outputs['class_logits'].max().item():.2f}]")
                
                # Weighted sum of losses
                loss = 0.2 * section_loss + 0.3 * class_loss + 0.5 * subclass_loss
            
            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model.predict_hierarchical(
                input_ids, attention_mask, processor, 
                confidence_threshold=config.parent_confidence_threshold,
                use_soft_masking=True  # Soft masking for stable training
            )
            
            # Compute losses
            section_loss = criterion(outputs['section_logits'], section_labels)
            class_loss = criterion(outputs['class_logits'], class_labels)
            subclass_loss = criterion(outputs['subclass_logits'], subclass_labels)
            
            # Weighted sum of losses
            loss = 0.1 * section_loss + 0.2 * class_loss + 0.7 * subclass_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Step scheduler AFTER optimizer step
        scheduler.step()
        
        total_loss += loss.item()
        
        # Update progress bar with current loss values
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'sec': f'{section_loss.item():.4f}',
            'cls': f'{class_loss.item():.4f}',
            'sub': f'{subclass_loss.item():.4f}'
        })
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, processor, config, device):
    """Evaluate model."""
    model.eval()
    
    all_preds = {level: [] for level in ['section', 'class', 'subclass']}
    all_labels = {level: [] for level in ['section', 'class', 'subclass']}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get predictions with HARD constraints for evaluation
            outputs = model.predict_hierarchical(
                input_ids, attention_mask, processor,
                confidence_threshold=config.parent_confidence_threshold,
                use_soft_masking=False  # Hard masking for evaluation
            )
            
            # Apply thresholds
            for level in ['section', 'class', 'subclass']:
                probs = torch.sigmoid(outputs[f'{level}_logits'])
                preds = (probs > config.prediction_threshold[level]).float()
                
                all_preds[level].append(preds.cpu().numpy())
                all_labels[level].append(batch[f'{level}_labels'].numpy())
    
    # Concatenate and compute metrics
    metrics = {}
    for level in ['section', 'class', 'subclass']:
        y_true = np.vstack(all_labels[level])
        y_pred = np.vstack(all_preds[level])
        metrics.update(compute_metrics(y_true, y_pred, level))
    
    return metrics


def main():
    # Load config
    config = Config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize GradScaler for fp16 if enabled and using cuda
    scaler = torch.amp.GradScaler('cuda') if config.use_fp16 and device.type == 'cuda' else None
    if scaler:
        print("Using mixed precision training (fp16)")
    
    # Initialize processor
    print("Loading data processor...")
    processor = IPCDataProcessor(config.ipc_metadata_path)
    
    # Load and process data
    print("Loading datasets...")
    train_df = pd.read_csv(config.train_path)
    val_df = pd.read_csv(config.val_path)
    
    train_df = processor.process_dataframe(train_df)
    val_df = processor.process_dataframe(val_df)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    # Fit encoders
    encoders = processor.fit_encoders(train_df)
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Create datasets
    train_dataset = IPCDataset(train_df, tokenizer, encoders, config.max_length)
    val_dataset = IPCDataset(val_df, tokenizer, encoders, config.max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=1)
    
    # Initialize model
    print("Initializing model...")
    model = HierarchicalIPCClassifier(
        model_name=config.model_name,
        n_section=len(encoders['section'].classes_),
        n_class=len(encoders['class'].classes_),
        n_subclass=len(encoders['subclass'].classes_),
        dropout=config.dropout
    ).to(device)
    
    # Setup optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(config.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # Training loop
    best_f1 = 0
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, processor, config, device, scaler)
        print(f"Train loss: {train_loss:.4f}")
        
        # Evaluate
        metrics = evaluate(model, val_loader, processor, config, device)
        
        # Print metrics
        for level in ['section', 'class', 'subclass']:
            print(f"{level} - F1 micro: {metrics[f'{level}_f1_micro']:.4f}, "
                  f"F1 macro: {metrics[f'{level}_f1_macro']:.4f}, "
                  f"Precision micro: {metrics[f'{level}_precision_micro']:.4f}, "
                  f"Recall micro: {metrics[f'{level}_recall_micro']:.4f}")
        
        # Average F1 for model selection
        avg_f1 = np.mean([metrics[f'{level}_f1_micro'] for level in ['section', 'class', 'subclass']])
        print(f"Average F1: {avg_f1:.4f}")
        
        # Save best model
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'config': config.__dict__
            }, os.path.join(config.checkpoint_dir, 'best_model.pt'))
            
            # Save processor
            with open(os.path.join(config.checkpoint_dir, 'processor.pkl'), 'wb') as f:
                pickle.dump({
                    'encoders': processor.encoders,
                    'hierarchy_mappings': processor.hierarchy_mappings
                }, f)
            
            print(f"New best model saved! (F1: {avg_f1:.4f})")
    
    print(f"\nTraining complete! Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()

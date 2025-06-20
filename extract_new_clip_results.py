#!/usr/bin/env python3
"""
Extract training results from new_clip_train.log and generate CSV and visualization
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def extract_epoch_data(log_file):
    """Extract epoch data from log file"""
    epochs = []
    train_losses = []
    train_accuracies = []
    val_losses = []
    
    # Organ validation results
    organ_results = {
        'liver': {'loss': [], 'dice': []},
        'right_kidney': {'loss': [], 'dice': []},
        'spleen': {'loss': [], 'dice': []},
        'pancreas': {'loss': [], 'dice': []},
        'aorta': {'loss': [], 'dice': []},
        'ivc': {'loss': [], 'dice': []},
        'right_adrenal_gland': {'loss': [], 'dice': []},
        'left_adrenal_gland': {'loss': [], 'dice': []},
        'gallbladder': {'loss': [], 'dice': []},
        'esophagus': {'loss': [], 'dice': []},
        'stomach': {'loss': [], 'dice': []},
        'left_kidney': {'loss': [], 'dice': []}
    }
    
    current_epoch = None
    current_train_loss = None
    current_train_accuracy = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Extract epoch number
            epoch_match = re.search(r'Epoch (\d+)/100', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue
            
            # Extract training loss and accuracy
            train_match = re.search(r'train_Loss: ([\d.]+), train_accuracy : ([\d.]+)', line)
            if train_match:
                current_train_loss = float(train_match.group(1))
                current_train_accuracy = float(train_match.group(2))
                continue
            
            # Extract validation results for each organ
            for organ in organ_results.keys():
                # Pattern for validation complete lines
                val_pattern = rf'{organ.capitalize()} Validation Complete: Loss: ([\d.]+), Dice: ([\d.]+)'
                val_match = re.search(val_pattern, line)
                if val_match:
                    loss = float(val_match.group(1))
                    dice = float(val_match.group(2))
                    organ_results[organ]['loss'].append(loss)
                    organ_results[organ]['dice'].append(dice)
                    break
            
            # Extract total validation loss
            val_loss_match = re.search(r'val loss: ([\d.]+)', line)
            if val_loss_match and current_epoch is not None:
                val_loss = float(val_loss_match.group(1))
                
                # Store epoch data
                epochs.append(current_epoch)
                train_losses.append(current_train_loss)
                train_accuracies.append(current_train_accuracy)
                val_losses.append(val_loss)
                
                # Reset for next epoch
                current_epoch = None
                current_train_loss = None
                current_train_accuracy = None
    
    return epochs, train_losses, train_accuracies, val_losses, organ_results

def create_dataframe(epochs, train_losses, train_accuracies, val_losses, organ_results):
    """Create a pandas DataFrame with all the data"""
    data = {
        'epoch': epochs,
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_loss': val_losses
    }
    
    # Add organ-specific results
    for organ, results in organ_results.items():
        if results['loss']:  # Only add if we have data
            data[f'{organ}_loss'] = results['loss']
            data[f'{organ}_dice'] = results['dice']
    
    return pd.DataFrame(data)

def plot_training_curves(df, save_path='new_clip_training_curves.png'):
    """Create training curves visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training and validation loss
    axes[0, 0].plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training accuracy
    axes[0, 1].plot(df['epoch'], df['train_accuracy'], 'g-', label='Train Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Organ Dice scores
    dice_columns = [col for col in df.columns if col.endswith('_dice')]
    for col in dice_columns:
        organ_name = col.replace('_dice', '').replace('_', ' ').title()
        axes[1, 0].plot(df['epoch'], df[col], label=organ_name, linewidth=1.5)
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Score')
    axes[1, 0].set_title('Validation Dice Scores by Organ')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Organ Loss scores
    loss_columns = [col for col in df.columns if col.endswith('_loss') and col != 'train_loss' and col != 'val_loss']
    for col in loss_columns:
        organ_name = col.replace('_loss', '').replace('_', ' ').title()
        axes[1, 1].plot(df['epoch'], df[col], label=organ_name, linewidth=1.5)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Validation Loss by Organ')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_path}")

def plot_dice_comparison(df, save_path='new_clip_dice_comparison.png'):
    """Create a bar chart comparing final Dice scores"""
    dice_columns = [col for col in df.columns if col.endswith('_dice')]
    
    # Get final Dice scores
    final_dice = {}
    for col in dice_columns:
        organ_name = col.replace('_dice', '').replace('_', ' ').title()
        final_dice[organ_name] = df[col].iloc[-1]
    
    # Sort by Dice score
    sorted_organs = sorted(final_dice.items(), key=lambda x: x[1], reverse=True)
    organs, scores = zip(*sorted_organs)
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(organs)), scores, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Organs')
    plt.ylabel('Final Dice Score')
    plt.title('Final Validation Dice Scores by Organ (New CLIP Training)')
    plt.xticks(range(len(organs)), organs, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dice comparison saved to {save_path}")

def main():
    # Extract data from log file
    print("Extracting data from new_clip_train.log...")
    epochs, train_losses, train_accuracies, val_losses, organ_results = extract_epoch_data('new_clip_train.log')
    
    # Create DataFrame
    df = create_dataframe(epochs, train_losses, train_accuracies, val_losses, organ_results)
    
    # Save to CSV
    csv_path = 'new_clip_training_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Print summary statistics
    print("\n=== Training Summary ===")
    print(f"Total epochs: {len(df)}")
    print(f"Final train loss: {df['train_loss'].iloc[-1]:.4f}")
    print(f"Final train accuracy: {df['train_accuracy'].iloc[-1]:.4f}")
    print(f"Final validation loss: {df['val_loss'].iloc[-1]:.4f}")
    
    print("\n=== Final Dice Scores ===")
    dice_columns = [col for col in df.columns if col.endswith('_dice')]
    for col in dice_columns:
        organ_name = col.replace('_dice', '').replace('_', ' ').title()
        final_dice = df[col].iloc[-1]
        print(f"{organ_name}: {final_dice:.4f}")
    
    # Create visualizations
    plot_training_curves(df)
    plot_dice_comparison(df)
    
    print("\n=== Data Extraction Complete ===")
    print("Files generated:")
    print(f"1. {csv_path} - CSV data file")
    print("2. new_clip_training_curves.png - Training curves")
    print("3. new_clip_dice_comparison.png - Final Dice comparison")

if __name__ == "__main__":
    main() 
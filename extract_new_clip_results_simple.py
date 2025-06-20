#!/usr/bin/env python3
"""
Extract training results from new_clip_train.log and generate CSV (using only standard library)
"""

import re
import csv

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

def save_to_csv(epochs, train_losses, train_accuracies, val_losses, organ_results, filename):
    """Save data to CSV file"""
    with open(filename, 'w', newline='') as csvfile:
        # Define fieldnames
        fieldnames = ['epoch', 'train_loss', 'train_accuracy', 'val_loss']
        
        # Add organ-specific columns
        for organ in organ_results.keys():
            if organ_results[organ]['loss']:  # Only add if we have data
                fieldnames.extend([f'{organ}_loss', f'{organ}_dice'])
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write data rows
        for i, epoch in enumerate(epochs):
            row = {
                'epoch': epoch,
                'train_loss': train_losses[i],
                'train_accuracy': train_accuracies[i],
                'val_loss': val_losses[i]
            }
            
            # Add organ-specific data
            for organ, results in organ_results.items():
                if results['loss'] and i < len(results['loss']):
                    row[f'{organ}_loss'] = results['loss'][i]
                    row[f'{organ}_dice'] = results['dice'][i]
            
            writer.writerow(row)

def print_summary(epochs, train_losses, train_accuracies, val_losses, organ_results):
    """Print summary statistics"""
    print("\n=== Training Summary ===")
    print(f"Total epochs: {len(epochs)}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final train accuracy: {train_accuracies[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    
    print("\n=== Final Dice Scores ===")
    for organ, results in organ_results.items():
        if results['dice']:
            final_dice = results['dice'][-1]
            organ_name = organ.replace('_', ' ').title()
            print(f"{organ_name}: {final_dice:.4f}")

def main():
    # Extract data from log file
    print("Extracting data from new_clip_train.log...")
    epochs, train_losses, train_accuracies, val_losses, organ_results = extract_epoch_data('new_clip_train.log')
    
    # Save to CSV
    csv_path = 'new_clip_training_results.csv'
    save_to_csv(epochs, train_losses, train_accuracies, val_losses, organ_results, csv_path)
    print(f"Results saved to {csv_path}")
    
    # Print summary
    print_summary(epochs, train_losses, train_accuracies, val_losses, organ_results)
    
    print("\n=== Data Extraction Complete ===")
    print(f"CSV file generated: {csv_path}")
    print(f"Total epochs extracted: {len(epochs)}")
    
    # Print first few rows as preview
    print("\n=== First 3 rows preview ===")
    print("Epoch | Train Loss | Train Acc | Val Loss | Liver Dice | Right Kidney Dice")
    print("-" * 70)
    for i in range(min(3, len(epochs))):
        liver_dice = organ_results['liver']['dice'][i] if organ_results['liver']['dice'] else "N/A"
        right_kidney_dice = organ_results['right_kidney']['dice'][i] if organ_results['right_kidney']['dice'] else "N/A"
        print(f"{epochs[i]:5d} | {train_losses[i]:9.4f} | {train_accuracies[i]:9.4f} | {val_losses[i]:8.4f} | {liver_dice:9.4f} | {right_kidney_dice:16.4f}")

if __name__ == "__main__":
    main() 
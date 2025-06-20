#!/usr/bin/env python3
"""
Safe model saving utility for GPFS file system
"""

import os
import torch
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

def save_checkpoint_safely(checkpoint, save_path, max_retries=3, retry_delay=2):
    """
    Safely save a checkpoint with retry mechanism and verification
    
    Args:
        checkpoint: The checkpoint dictionary to save
        save_path: The target path to save the checkpoint
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    """
    save_dir = os.path.dirname(save_path)
    filename = os.path.basename(save_path)
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    for attempt in range(max_retries):
        temp_path = os.path.join(save_dir, f"{filename}.tmp.{attempt}")
        
        try:
            logger.info(f"Attempt {attempt + 1}: Saving to {temp_path}")
            
            # Save to temporary file
            torch.save(checkpoint, temp_path)
            
            # Verify the saved file
            logger.info("Verifying saved checkpoint...")
            test_checkpoint = torch.load(temp_path, map_location='cpu')
            
            # Check if all required keys are present
            required_keys = ['model', 'optimizer', 'epoch']
            if not all(key in test_checkpoint for key in required_keys):
                raise ValueError("Checkpoint verification failed - missing required keys")
            
            # Check file size (should be reasonable)
            file_size = os.path.getsize(temp_path)
            if file_size < 1024 * 1024:  # Less than 1MB
                raise ValueError("Checkpoint file too small - likely corrupted")
            
            # Atomic move to final location
            logger.info(f"Moving {temp_path} to {save_path}")
            os.rename(temp_path, save_path)
            
            logger.info(f"Successfully saved checkpoint to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            
            # Clean up temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.info(f"Cleaned up temporary file: {temp_path}")
                except Exception as cleanup_error:
                    logger.error(f"Failed to clean up temp file: {cleanup_error}")
            
            if attempt < max_retries - 1:
                logger.info(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to save checkpoint after {max_retries} attempts")
                return False
    
    return False

def save_model_only(model_state_dict, save_path, max_retries=3):
    """
    Save only the model state dict (smaller file size)
    
    Args:
        model_state_dict: The model's state dict
        save_path: The target path to save
        max_retries: Maximum number of retry attempts
    """
    checkpoint = {
        "model": model_state_dict,
        "epoch": 0,  # Placeholder
    }
    
    return save_checkpoint_safely(checkpoint, save_path, max_retries)

def create_backup_path(original_path):
    """
    Create a backup path with timestamp
    """
    dir_path = os.path.dirname(original_path)
    filename = os.path.basename(original_path)
    name, ext = os.path.splitext(filename)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{name}_backup_{timestamp}{ext}"
    return os.path.join(dir_path, backup_filename)

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Safe model saving utility")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument("--retries", type=int, default=3, help="Number of retry attempts")
    
    args = parser.parse_args()
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Save safely
    success = save_checkpoint_safely(checkpoint, args.output, args.retries)
    
    if success:
        print(f"Successfully saved to {args.output}")
    else:
        print(f"Failed to save to {args.output}")
        sys.exit(1) 
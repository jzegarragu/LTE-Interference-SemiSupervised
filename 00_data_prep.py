"""
Data Preparation Script
-----------------------
Thesis: "LTE Interference Classification via Semi-Supervised Teacher-Student Approach".

This script performs a stratified split of the original dataset into 
training/validation and test sets to ensure class representativeness.
"""

import sys
import shutil
import logging
from pathlib import Path
from typing import List
from sklearn.model_selection import train_test_split
import config

# Configure Logging System
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def prepare_data(
    source_dir: Path, 
    split_dir: Path, 
    test_ratio: float = 0.15, 
    seed: int = 42
) -> None:
    """
    Organizes the dataset by splitting it into train/val and test sets.
    
    Args:
        source_dir (Path): Directory containing the original class folders.
        split_dir (Path): Target directory where 'train_val' and 'test' folders will be created.
        test_ratio (float): Fraction of the dataset reserved for testing (0.0 to 1.0).
        seed (int): Random seed for reproducibility.
    
    Raises:
        FileNotFoundError: If the source directory does not exist.
        ValueError: If no images or class folders are found.
    """
    
    # 1. Initial Validations
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    if split_dir.exists():
        logger.info(f"Target directory '{split_dir}' already exists. Skipping data preparation.")
        return

    # 2. Identify Classes and Files
    # Filter out special folders or hidden files
    class_names = [p.name for p in source_dir.iterdir() if p.is_dir() and not p.name.startswith('.')]
    
    if not class_names:
        raise ValueError(f"No class subdirectories found in {source_dir}")

    all_files = []
    all_labels = []

    for class_name in class_names:
        class_path = source_dir / class_name
        # Search for PNG images (adjust extension if needed)
        files = list(class_path.glob("*.png"))
        all_files.extend(files)
        all_labels.extend([class_name] * len(files))

    if not all_files:
        raise ValueError("No image files (.png) found in class directories.")

    logger.info(f"Total images found: {len(all_files)} distributed across {len(class_names)} classes.")

    # 3. Stratified Data Split
    train_val_files, test_files, _, _ = train_test_split(
        all_files, 
        all_labels,
        test_size=test_ratio,
        random_state=seed,
        stratify=all_labels
    )
    
    logger.info(f"Split completed: {len(train_val_files)} train/val samples, {len(test_files)} test samples.")

    # 4. File Copying (Internal Helper)
    def _copy_subset(files: List[Path], subset_name: str):
        subset_dir = split_dir / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path in files:
            class_name = file_path.parent.name
            dest_dir = subset_dir / class_name
            dest_dir.mkdir(exist_ok=True)
            shutil.copy(file_path, dest_dir / file_path.name)
            
        logger.info(f"Subset '{subset_name}' successfully generated at {subset_dir}")

    # Execute Copy
    try:
        _copy_subset(train_val_files, "train_val")
        _copy_subset(test_files, "test")
        logger.info("Data preparation completed successfully.")
        
    except Exception as e:
        logger.error(f"Error during file copying: {e}")
        # Cleanup in case of error to avoid corrupt data
        if split_dir.exists():
            shutil.rmtree(split_dir)
        raise

if __name__ == "__main__":
    try:
        prepare_data(
            source_dir=config.RAW_DATA_DIR,   # Note: Updated to match new config names
            split_dir=config.PROCESSED_DATA_DIR, 
            test_ratio=0.15,
            seed=config.SEED
        )
    except Exception as e:
        logger.critical(f"Script failed: {e}")
        sys.exit(1)
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

from data.loaders.MyoGymLoader import MyoGymLoader
from data.loaders.RecoFitLoader import RecoFitLoader
from data.preprocessing.windowing import create_windows
from data.preprocessing.preprocess import setup_binary_classification, split_by_trainer


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline."""
    dataset: str  # 'myogym' or 'recofit'
    normalization_strategy: Optional[str] = None
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    window_size: int = 100
    window_step: int = 50
    output_dir: str = "data/processed"
    transform_units: bool = False  # Convert g->m/s^2, dps->rad/s
    seed: int = 42


def create_windowed_dataset(config: PipelineConfig) -> Dict[str, Path]:
    """
    Create a windowed dataset from raw IMU data.
    
    Orchestrates: Load → Window → Split → Save
    
    Parameters:
    -----------
    config : PipelineConfig
        Configuration object specifying dataset, preprocessing options, etc.
    
    Returns:
    --------
    dict : Paths to saved NPZ files for train, val, test splits
    """
    
    # Ensure output directory exists
    base_output_dir = Path(config.output_dir) / config.dataset
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate single timestamp and create timestamped directory
    timestamp = datetime.now().isoformat()
    timestamp_str = datetime.fromisoformat(timestamp).strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / timestamp_str
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data based on dataset choice
    print(f"Loading {config.dataset} dataset...")
    if config.dataset.lower() == 'myogym':
        loader = MyoGymLoader("data/datasets/MyoGym.mat", transform_units=config.transform_units)
    elif config.dataset.lower() == 'recofit':
        loader = RecoFitLoader("data/datasets/RecoFit", transform_units=config.transform_units)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    
    data_df, activity_mapping = loader.load_data()
    non_lifting_activities = loader.get_non_lifting_activities()
    
    # Create sliding windows
    print(f"Creating windows (size={config.window_size}, step={config.window_step})...")
    x, y, meta = create_windows(
        data_df,
        window_size=config.window_size,
        step=config.window_step
    )
    
    # Apply normalization if specified
    if config.normalization_strategy:
        print(f"Applying normalization: {config.normalization_strategy}")
        # TODO: Implement normalization strategies
        pass
    
    # Split data by trainer
    print(f"Splitting data (train={config.train_ratio}, val={config.val_ratio})...")
    x_train, y_train, meta_train, x_val, y_val, meta_val, x_test, y_test, meta_test = split_by_trainer(
        x, y, meta,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed
    )
    
    # Convert to binary classification (lifting vs non-lifting)
    print("Converting to binary classification (lifting vs non-lifting)...")
    y_train = setup_binary_classification(y_train, activity_mapping, non_lifting_activities)
    y_val = setup_binary_classification(y_val, activity_mapping, non_lifting_activities)
    y_test = setup_binary_classification(y_test, activity_mapping, non_lifting_activities)
    
    # Extract trainer IDs for each split
    train_trainer_ids = sorted(set(m["trainer"] for m in meta_train))
    val_trainer_ids = sorted(set(m["trainer"] for m in meta_val))
    test_trainer_ids = sorted(set(m["trainer"] for m in meta_test))
    
    # Calculate total samples and ratios
    total_samples = len(x_train) + len(x_val) + len(x_test)
    
    # Define units based on transform_units flag
    units = {
        "acceleration": "m/s^2" if config.transform_units else "g",
        "rotation": "rad/s" if config.transform_units else "dps",
    }
    
    # Create base pipeline metadata
    pipeline_metadata = {
        "date": timestamp,
        "dataset": config.dataset,
        "normalization_strategy": config.normalization_strategy,
        "window_size": config.window_size,
        "window_step": config.window_step,
        "train_ratio": config.train_ratio,
        "val_ratio": config.val_ratio,
        "seed": config.seed,
        "transform_units": config.transform_units,
        "units": units,
        "activity_mapping": activity_mapping,
        "non_lifting_activities": non_lifting_activities,
        "split_data": {
            "train": {
                "sample_count": len(x_train),
                "ratio_percentage": round((len(x_train) / total_samples) * 100, 2),
                "trainer_ids": list(train_trainer_ids),
            },
            "val": {
                "sample_count": len(x_val),
                "ratio_percentage": round((len(x_val) / total_samples) * 100, 2),
                "trainer_ids": list(val_trainer_ids),
            },
            "test": {
                "sample_count": len(x_test),
                "ratio_percentage": round((len(x_test) / total_samples) * 100, 2),
                "trainer_ids": list(test_trainer_ids),
            },
        },
    }
    
    # Save splits
    saved_files = {}
    splits = {
        "train": (x_train, y_train, meta_train),
        "val": (x_val, y_val, meta_val),
        "test": (x_test, y_test, meta_test),
    }
    
    for split_name, (x_split, y_split, meta_split) in splits.items():
        filename = f"{config.dataset}_{split_name}.npz"
        filepath = output_dir / filename
        
        print(f"Saving {split_name} split ({len(x_split)} samples) to {filepath}...")
        np.savez_compressed(
            filepath,
            x=x_split,
            y=y_split,
            meta=np.array(meta_split, dtype=object),
            metadata=pipeline_metadata
        )
        
        saved_files[split_name] = filepath
    
    # Save metadata as JSON
    metadata_json_path = output_dir / "metadata.json"
    with open(metadata_json_path, 'w') as f:
        json.dump(pipeline_metadata, f, indent=2, default=str)
    print(f"Saved metadata to {metadata_json_path}")
    
    print(f"\n✓ Dataset creation complete!")
    print(f"  Train: {len(x_train)} samples | Trainers: {train_trainer_ids}")
    print(f"  Val:   {len(x_val)} samples | Trainers: {val_trainer_ids}")
    print(f"  Test:  {len(x_test)} samples | Trainers: {test_trainer_ids}")
    
    return saved_files


if __name__ == "__main__":
    # Example usage
    config = PipelineConfig(
        dataset="recofit",
        transform_units=True,
        normalization_strategy=None,
        train_ratio=0.8,
        val_ratio=0.2,
        output_dir="D:/GAR/data/processed",
    )
    
    saved_files = create_windowed_dataset(config)
    print("\nSaved files:")
    for split, path in saved_files.items():
        print(f"  {split}: {path}")
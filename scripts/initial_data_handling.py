"""
This script takes the full dataset and sets aside 150,000 as final test set.
From the remaining data, it leaves one class label out, 
and randomly picks 150,000 samples from the remaining samples for intial work. 
"""

import pandas as pd
from pathlib import Path
import argparse
from typing import Optional

# --- Configuration Constants ---
RAW_DATA_PATH = Path("data/train.csv")
TEST_SET_OUTPUT_PATH = Path("data/test_set.csv")
INITIAL_DATASET_OUTPUT_PATH = Path("data/dataset_initial.csv")
TARGET_COLUMN = 'Fertilizer Name'
TEST_SAMPLE_SIZE = 150_000
INITIAL_SAMPLE_SIZE = 150_000
EXCLUDE_LABEL = 'DAP'

def prepare_initial_datasets(
    raw_path: Path, 
    test_output: Path, 
    initial_output: Path, 
    target_col: str, 
    test_size: int, 
    initial_size: int, 
    exclude_label: str, 
    random_seed: Optional[int] = 42
) -> None:
    """
    Loads the raw CSV data, performs sampling for the permanent test set, 
    and generates a filtered development dataset.
    """
    
    # Ensure the output directory exists
    test_output.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Load the full dataset
        print(f"Loading raw data from: {raw_path}")
        df_full = pd.read_csv(raw_path)
    except FileNotFoundError:
        print(f"ERROR: Raw data file not found at {raw_path}. Please check the path.")
        return

    # --- A. Create Permanent Test Set ---
    
    # Randomly sample the test set (this data is now reserved for final evaluation)
    df_test_set = df_full.sample(n=test_size, random_state=random_seed)
    
    # Save the test set
    df_test_set.to_csv(test_output, index=False)
    print(f"SUCCESS: Permanent test set ({len(df_test_set)} rows) saved to {test_output}")

    # --- B. Create Initial Development Dataset ---

    # Get the remaining data (excluding the test set to maintain independence)
    test_indices = df_test_set.index
    df_remaining = df_full.drop(test_indices)

    # Filter out the excluded label ('DAP') for initial work
    df_filtered = df_remaining[df_remaining[target_col] != exclude_label]
    
    # Sample the initial development dataset from the filtered remaining data
    if len(df_filtered) < initial_size:
        print(f"WARNING: Only {len(df_filtered)} rows available after filtering (excluding {exclude_label}). Using all available rows.")
        df_initial = df_filtered
    else:
        df_initial = df_filtered.sample(n=initial_size, random_state=random_seed)

    # Save the initial development set
    df_initial.to_csv(initial_output, index=False)
    print(f"SUCCESS: Initial development dataset ({len(df_initial)} rows, excluding '{exclude_label}') saved to {initial_output}")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Prepare initial test and development datasets from raw CSV.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    prepare_initial_datasets(
        raw_path=RAW_DATA_PATH,
        test_output=TEST_SET_OUTPUT_PATH,
        initial_output=INITIAL_DATASET_OUTPUT_PATH,
        target_col=TARGET_COLUMN,
        test_size=TEST_SAMPLE_SIZE,
        initial_size=INITIAL_SAMPLE_SIZE,
        exclude_label=EXCLUDE_LABEL,
        random_seed=args.seed
    )
# src/data_splitter.py

import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

def split_data(
    df: pd.DataFrame, 
    target_column: str,
    validation_size: float,
    random_seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the input DataFrame into training and validation sets.
    """
    
    # Separate features (X) from the target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Perform the split using the configured validation_size
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=validation_size,
        random_state=random_seed, 
        shuffle=True,
        stratify=y 
    )
    
    # Recombine X and y for easier passing (features + target)
    df_train = pd.concat([X_train, y_train], axis=1)
    df_val = pd.concat([X_val, y_val], axis=1)
    
    # Returning just the split DataFrames, as requested
    return df_train, df_val
import pandas as pd
import numpy as np
from typing import Dict

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize dataframe memory usage"""
    df_optimized = df.copy()
    
    # Optimize numeric columns
    for col in df_optimized.select_dtypes(include=[np.number]).columns:
        col_min = df_optimized[col].min()
        col_max = df_optimized[col].max()
        
        if str(df_optimized[col].dtype)[:3] == 'int':
            if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                df_optimized[col] = df_optimized[col].astype(np.int8)
            elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                df_optimized[col] = df_optimized[col].astype(np.int16)
            elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                df_optimized[col] = df_optimized[col].astype(np.int32)
        else:
            if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max:
                df_optimized[col] = df_optimized[col].astype(np.float32)
    
    return df_optimized

def get_memory_usage(df: pd.DataFrame) -> Dict[str, float]:
    """Get memory usage statistics"""
    return {
        'total_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'per_column': df.memory_usage(deep=True).to_dict()
    }
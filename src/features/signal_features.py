import pandas as pd
import numpy as np
from scipy import signal
from typing import List, Dict
import logging

class SignalFeatureExtractor:
    """Extract features from physiological signals"""
    
    def __init__(self, sampling_rate: int = 1):
        self.sampling_rate = sampling_rate
        self.physiological_cols = ['EDA', 'HR', 'TEMP']
        self.movement_cols = ['X', 'Y', 'Z']
        self.logger = logging.getLogger(__name__)
    
    def extract_rolling_features(self, df: pd.DataFrame, 
                               windows: List[int] = [60, 300]) -> pd.DataFrame:
        """Extract rolling window features efficiently"""
        df_features = df.copy()
        
        for window in windows:
            window_name = f"{window}s"
            
            for col in self.physiological_cols:
                # Basic rolling statistics
                rolling = df_features[col].rolling(window=window, min_periods=1)
                df_features[f'{col}_mean_{window_name}'] = rolling.mean()
                df_features[f'{col}_std_{window_name}'] = rolling.std()
                
                # Rate of change
                df_features[f'{col}_slope'] = df_features[col].diff()
        
        return df_features
    
    def extract_movement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract movement-related features"""
        df_movement = df.copy()
        
        # Movement magnitude
        df_movement['movement_magnitude'] = np.sqrt(
            df_movement['X']**2 + df_movement['Y']**2 + df_movement['Z']**2
        )
        
        # Activity level classification
        df_movement['activity_level'] = pd.cut(
            df_movement['movement_magnitude'],
            bins=[0, 1, 2, 5, np.inf],
            labels=[0, 1, 2, 3]  # sedentary, light, moderate, vigorous
        )

        # Convert categorical to nullable integer and fill missing as 0 (sedentary)
        try:
            df_movement['activity_level'] = df_movement['activity_level'].astype('Int64').fillna(0)
        except Exception:
            # Fallback: use category codes and replace -1 with 0
            codes = df_movement['activity_level'].cat.codes.replace(-1, 0)
            df_movement['activity_level'] = codes.astype('Int64')
        
        return df_movement
    
    def extract_physiological_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract stress-specific physiological features"""
        df_physio = df.copy()
        
        # Heart rate variability (simple approximation)
        df_physio['HR_variability'] = df_physio['HR'].rolling(
            window=300, min_periods=1
        ).std()
        
        # EDA peaks (stress responses)
        df_physio['EDA_diff'] = df_physio['EDA'].diff()
        df_physio['EDA_peaks'] = (
            (df_physio['EDA_diff'] > 0) & 
            (df_physio['EDA_diff'].shift(-1) < 0)
        ).astype(int)
        
        return df_physio

    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience wrapper that runs all feature extractors and returns a single DataFrame.

        Keeps original `id`, `datetime`, and `label` columns and appends derived features.
        """
        if df.empty:
            return df.copy()

        self.logger.info(f"Extracting features for dataframe with {len(df)} rows")

        # Start from a copy to avoid mutating caller frame
        base = df.copy()

        # Extract movement and physiological features
        try:
            mov = self.extract_movement_features(base)
            phys = self.extract_physiological_features(base)
            roll = self.extract_rolling_features(base)
        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            raise

        # Combine features: prefer computed columns from mov/phys/roll, avoid duplicating original cols
        combined = base[['id', 'datetime', 'label']].copy()

        # Helper to add new columns from a dataframe
        def _add_new_columns(src_df):
            for c in src_df.columns:
                if c in combined.columns:
                    continue
                if c in base.columns and c in ['X', 'Y', 'Z', 'EDA', 'HR', 'TEMP']:
                    # skip raw sensor columns to keep only derived features
                    continue
                combined[c] = src_df[c].values

        _add_new_columns(mov)
        _add_new_columns(phys)
        _add_new_columns(roll)

        self.logger.info(f"Extracted {len([c for c in combined.columns if c not in ['id','datetime','label']])} feature columns")
        return combined
import pandas as pd
import numpy as np
from scipy import signal
from typing import List, Dict

class SignalFeatureExtractor:
    """Extract features from physiological signals"""
    
    def __init__(self, sampling_rate: int = 1):
        self.sampling_rate = sampling_rate
        self.physiological_cols = ['EDA', 'HR', 'TEMP']
        self.movement_cols = ['X', 'Y', 'Z']
    
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
        ).astype(int)
        
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
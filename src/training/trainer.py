import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from typing import Dict, Any, Tuple, List
import logging

class StressDetectionTrainer:
    """Train stress detection models with memory-efficient processing"""
    
    def __init__(self, model, feature_extractor, config: Dict[str, Any]):
        self.model = model
        self.feature_extractor = feature_extractor
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def train_by_subjects(self, data_loader, subject_ids: List[int]) -> Dict:
        """Train model processing one subject at a time"""
        all_features = []
        all_labels = []
        
        for subject_id in subject_ids:
            self.logger.info(f"Processing subject {subject_id}")
            
            # Load subject data
            subject_data = data_loader.load_subject_data(subject_id)
            
            if subject_data.empty:
                continue
            
            # Extract features
            features = self.feature_extractor.extract_all_features(subject_data)
            
            # Prepare for training
            feature_cols = [col for col in features.columns 
                          if col not in ['id', 'datetime', 'label']]
            
            X_subject = features[feature_cols].fillna(0)
            y_subject = features['label'] != 0  # Binary classification
            
            all_features.append(X_subject)
            all_labels.append(y_subject)
        
        # Combine all subjects
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_labels, ignore_index=True)
        
        # Train model
        self.model.fit(X, y)
        
        # Evaluate
        y_pred = self.model.predict(X)
        metrics = {
            'classification_report': classification_report(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save trained model"""
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to {filepath}")
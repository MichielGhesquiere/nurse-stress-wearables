#!/usr/bin/env python3
"""Main training script for stress detection"""

import argparse
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from src.data.data_loader import SensorDataLoader
from src.features.signal_features import SignalFeatureExtractor
from src.training.trainer import StressDetectionTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    data_loader = SensorDataLoader(args.data_path, chunk_size=config['chunk_size'])
    feature_extractor = SignalFeatureExtractor()
    model = RandomForestClassifier(**config['model_params'])
    
    trainer = StressDetectionTrainer(model, feature_extractor, config)
    
    # Get subject list
    subjects = data_loader.get_subject_list()
    
    # Train model
    metrics = trainer.train_by_subjects(data_loader, subjects)
    print("Training completed!")
    print(metrics['classification_report'])
    
    # Save model
    trainer.save_model(config['model_output_path'])

if __name__ == "__main__":
    main()
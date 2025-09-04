#!/usr/bin/env python3
"""Main training script for stress detection"""

import argparse
import yaml
import logging
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from src.data.data_loader import SensorDataLoader
from src.features.signal_features import SignalFeatureExtractor
from src.training.trainer import StressDetectionTrainer
from src.utils.plotting import save_combined_roc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    args = parser.parse_args()
    # Basic logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger('train_model')

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {args.config}")
    
    # Initialize components
    # Ensure results directory exists
    results_dir = config.get('results_dir')
    if results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Results will be saved under: {results_dir}")

    data_loader = SensorDataLoader(args.data_path, chunk_size=config['chunk_size'])
    feature_extractor = SignalFeatureExtractor()
    model = RandomForestClassifier(**config['model_params'])
    logger.info("Initialized data loader, feature extractor and model")
    
    trainer = StressDetectionTrainer(model, feature_extractor, config)
    
    # Get subject list
    logger.info("Gathering subject list from data")
    subjects = data_loader.get_subject_list()
    logger.info(f"Found {len(subjects)} subjects to process")
    
    # Decide which pipeline(s) to run
    model_type = str(config.get('model_type', 'rf')).lower()
    summary = {}

    if model_type in ('rf', 'both'):
        # Prefer training from cached features if configured and present
        use_feature_store = bool(config.get('use_feature_store', True))
        feature_store_root = config.get('feature_store_root', str(Path('data') / 'features'))
        feature_set = config.get('feature_set', 'v1')
        metrics_rf = None
        if use_feature_store:
            try:
                logger.info("RF: training from cached feature store")
                metrics_rf = trainer.train_from_feature_store(feature_store_root, feature_set=feature_set)
            except Exception as e:
                logger.warning(f"RF: cached feature training failed ({e}); falling back to on-the-fly extraction")
        if metrics_rf is None:
            logger.info("Starting RF training by subject (engineered features)")
            metrics_rf = trainer.train_by_subjects(data_loader, subjects)
        logger.info("RF training completed")
        logger.info("RF Classification report:\n%s", metrics_rf.get('classification_report', 'N/A'))
        trainer.save_model(config['model_output_path'])
        logger.info(f"Saved RF model to {config['model_output_path']}")
        summary['rf'] = metrics_rf
        rf_curve = (
            'RF',
            metrics_rf.get('y_true'),
            metrics_rf.get('y_score')
        ) if metrics_rf.get('y_true') is not None and metrics_rf.get('y_score') is not None else None

    if model_type in ('dl', 'both'):
        try:
            from src.training.dl_trainer import DLTrainer
            dl_tr = DLTrainer(config)
            logger.info("Building DL dataset (raw windows)")
            ds, signals = dl_tr.build_dataset(data_loader, subjects)
            model_obj = dl_tr.train(ds)
            metrics_dl = dl_tr.evaluate(model_obj, ds)
            logger.info("DL training completed")
            logger.info("DL Classification report:\n%s", metrics_dl.get('classification_report', 'N/A'))
            summary['dl'] = metrics_dl
            # Save DL model state dict
            if results_dir:
                dl_model_path = Path(results_dir) / 'model_dl.pt'
                try:
                    dl_tr.save_model(model_obj, str(dl_model_path))
                    logger.info(f"Saved DL model to {dl_model_path}")
                except Exception as e:
                    logger.warning(f"Saving DL model failed: {e}")
            # Save curves for DL if scores available
            if results_dir and metrics_dl.get('y_score') is not None:
                try:
                    from src.utils.plotting import save_curves
                    import numpy as np
                    curves_dir = Path(results_dir) / 'curves'
                    curves_dir.mkdir(parents=True, exist_ok=True)
                    y_true = np.array(metrics_dl['y_true'])
                    y_score = np.array(metrics_dl['y_score'])
                    save_curves(y_true, y_score, str(curves_dir), prefix='dl_')
                except Exception as e:
                    logger.warning(f"Saving DL curves failed: {e}")
            dl_curve = (
                'DL',
                np.array(metrics_dl.get('y_true')) if metrics_dl.get('y_true') is not None else None,
                np.array(metrics_dl.get('y_score')) if metrics_dl.get('y_score') is not None else None,
            )
        except Exception as e:
            logger.exception(f"DL pipeline failed: {e}")

    # Combined ROC across RF and DL
    try:
        curves = []
        if 'rf' in summary and summary['rf'].get('y_true') is not None and summary['rf'].get('y_score') is not None:
            import numpy as np
            curves.append(('RF', np.array(summary['rf']['y_true']), np.array(summary['rf']['y_score'])))
        if 'dl' in summary and summary['dl'].get('y_true') is not None and summary['dl'].get('y_score') is not None:
            import numpy as np
            curves.append(('DL', np.array(summary['dl']['y_true']), np.array(summary['dl']['y_score'])))
        if results_dir and curves:
            curves_dir = Path(results_dir) / 'curves'
            curves_dir.mkdir(parents=True, exist_ok=True)
            save_combined_roc(curves, str(curves_dir / 'roc_rf_vs_dl.png'), title='ROC: RF vs DL')
            logger.info("Saved combined ROC: curves/roc_rf_vs_dl.png")
    except Exception as e:
        logger.warning(f"Combined ROC plotting failed: {e}")

    # Write comparison summary if results_dir provided
    results_dir = config.get('results_dir')
    if results_dir and summary:
        out = Path(results_dir) / 'comparison_summary.txt'
        with open(out, 'w') as f:
            for k, v in summary.items():
                f.write(f"=== {k.upper()} ===\n")
                f.write(v.get('classification_report', 'N/A'))
                f.write("\n\n")
        logger.info(f"Saved comparison summary to {out}")

if __name__ == "__main__":
    main()
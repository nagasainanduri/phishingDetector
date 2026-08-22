# scripts/run_benchmarks.py
import sys
import os
import json
import logging
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.data import DataLoader, DatasetSplitter
from training.models.rf_baseline import RandomForestTrainer

from training.models.xgboost_trainer import XGBoostTrainer
from training.models.catboost_trainer import CatBoostTrainer
from training.models.cnn_trainer import CharCNNTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run():
    logger.info("Initializing Data Loader")
    loader = DataLoader()
    X, y, urls = loader.get_features()
    
    logger.info(f"Total dataset size: {len(X)}")
    
    logger.info("Performing domain-aware split")
    X_train, X_test, y_train, y_test = DatasetSplitter.split(X, y, urls, test_size=0.2, random_state=42, domain_aware=True)
    
    # Pass URLs to training for text-based models
    urls_train = urls[X_train.index]
    urls_test = urls[X_test.index]
    
    logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    trainers = [
        RandomForestTrainer(),
        XGBoostTrainer(),
        CatBoostTrainer(),
        CharCNNTrainer()
    ]
    
    reports = {}
    
    for trainer in trainers:
        logger.info(f"--- Benchmarking {trainer.model_name} ---")
        try:
            logger.info("Training model...")
            trainer.train(X_train, y_train, urls_train)
            
            logger.info("Evaluating model...")
            metrics = trainer.evaluate(X_test, y_test, urls_test)
            
            logger.info("Saving model and metrics")
            trainer.save()
            
            reports[trainer.model_name] = {
                'timestamp': datetime.now().isoformat(),
                'model': trainer.model_name,
                'metrics': metrics
            }
            
            if hasattr(trainer.model, 'feature_importances_'):
                importances = dict(zip(X_train.columns, trainer.model.feature_importances_))
                importances = {k: float(v) for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}
                reports[trainer.model_name]['feature_importances'] = importances

            logger.info(json.dumps(metrics['evaluation'], indent=4))
        except Exception as e:
            logger.error(f"Failed to benchmark {trainer.model_name}: {e}")
            reports[trainer.model_name] = {'error': str(e)}
            
    os.makedirs('benchmarks', exist_ok=True)
    report_path = 'benchmarks/comparison_report.json'
    with open(report_path, 'w') as f:
        json.dump(reports, f, indent=4)
        
    logger.info(f"Combined benchmark report saved to {report_path}")

if __name__ == '__main__':
    run()

# scripts/run_benchmarks.py
import sys
import os
import json
import logging
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.data import DataLoader, DatasetSplitter
from training.models.rf_baseline import RandomForestTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run():
    logger.info("Initializing Data Loader")
    loader = DataLoader()
    X, y, urls = loader.get_features()
    
    logger.info(f"Total dataset size: {len(X)}")
    
    logger.info("Performing domain-aware split")
    X_train, X_test, y_train, y_test = DatasetSplitter.split(X, y, urls, test_size=0.2, random_state=42, domain_aware=True)
    
    logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    logger.info("Initializing Random Forest Baseline Trainer")
    trainer = RandomForestTrainer()
    
    logger.info("Training model...")
    trainer.train(X_train, y_train)
    
    logger.info("Evaluating model...")
    metrics = trainer.evaluate(X_test, y_test)
    
    logger.info("Saving model and metrics")
    trainer.save()
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'model': 'RandomForest Baseline',
        'split': {
            'strategy': 'Domain-Aware GroupShuffleSplit',
            'train_size': len(X_train),
            'test_size': len(X_test),
        },
        'metrics': metrics
    }
    
    os.makedirs('benchmarks', exist_ok=True)
    report_path = 'benchmarks/rf_baseline.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"Benchmark report saved to {report_path}")
    logger.info(json.dumps(metrics['evaluation'], indent=4))

if __name__ == '__main__':
    run()

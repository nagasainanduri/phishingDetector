# training/metrics.py
import time
import tracemalloc
import psutil
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import logging

logger = logging.getLogger(__name__)

def calculate_classification_metrics(y_true, y_pred, y_prob=None):
    """Calculates a comprehensive set of classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        except ValueError:
            pass # Handle single class cases gracefully
            
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
    return metrics

class ResourceTracker:
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        
    def start(self):
        tracemalloc.start()
        self.start_time = time.time()
        self.start_memory = tracemalloc.get_traced_memory()[0]
        
    def stop(self):
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'time_seconds': end_time - self.start_time,
            'peak_memory_mb': peak / (1024 * 1024),
            'process_memory_mb': memory_info.rss / (1024 * 1024)
        }

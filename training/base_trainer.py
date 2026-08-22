# training/base_trainer.py
from abc import ABC, abstractmethod
import pickle
import os
from .metrics import calculate_classification_metrics, ResourceTracker

class BaseTrainer(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.metrics = {}
        
    @abstractmethod
    def build_model(self):
        pass
        
    def train(self, X_train, y_train, urls_train=None):
        if self.model is None:
            self.build_model()
            
        tracker = ResourceTracker()
        tracker.start()
        
        self.model.fit(X_train, y_train)
        
        res = tracker.stop()
        self.metrics['training_resources'] = res
        return res
        
    def evaluate(self, X_test, y_test, urls_test=None):
        tracker = ResourceTracker()
        tracker.start()
        
        y_pred = self.model.predict(X_test)
        
        inference_res = tracker.stop()
        self.metrics['inference_resources'] = inference_res
        
        # Calculate latency per sample
        if len(X_test) > 0:
            self.metrics['inference_resources']['latency_per_sample_ms'] = (inference_res['time_seconds'] / len(X_test)) * 1000
            
        y_prob = None
        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(X_test)[:, 1]
            
        clf_metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
        self.metrics['evaluation'] = clf_metrics
        return self.metrics
        
    def save(self, output_dir='models/benchmarks'):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{self.model_name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        return path

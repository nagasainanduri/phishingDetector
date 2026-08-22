# training/models/rf_baseline.py
from sklearn.ensemble import RandomForestClassifier
from ..base_trainer import BaseTrainer

class RandomForestTrainer(BaseTrainer):
    def __init__(self):
        super().__init__(model_name="rf_baseline")
        
    def build_model(self):
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=15, 
            random_state=42, 
            n_jobs=-1, 
            class_weight='balanced'
        )

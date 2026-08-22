from catboost import CatBoostClassifier
from ..base_trainer import BaseTrainer

class CatBoostTrainer(BaseTrainer):
    def __init__(self):
        super().__init__(model_name="catboost")
        
    def build_model(self):
        self.model = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            task_type='GPU',
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=0
        )

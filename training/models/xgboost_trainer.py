from xgboost import XGBClassifier
from ..base_trainer import BaseTrainer

class XGBoostTrainer(BaseTrainer):
    def __init__(self):
        super().__init__(model_name="xgboost")
        
    def build_model(self):
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            tree_method='hist',
            device='cuda',
            scale_pos_weight=1.0,
            random_state=42
        )

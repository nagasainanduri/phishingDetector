import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from ..base_trainer import BaseTrainer
from ..metrics import ResourceTracker, calculate_classification_metrics

logger = logging.getLogger(__name__)

class CharCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, num_filters=128, kernel_size=5):
        super(CharCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(embedding_dim, num_filters, kernel_size)
        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(num_filters, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.maxpool(x).squeeze(2)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(1)

class CharCNNClassifier:
    def __init__(self, max_len=200, batch_size=256, epochs=5, lr=0.001):
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab_size = 256
        self.model = None

    def _tokenize(self, urls):
        X = np.zeros((len(urls), self.max_len), dtype=np.int64)
        for i, url in enumerate(urls):
            for j, char in enumerate(str(url)[:self.max_len]):
                code = ord(char)
                X[i, j] = code if code < 256 else 255
        return torch.tensor(X, dtype=torch.long)

    def fit(self, urls_train, y_train):
        self.model = CharCNN(self.vocab_size).to(self.device)
        X_tensor = self._tokenize(urls_train)
        y_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.4f}")

    def predict_proba(self, urls_test):
        self.model.eval()
        X_tensor = self._tokenize(urls_test)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        probs = []
        with torch.no_grad():
            for inputs in loader:
                inputs = inputs[0].to(self.device)
                outputs = self.model(inputs)
                probs.extend(torch.sigmoid(outputs).cpu().numpy())
                
        probs_np = np.array(probs)
        return np.column_stack((1 - probs_np, probs_np))

    def predict(self, urls_test):
        probs = self.predict_proba(urls_test)[:, 1]
        return (probs > 0.5).astype(int)

class CharCNNTrainer(BaseTrainer):
    def __init__(self):
        super().__init__(model_name="char_cnn")
        
    def build_model(self):
        self.model = CharCNNClassifier()

    def train(self, X_train, y_train, urls_train=None):
        if self.model is None:
            self.build_model()
        tracker = ResourceTracker()
        tracker.start()
        
        self.model.fit(urls_train, y_train)
        
        res = tracker.stop()
        self.metrics['training_resources'] = res
        return res

    def evaluate(self, X_test, y_test, urls_test=None):
        tracker = ResourceTracker()
        tracker.start()
        
        y_pred = self.model.predict(urls_test)
        
        inference_res = tracker.stop()
        self.metrics['inference_resources'] = inference_res
        
        if len(X_test) > 0:
            self.metrics['inference_resources']['latency_per_sample_ms'] = (inference_res['time_seconds'] / len(X_test)) * 1000
            
        y_prob = self.model.predict_proba(urls_test)[:, 1]
            
        clf_metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
        self.metrics['evaluation'] = clf_metrics
        return self.metrics

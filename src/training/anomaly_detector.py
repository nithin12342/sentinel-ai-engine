"""
Anomaly Detection Model
Detects unusual patterns in sensor data and transactions using Autoencoder
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from typing import Tuple, List, Dict
import mlflow
from sklearn.metrics import precision_score, recall_score, f1_score


class AnomalyDetector(nn.Module):
    """Autoencoder-based anomaly detector"""
    
    def __init__(self, input_dim: int, encoding_dim: int = 32):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, encoding_dim),
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_encoding(self, x):
        return self.encoder(x)


class AnomalyDataset(Dataset):
    """Dataset for anomaly detection"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray = None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], torch.tensor(self.labels[idx], dtype=torch.float32)
        return self.features[idx]


class AnomalyTrainer:
    """Trainer for anomaly detection model"""
    
    def __init__(self, model: AnomalyDetector, threshold: float = 0.1):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.threshold = threshold
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            if isinstance(batch, tuple):
                features = batch[0]
            else:
                features = batch
            
            self.optimizer.zero_grad()
            
            reconstruction = self.model(features)
            loss = self.criterion(reconstruction, features)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def compute_anomaly_scores(self, dataloader: DataLoader) -> np.ndarray:
        """Compute reconstruction error as anomaly score"""
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, tuple):
                    features = batch[0]
                else:
                    features = batch
                
                reconstruction = self.model(features)
                errors = torch.mean((reconstruction - features) ** 2, dim=1)
                scores.extend(errors.numpy())
        
        return np.array(scores)
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies"""
        dataset = TensorDataset(torch.tensor(features, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=32)
        
        scores = self.compute_anomaly_scores(dataloader)
        predictions = (scores > self.threshold).astype(int)
        
        return predictions, scores


def train_anomaly_detector(
    train_features: np.ndarray,
    val_features: np.ndarray,
    input_dim: int,
    epochs: int = 100,
    batch_size: int = 64,
    encoding_dim: int = 32,
    threshold: float = 0.1
):
    """Train anomaly detection model"""
    
    mlflow.set_experiment("anomaly_detection")
    
    # Create data loaders
    train_dataset = AnomalyDataset(train_features)
    val_dataset = AnomalyDataset(val_features)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = AnomalyDetector(input_dim, encoding_dim)
    trainer = AnomalyTrainer(model, threshold)
    
    best_loss = float('inf')
    
    with mlflow.start_run():
        mlflow.log_params({
            'input_dim': input_dim,
            'encoding_dim': encoding_dim,
            'epochs': epochs,
            'batch_size': batch_size,
            'threshold': threshold,
        })
        
        for epoch in range(epochs):
            train_loss = trainer.train_epoch(train_loader)
            
            # Validation
            val_scores = trainer.compute_anomaly_scores(val_loader)
            val_loss = np.mean(val_scores)
            
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, step=epoch)
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), "models/anomaly_detector.pt")
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    return model, trainer


def detect_streaming_anomalies(
    model: AnomalyDetector,
    feature_window: List[np.ndarray],
    threshold: float = 0.1
) -> Dict:
    """Detect anomalies in streaming data"""
    model.eval()
    
    with torch.no_grad():
        features = torch.tensor(np.array(feature_window), dtype=torch.float32)
        reconstruction = model(features)
        errors = torch.mean((reconstruction - features) ** 2, dim=1)
        
        is_anomaly = errors > threshold
        anomaly_score = errors.numpy()
    
    return {
        'is_anomaly': is_anomaly.any().item(),
        'anomaly_scores': anomaly_score.tolist(),
        'max_score': anomaly_score.max(),
        'avg_score': anomaly_score.mean(),
    }


if __name__ == "__main__":
    print("Anomaly detection model configured")

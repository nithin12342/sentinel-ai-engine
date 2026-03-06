"""
Agricultural Prediction Model Training
Predicts crop yield, disease risk, and optimal harvest time
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import mlflow
from mlflow.tracking import MlflowClient

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "models/agricultural"
MLFLOW_TRACKING_URI = "http://mlflow:5000"


class AgriculturalDataset(Dataset):
    """Dataset for agricultural data"""
    
    def __init__(self, data_path: str, transform=None):
        self.data = pd.read_csv(data_path)
        self.transform = transform
        
        # Features: temperature, humidity, rainfall, soil_moisture, sunlight_hours
        self.feature_cols = ['temperature', 'humidity', 'rainfall', 'soil_moisture', 'sunlight_hours']
        # Targets: yield_prediction, disease_risk, harvest_day
        self.target_cols = ['yield_prediction', 'disease_risk', 'harvest_day']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.data.iloc[idx][self.feature_cols].values, dtype=torch.float32)
        targets = torch.tensor(self.data.iloc[idx][self.target_cols].values, dtype=torch.float32)
        
        if self.transform:
            features = self.transform(features)
        
        return features, targets


class AgriculturalPredictor(nn.Module):
    """Multi-task neural network for agricultural predictions"""
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 128):
        super().__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
        )
        
        # Task-specific heads
        self.yield_head = nn.Linear(hidden_dim, 1)
        self.disease_head = nn.Linear(hidden_dim, 1)
        self.harvest_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        encoded = self.encoder(x)
        
        yield_pred = torch.sigmoid(self.yield_head(encoded))
        disease_risk = torch.sigmoid(self.disease_head(encoded))
        harvest_day = torch.relu(self.harvest_head(encoded))
        
        return yield_pred, disease_risk, harvest_day


class AgriculturalModelTrainer:
    """Trainer for agricultural prediction models"""
    
    def __init__(self, model: AgriculturalPredictor, learning_rate: float = 0.001):
        self.model = model.to(DEVICE)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Multi-task loss
        self.yield_criterion = nn.MSELoss()
        self.disease_criterion = nn.BCELoss()
        self.harvest_criterion = nn.MSELoss()
        
        # Weights for each task
        self.task_weights = {
            'yield': 1.0,
            'disease': 1.5,  # Higher weight for disease prediction
            'harvest': 1.0
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        metrics = {'yield_loss': 0, 'disease_loss': 0, 'harvest_loss': 0}
        
        for features, targets in dataloader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            
            self.optimizer.zero_grad()
            
            yield_pred, disease_pred, harvest_pred = self.model(features)
            
            # Calculate losses
            yield_loss = self.yield_criterion(yield_pred, targets[:, 0:1])
            disease_loss = self.disease_criterion(disease_pred, targets[:, 1:2])
            harvest_loss = self.harvest_criterion(harvest_pred, targets[:, 2:3])
            
            # Combined loss
            loss = (
                self.task_weights['yield'] * yield_loss +
                self.task_weights['disease'] * disease_loss +
                self.task_weights['harvest'] * harvest_loss
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            metrics['yield_loss'] += yield_loss.item()
            metrics['disease_loss'] += disease_loss.item()
            metrics['harvest_loss'] += harvest_loss.item()
        
        # Average metrics
        n = len(dataloader)
        return {
            'train_loss': total_loss / n,
            'yield_loss': metrics['yield_loss'] / n,
            'disease_loss': metrics['disease_loss'] / n,
            'harvest_loss': metrics['harvest_loss'] / n,
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        metrics = {'yield_loss': 0, 'disease_loss': 0, 'harvest_loss': 0}
        
        with torch.no_grad():
            for features, targets in dataloader:
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                
                yield_pred, disease_pred, harvest_pred = self.model(features)
                
                yield_loss = self.yield_criterion(yield_pred, targets[:, 0:1])
                disease_loss = self.disease_criterion(disease_pred, targets[:, 1:2])
                harvest_loss = self.harvest_criterion(harvest_pred, targets[:, 2:3])
                
                loss = (
                    self.task_weights['yield'] * yield_loss +
                    self.task_weights['disease'] * disease_loss +
                    self.task_weights['harvest'] * harvest_loss
                )
                
                total_loss += loss.item()
                metrics['yield_loss'] += yield_loss.item()
                metrics['disease_loss'] += disease_loss.item()
                metrics['harvest_loss'] += harvest_loss.item()
        
        n = len(dataloader)
        return {
            'val_loss': total_loss / n,
            'yield_loss': metrics['yield_loss'] / n,
            'disease_loss': metrics['disease_loss'] / n,
            'harvest_loss': metrics['harvest_loss'] / n,
        }


def train_agricultural_model(
    train_data_path: str,
    val_data_path: str,
    epochs: int = 50,
    batch_size: int = 32,
    experiment_name: str = "agricultural_prediction"
):
    """Main training function with MLflow tracking"""
    
    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    
    # Create datasets
    train_dataset = AgriculturalDataset(train_data_path)
    val_dataset = AgriculturalDataset(val_data_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model and trainer
    model = AgriculturalPredictor(input_dim=5, hidden_dim=128)
    trainer = AgriculturalModelTrainer(model, learning_rate=0.001)
    
    # Training loop
    best_val_loss = float('inf')
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': 0.001,
            'hidden_dim': 128,
            'model_type': 'AgriculturalPredictor',
        })
        
        for epoch in range(epochs):
            train_metrics = trainer.train_epoch(train_loader)
            val_metrics = trainer.validate(val_loader)
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                'train_loss': train_metrics['train_loss'],
                'val_loss': val_metrics['val_loss'],
                'yield_loss': val_metrics['yield_loss'],
                'disease_loss': val_metrics['disease_loss'],
                'harvest_loss': val_metrics['harvest_loss'],
            }, step=epoch)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                torch.save(model.state_dict(), f"{MODEL_DIR}/best_model.pt")
                mlflow.pytorch.log_model(model, "best_model")
        
        # Log final metrics
        mlflow.log_metrics({
            'best_val_loss': best_val_loss,
        })
    
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    return model


if __name__ == "__main__":
    # This would run with actual data
    print("Agricultural model training configured")

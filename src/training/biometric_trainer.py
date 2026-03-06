"""
Biometric Authentication Model Training
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
from pathlib import Path
from datetime import datetime
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiometricModel(nn.Module):
    """Face recognition model using InsightFace backbone"""
    
    def __init__(self, num_classes: int = 1000, embedding_dim: int = 512):
        super().__init__()
        # Using a simple CNN backbone for demonstration
        # In production, use InsightFace/ArcFace pretrained model
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,  nn.Conv2d2),
            
           (64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.embedding = nn.Linear(512, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        embedding = self.embedding(x)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        logits = self.classifier(embedding)
        return logits, embedding


class BiometricTrainer:
    """Trainer for biometric authentication models"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.criterion = None
        
    def setup(self):
        """Initialize model, optimizer, and loss"""
        self.model = BiometricModel(
            num_classes=self.config.get("num_classes", 1000),
            embedding_dim=self.config.get("embedding_dim", 512)
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 1e-4),
            weight_decay=self.config.get("weight_decay", 1e-4)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Model initialized on {self.device}")
        
    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            logits, embeddings = self.model(images)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return {
            "loss": total_loss / len(dataloader),
            "accuracy": 100. * correct / total
        }
    
    def validate(self, dataloader: DataLoader) -> dict:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits, embeddings = self.model(images)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return {
            "loss": total_loss / len(dataloader),
            "accuracy": 100. * correct / total
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        """Full training loop with MLflow tracking"""
        self.setup()
        
        with mlflow.start_run(run_name=f"biometric_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_params({
                "num_classes": self.config.get("num_classes", 1000),
                "embedding_dim": self.config.get("embedding_dim", 512),
                "learning_rate": self.config.get("learning_rate", 1e-4),
                "batch_size": self.config.get("batch_size", 32),
                "num_epochs": num_epochs
            })
            
            best_val_acc = 0
            
            for epoch in range(num_epochs):
                train_metrics = self.train_epoch(train_loader)
                val_metrics = self.validate(val_loader)
                
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.2f}% - "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.2f}%"
                )
                
                mlflow.log_metrics({
                    "train_loss": train_metrics['loss'],
                    "train_accuracy": train_metrics['accuracy'],
                    "val_loss": val_metrics['loss'],
                    "val_accuracy": val_metrics['accuracy']
                }, step=epoch)
                
                # Save best model
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    self.save_checkpoint("best_model.pt")
                    
            logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
            
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config
        }
        torch.save(checkpoint, path)
        mlflow.log_artifact(path)
        logger.info(f"Model saved to {path}")
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.setup()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Model loaded from {path}")


def main():
    """Example training run"""
    config = {
        "num_classes": 100,
        "embedding_dim": 512,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 32
    }
    
    # In production, load actual data
    # train_dataset = FaceDataset(...)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    trainer = BiometricTrainer(config)
    
    # Training would use actual data loaders
    # trainer.train(train_loader, val_loader, num_epochs=10)


if __name__ == "__main__":
    main()

import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

class PyTorchAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32, 16]):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(curr_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.ReLU())
            curr_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        rev_dims = hidden_dims[::-1][1:] + [input_dim]
        for h_dim in rev_dims[:-1]:
            decoder_layers.append(nn.Linear(curr_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.ReLU())
            curr_dim = h_dim
        
        decoder_layers.append(nn.Linear(curr_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

class UnsupervisedAnomalyDetector:
    def __init__(
        self,
        model_type: str = "autoencoder",
        input_dim: int = 30,
        hidden_dims: list = [64, 32, 16],
        learning_rate: float = 0.001,
        epochs: int = 40,
        batch_size: int = 16,
        threshold_percentile: float = 99.0,
        random_seed: int = 42
    ):
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_percentile = threshold_percentile
        self.random_seed = random_seed

        self.scaler = StandardScaler()
        self.threshold = 0.0
        self.model = None

    def fit(self, X_train_normal: np.ndarray):
        # 1. Fit scaler ONLY on normal train data
        X_scaled = self.scaler.fit_transform(X_train_normal)

        if self.model_type == "autoencoder":
            torch.manual_seed(self.random_seed)
            self.model = PyTorchAutoencoder(self.input_dim, self.hidden_dims)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()

            dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
            loader = DataLoader(dataset, batch_size=min(self.batch_size, len(X_train_normal)), shuffle=True)

            self.model.train()
            for epoch in range(self.epochs):
                for (batch_x,) in loader:
                    optimizer.zero_grad()
                    recon = self.model(batch_x)
                    loss = criterion(recon, batch_x)
                    loss.backward()
                    optimizer.step()

            # Derive Anomaly Threshold on Normal Train Reconstruction Errors
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                recon_train = self.model(X_tensor)
                train_errors = torch.mean((recon_train - X_tensor)**2, dim=1).numpy()
                self.threshold = float(np.percentile(train_errors, self.threshold_percentile))

        elif self.model_type == "isolation_forest":
            self.model = IsolationForest(
                contamination=1.0 - (self.threshold_percentile / 100.0),
                random_state=self.random_seed
            )
            self.model.fit(X_scaled)
            train_scores = -self.model.score_samples(X_scaled)
            self.threshold = float(np.percentile(train_scores, self.threshold_percentile))

    def compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)

        if self.model_type == "autoencoder":
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                recon = self.model(X_tensor)
                errors = torch.mean((recon - X_tensor)**2, dim=1).numpy()
            return errors

        elif self.model_type == "isolation_forest":
            scores = -self.model.score_samples(X_scaled)
            return scores

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = self.compute_anomaly_scores(X)
        predictions = (scores >= self.threshold).astype(int)
        return predictions, scores

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        meta = {
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "threshold": self.threshold,
            "threshold_percentile": self.threshold_percentile,
            "random_seed": self.random_seed
        }
        joblib.dump(meta, os.path.join(save_dir, "metadata.pkl"))
        joblib.dump(self.scaler, os.path.join(save_dir, "scaler.pkl"))

        if self.model_type == "autoencoder":
            torch.save(self.model.state_dict(), os.path.join(save_dir, "autoencoder.pt"))
        else:
            joblib.dump(self.model, os.path.join(save_dir, "isolation_forest.pkl"))

    @classmethod
    def load(cls, load_dir: str):
        meta = joblib.load(os.path.join(load_dir, "metadata.pkl"))
        instance = cls(
            model_type=meta["model_type"],
            input_dim=meta["input_dim"],
            hidden_dims=meta["hidden_dims"],
            threshold_percentile=meta["threshold_percentile"],
            random_seed=meta["random_seed"]
        )
        instance.threshold = meta["threshold"]
        instance.scaler = joblib.load(os.path.join(load_dir, "scaler.pkl"))

        if instance.model_type == "autoencoder":
            instance.model = PyTorchAutoencoder(instance.input_dim, instance.hidden_dims)
            instance.model.load_state_dict(torch.load(os.path.join(load_dir, "autoencoder.pt")))
            instance.model.eval()
        else:
            instance.model = joblib.load(os.path.join(load_dir, "isolation_forest.pkl"))

        return instance

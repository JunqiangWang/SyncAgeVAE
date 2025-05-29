import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ----- 1. Create Synthetic Dataset -----
class GeneExpressionDataset(Dataset):
    def __init__(self, n_cells=1000, n_genes=2000, n_cell_types=5, n_age_classes=4):
        self.X_gene = np.random.rand(n_cells, n_genes).astype(np.float32)
        self.cell_types = np.random.randint(0, n_cell_types, size=(n_cells,))
        self.age_labels = np.random.randint(0, n_age_classes, size=(n_cells,))

    def __len__(self):
        return len(self.X_gene)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_gene[idx]),
            torch.tensor(self.cell_types[idx], dtype=torch.long),
            torch.tensor(self.age_labels[idx], dtype=torch.long)
        )

# ----- 2. Define Model -----
class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=256):
        super(VAEEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=256):
        super(VAEDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))

class VAEClassifier(nn.Module):
    def __init__(self, input_dim, n_cell_types, latent_dim, n_classes, cell_type_emb_dim=16):
        super(VAEClassifier, self).__init__()
        self.encoder = VAEEncoder(input_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, input_dim)
        self.cell_type_emb = nn.Embedding(n_cell_types, cell_type_emb_dim)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim + cell_type_emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )

    def forward(self, x_gene, cell_type):
        z, mu, logvar = self.encoder(x_gene)
        cell_emb = self.cell_type_emb(cell_type)
        features = torch.cat([z, cell_emb], dim=1)
        logits = self.classifier(features)
        recon = self.decoder(z)
        return logits, mu, logvar, recon

# ----- 3. Loss Functions -----
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

classification_loss_fn = nn.CrossEntropyLoss()

# ----- 4. Training Function -----
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for x_gene, cell_type, age_label in dataloader:
        x_gene, cell_type, age_label = x_gene.to(device), cell_type.to(device), age_label.to(device)

        optimizer.zero_grad()
        logits, mu, logvar, recon = model(x_gene, cell_type)
        class_loss = classification_loss_fn(logits, age_label)
        vae_l = vae_loss(recon, x_gene, mu, logvar)
        loss = class_loss + vae_l
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ----- 5. Run Everything -----
def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GeneExpressionDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = 2000
    n_cell_types = 5
    latent_dim = 32
    n_classes = 4

    model = VAEClassifier(input_dim, n_cell_types, latent_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        loss = train(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1} - Loss: {loss:.4f}")




class AnnDataset(Dataset):
    def __init__(self, adata):
        self.X = adata.X.toarray().astype(np.float32)  # dense matrix
        self.cell_types = adata.obs['cell_type_encoded'].values.astype(np.int64)
        self.age_labels = adata.obs['age_class'].values.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx]),
            torch.tensor(self.cell_types[idx]),
            torch.tensor(self.age_labels[idx])
        )


#if __name__ == "__main__":
 #   run_training()


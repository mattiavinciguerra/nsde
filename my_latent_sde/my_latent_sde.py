#!/usr/bin/env python
# coding: utf-8

# # Loading data

# In[ ]:


import pickle

with open("../dataset/fixs.pkl", "rb") as f:
    fixs = pickle.load(f)

#print(len(fixs), len(fixs[0]), len(fixs[0][0]), len(fixs[0][0][0]))
# n_sbj x n_img x n_fix x n_coord x 2


# # Encoder

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchsde
from torch.utils.data import DataLoader


# In[3]:


from torch.nn.utils.rnn import pack_padded_sequence

# Encoder: mappa le coordinate delle fissazioni nello spazio latente
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.project = nn.Linear(hidden_size, latent_size)

    def forward(self, batch, lengths):
        """
        batch: [B, T, 2] - Batches of fixations (Batch size, Sequence length, 2 coordinates)
        lengths: [B] - Lengths of each fixation in the batch
        """
        packed = pack_padded_sequence(batch, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, last_hidden_states = self.gru(packed)
        # packed_out: [B, T, H] - Output of the GRU for each time step, not used here
        # last_hidden_states: [1, B, H] - Last hidden state for each fixation in the batch
        last_hidden_states = last_hidden_states.squeeze(0)  # [1, B, H] -> [B, H]
        latent_states = self.project(last_hidden_states)  # [B, latent_size]
        return latent_states


# # SDE

# In[ ]:


class LatentSDE(torchsde.SDEIto):
    def __init__(self, input_size, hidden_size, latent_size, device):
        super().__init__(noise_type="diagonal")
        self.device = device

        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size, latent_size=latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

        self.drift_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, latent_size)
        )

        self.diffusion_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, latent_size)
        )

    def f(self, t, y):
        return self.drift_net(y)

    def g(self, t, y):
        return self.diffusion_net(y)

    def encode(self, batch, mask):
        """
        batch: [B, T, 2] - Batches of fixations (Batch size, Sequence length, 2 coordinates)
        lengths: [B] - Lengths of each fixation in the batch
        """

        lengths = mask.sum(dim=1).long() # Fixations lengths

        return self.encoder(batch, lengths)

    def generate_fixation(self, z0, length):
        """
        Genera una sequenza di coordinate a partire da uno stato latente iniziale.

        Args:
            z0 (torch.Tensor): Stato latente iniziale, shape [1, latent_size].
            length (int): Numero di passi temporali da simulare.
            t0 (float): Tempo iniziale.
            t1 (float): Tempo finale.
            device (str): 'cpu' o 'cuda'.

        Returns:
            torch.Tensor: Sequenza generata di coordinate, shape [length, 2].
        """
        ts = torch.linspace(0, 1, length, device=self.device)  # [length] - Time steps for the SDE
        zs = torchsde.sdeint(self, z0, ts)  # [length, 1, latent_size]
        decoded = self.decoder(zs)  # [length, 1, 2]
        return decoded.squeeze(1)  # [length, 2]

    def forward(self, batch, mask):
        """
        batch: [B, T, 2] - Batches of fixations (Batch size, Sequence length, 2 coordinates)
        lengths: [B] - Lengths of each fixation in the batch
        """
        ts = torch.linspace(0, 1, batch.size(1), device=self.device) # [T] - Time steps for the SDE

        lengths = mask.sum(dim=1).long() # Fixations lengths
        latent_states = self.encoder(batch, lengths) # [B, latent_size] - z0

        zs = torchsde.sdeint(self, latent_states, ts) # [T, B, latent_size]
        zs = zs.permute(1, 0, 2) # [B, T, latent_size]

        recon_x = self.decoder(zs) # [B, T, input_size]

        return recon_x

class EarlyStopping:
    def __init__(self, patience=30, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# # DataLoader

# In[5]:


from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    lengths = torch.tensor([seq.shape[0] for seq in batch])
    # Padding the sequences to the maximum length in the batch
    padded = pad_sequence(batch, batch_first=True) # [B, T, 2]
    mask = torch.arange(padded.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.float()  # Convert to float for compatibility with loss functions
    return padded, mask


# In[6]:


import random

def create_dataloaders(sbj_fixs, batch_size):
    random.shuffle(sbj_fixs)

    # 70% training, 15% validation, 15% test
    train_size = max(1, int(0.7 * len(sbj_fixs)))
    val_size = int(0.15 * len(sbj_fixs))
    train_set = [torch.tensor(fix, dtype=torch.float) for img in sbj_fixs[:train_size] for fix in img]
    val_set = [torch.tensor(fix, dtype=torch.float) for img in sbj_fixs[train_size:train_size + val_size] for fix in img]
    test_set = [torch.tensor(fix, dtype=torch.float) for img in sbj_fixs[train_size + val_size:] for fix in img]

    # DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


# # Parameters

# In[7]:


from geomloss import SamplesLoss

latent_size = 16 # Dimensionalità dello spazio latente
input_size = 2 # Coppie di coordinate
hidden_size = 128 # Dimensione dello stato nascosto
batch_size = 1 # 64 Dimensione del batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

num_epochs = 1000
val_every = 2000
log_every = 10
lr = 1e-3

mse = nn.MSELoss()
sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)


# # Main

# In[ ]:


import random
import matplotlib.pyplot as plt
import tqdm
import os

os.makedirs("sdes", exist_ok=True)
os.makedirs("losses", exist_ok=True)
os.makedirs("test_loaders", exist_ok=True)

from_sbj = 0 # Starting subject index
n_sbj = 1
training = True # Set to False to skip training and only generate fixations
saving = True # Set to False to skip saving the model

subject_bar = tqdm.tqdm(range(n_sbj), desc="Subjects", leave=True, position=0)
for i in subject_bar:
    if not training:
        break
    if i < from_sbj:
        continue

    #sbj_fixs = fixs[i]

    # Overfitting on the first subject
    sbj_fixs = [fixs[0]]         # soggetto 0
    sbj_fixs[0] = [fixs[0][0]]   # immagine 0
    sbj_fixs[0][0] = [fixs[0][0][0]]  # solo 1 fissazione
    sbj_fixs = sbj_fixs[0]

    sde = LatentSDE(input_size=input_size, hidden_size=hidden_size, latent_size=latent_size, device=device).to(device)
    optimizer = optim.Adam(list(sde.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    early_stopping = EarlyStopping(patience=20)

    train_loader, val_loader, test_loader = create_dataloaders(sbj_fixs, batch_size)

    with open(f"test_loaders/test_loader_{i}.pkl", "wb") as f:
        pickle.dump(test_loader, f)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0

    sde.train()

    epoch_bar = tqdm.tqdm(range(num_epochs), desc="Epochs", leave=False, position=1)
    for epoch in epoch_bar:

        # Training
        epoch_train_loss = 0.0
        train_bar = tqdm.tqdm(train_loader, desc="Batches", leave=False, position=2)
        for batch, mask in train_bar:
            batch = batch.to(device)
            mask = mask.to(device)

            recon_x = sde(batch, mask)  # [B, T, latent_size]

            mse_loss = mse(recon_x[mask.bool()], batch[mask.bool()])  # MSE loss on the masked elements

            sinkhorn_loss = sinkhorn(recon_x[mask.bool()], batch[mask.bool()])

            batch_loss = mse_loss + sinkhorn_loss

            train_bar.set_postfix({"Batch Loss": batch_loss.item()})

            # Backpropagation
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            epoch_train_loss += batch_loss.item()

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        epoch_val_loss = 0.0
        # Validation
        if (epoch + 1) % val_every == 0:
            sde.eval() # Set the model to evaluation mode
            with torch.no_grad():
                val_bar = tqdm.tqdm(val_loader, desc="Batches", leave=False, position=2)
                for batch, mask in val_bar:
                    batch = batch.to(device)
                    mask = mask.to(device)

                    recon_x = sde(batch, mask)  # [B, T, latent_size]

                    #mse_loss = mse(recon_x[mask.bool()], batch[mask.bool()])  # MSE loss on the masked elements
                    
                    sinkhorn_loss = sinkhorn(recon_x[mask.bool()], batch[mask.bool()])

                    batch_loss = sinkhorn_loss

                    val_bar.set_postfix({"Batch Loss": batch_loss.item()})

                    epoch_val_loss += batch_loss.item()

            epoch_val_loss /= len(val_loader)
            val_losses.append(epoch_val_loss)
            sde.train()

            scheduler.step(epoch_val_loss)
            early_stopping(epoch_val_loss)

            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        epoch_bar.set_postfix({"Epoch Train Loss": epoch_train_loss, "Epoch Val Loss": epoch_val_loss, "Learning Rate": scheduler.get_last_lr()})

        if (num_epochs + 1) % log_every == 0:
            tqdm.tqdm.write(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        # Saving the best model
        if saving:
        #if saving and epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss            
            best_epoch = epoch
            subject_bar.set_postfix({"Best Epoch": best_epoch, "Best Val Loss": best_val_loss})
            torch.save({
                'sde': sde.state_dict(),
                'epoch': best_epoch,
                'val_loss': best_val_loss
            }, "sdes/best_sde_" + str(i) + ".pth")

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.savefig('losses/losses_' + str(i) + '.png')
    plt.clf() # Clear


# # Generation

# In[ ]:


sum = 0
count = 0
for sbj in fixs:
    for img in sbj:
        for fix in img:
            sum += len(fix)
            count += 1
avg_fix_len = (int) (sum / count)


# In[ ]:


import matplotlib.pyplot as plt
import torch
import pickle

fig, axes = plt.subplots(2, 4, figsize=(16, 6))  # 2 righe, 4 colonne
axes = axes.flatten()

for i in range(n_sbj):
    # Carica modello
    data = torch.load(f"sdes/best_sde_{i}.pth", map_location=torch.device('cpu'))
    sde = LatentSDE(input_size, hidden_size, latent_size, device)
    sde.load_state_dict(data['sde'])
    sde.eval()

    # Carica test loader
    #test_loader = pickle.load(open(f"test_loaders/test_loader_{i}.pkl", "rb"))
    #batch, mask = next(iter(test_loader))
    batch, mask = next(iter(train_loader))
    batch = batch.to(device)
    mask = mask.to(device)

    # Seleziona il primo esempio (shape [1, T, 2] e [1, T])
    x = batch[0].unsqueeze(0)
    m = mask[0].unsqueeze(0)

    with torch.no_grad():
        pred = sde(x, m)  # Forward completo

    # Plot originale vs predetto
    ax = axes[i]
    ax.plot(x[0, :, 0].cpu(), x[0, :, 1].cpu(), label='Originale', alpha=0.7)
    ax.plot(pred[0][:][0].cpu(), pred[0][:][1].cpu(), label='Predetta', alpha=0.7)
    ax.set_title(f"Subject {i}")
    ax.legend()

plt.tight_layout()
plt.savefig("forward_vs_original.png")
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
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(latent_size)
        self.project = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, latent_size)
        )

    def forward(self, batch, lengths):
        """
        batch: [B, T, 2] - Batches of fixations (Batch size, Sequence length, 2 coordinates)
        lengths: [B] - Lengths of each fixation in the batch
        """
        packed = pack_padded_sequence(batch, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, last_hidden_states = self.gru(packed)
        # packed_out: [B, T, 2 * H] - Output of the GRU for each time step, not used here
        # last_hidden_states: [2, B, H] - Last hidden states for each direction in the batch
        # last_hidden_states is of shape [num_directions, batch_size, hidden_size]
        # We take the last hidden state of the forward direction (index 0) and the last hidden state of the backward direction (index 1)
        # and concatenate them
        last_hidden_states = last_hidden_states.transpose(0, 1).reshape(batch.size(0), -1) # [B, 2 * H]
        latent_states = self.project(last_hidden_states)  # [B, latent_size]
        return self.norm(latent_states)


# # SDE

# In[ ]:


class LatentSDE(torchsde.SDEIto):
    def __init__(self, input_size, hidden_size, latent_size, device):
        super().__init__(noise_type="diagonal")
        self.device = device

        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size, latent_size=latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, input_size)
        )

        self.drift_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, latent_size)
        )

        self.diffusion_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, latent_size),
            nn.Softplus()
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


from torch.utils.data import DataLoader, Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import torch
import random


class FixationDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        self.lengths = [len(seq) for seq in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

class BucketSampler(Sampler):
    def __init__(self, lengths, batch_size, bucket_size):
        self.batch_size = batch_size
        self.buckets = []

        # 1. Raggruppa tutti gli indici e relative lunghezze
        data = list(enumerate(lengths))

        # 2. Ordina per lunghezza
        data.sort(key=lambda x: x[1])

        # 3. Suddividi in bucket di dimensione `bucket_size`
        for i in range(0, len(data), bucket_size):
            bucket = data[i:i + bucket_size]
            indices = [idx for idx, _ in bucket]
            random.shuffle(indices)
            self.buckets.append(indices)

        # Shuffle globale sui batch
        random.shuffle(self.buckets)

    def __iter__(self):
        return iter(self.buckets)

    def __len__(self):
        return len(self.buckets)


def collate_fn(batch):
    lengths = [seq.shape[0] for seq in batch]
    max_len = max(lengths)
    padded = pad_sequence(batch, batch_first=True)
    mask = torch.arange(max_len)[None, :] < torch.tensor(lengths)[:, None]
    return padded, mask.float()


def create_dataloaders(sbj_fixs, batch_size, bucket_size):
    random.shuffle(sbj_fixs)

    train_size = int(0.7 * len(sbj_fixs))
    val_size = int(0.15 * len(sbj_fixs))

    def preprocess(split):
        return [torch.tensor(fix, dtype=torch.float) for img in split for fix in img]

    train_set = preprocess(sbj_fixs[:train_size])
    val_set = preprocess(sbj_fixs[train_size:train_size + val_size])
    test_set = preprocess(sbj_fixs[train_size + val_size:])

    def make_loader(data):
        dataset = FixationDataset(data)
        sampler = BucketSampler(dataset.lengths, batch_size=batch_size, bucket_size=bucket_size)
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)

    return make_loader(train_set), make_loader(val_set), make_loader(test_set)


# # Parameters

# In[7]:


def chamfer_distance(x, y):
    # x, y: [B, N, D] e [B, M, D]
    x_exp = x.unsqueeze(2)  # [B, N, 1, D]
    y_exp = y.unsqueeze(1)  # [B, 1, M, D]
    dist = torch.norm(x_exp - y_exp, dim=-1)  # [B, N, M]
    min_x_to_y = dist.min(dim=2).values  # [B, N]
    min_y_to_x = dist.min(dim=1).values  # [B, M]
    return (min_x_to_y.mean(dim=1) + min_y_to_x.mean(dim=1)).mean()

latent_size = 8 # Dimensionalità dello spazio latente
input_size = 2 # Coppie di coordinate
hidden_size = 64 # Dimensione dello stato nascosto
batch_size = 128 # Dimensione del batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

num_epochs = 300
val_every = 1
log_every = 1000
lr = 1e-3

mse = nn.MSELoss()


# # Main

# In[ ]:


import random
import matplotlib.pyplot as plt
import tqdm
import os

os.makedirs("sdes", exist_ok=True)
os.makedirs("losses", exist_ok=True)
os.makedirs("train_loaders", exist_ok=True)
os.makedirs("val_loaders", exist_ok=True)
os.makedirs("test_loaders", exist_ok=True)
os.makedirs("reconstructions", exist_ok=True)

training = True # Set to False to skip training and only generate fixations
from_sbj = 0 # Starting subject index
n_sbj = 1 # Number of subjects to train on

subject_bar = tqdm.tqdm(range(n_sbj) if training else [], desc="Subjects", leave=True, position=0)
for i in subject_bar:
    if i < from_sbj:
        continue

    sbj_fixs = fixs[i]

    sde = LatentSDE(input_size=input_size, hidden_size=hidden_size, latent_size=latent_size, device=device).to(device)
    optimizer = optim.Adam(list(sde.parameters()), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5) # lr * = 0.5 every 20 epochs without improvements
    early_stopping = EarlyStopping(patience=30, delta=1e-3) # Stop training if validation loss does not improve for 30 epochs

    train_loader, val_loader, test_loader = create_dataloaders(sbj_fixs, batch_size=batch_size, bucket_size=batch_size)

    with open(f"train_loaders/train_loader_{i}.pkl", "wb") as f:
        pickle.dump(train_loader, f)

    with open(f"val_loaders/val_loader_{i}.pkl", "wb") as f:
        pickle.dump(val_loader, f)

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
            #print(f"Processing batch of size {batch.shape}, lengths = {[int(mask[i].sum()) for i in range(mask.shape[0])]}")

            batch = batch.to(device)
            mask = mask.to(device)

            recon_x = sde(batch, mask) # [B, T, latent_size]

            mse_loss = mse(recon_x * mask.unsqueeze(-1), batch * mask.unsqueeze(-1))

            chamfer_loss = chamfer_distance(recon_x * mask.unsqueeze(-1), batch * mask.unsqueeze(-1))

            batch_loss = mse_loss + chamfer_loss

            train_bar.set_postfix({"Batch Loss": batch_loss.item()})

            # Backpropagation
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            epoch_train_loss += batch_loss.item()

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        epoch_val_loss = float('inf')  # Initialize validation loss to infinity
        # Validation
        if (epoch + 1) % val_every == 0:
            epoch_val_loss = 0.0
            sde.eval() # Set the model to evaluation mode
            with torch.no_grad():
                val_bar = tqdm.tqdm(val_loader, desc="Batches", leave=False, position=2)
                for batch, mask in val_bar:
                    batch = batch.to(device)
                    mask = mask.to(device)

                    recon_x = sde(batch, mask)  # [B, T, latent_size]

                    mse_loss = mse(recon_x * mask.unsqueeze(-1), batch * mask.unsqueeze(-1))
                    
                    chamfer_loss = chamfer_distance(recon_x * mask.unsqueeze(-1), batch * mask.unsqueeze(-1))

                    batch_loss = mse_loss + chamfer_loss

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
        if epoch_val_loss < best_val_loss:
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


import matplotlib.pyplot as plt
import torch
import pickle

for i in range(n_sbj):
    # Carica modello
    data = torch.load(f"sdes/best_sde_{i}.pth", map_location=torch.device('cpu'), weights_only=True)
    sde = LatentSDE(input_size, hidden_size, latent_size, device)
    sde.load_state_dict(data['sde'])
    sde.to(device)
    sde.eval()
    print(f"Loaded SDE for Subject {i} from epoch {data['epoch']} with validation loss {data['val_loss']:.4f}")

    # Carica test loader
    test_loader = pickle.load(open(f"test_loaders/test_loader_{i}.pkl", "rb"))
    test_loss = 0.0
    with torch.no_grad():
        for batch, mask in test_loader:
            batch = batch.to(device)
            mask = mask.to(device)

            recon_x = sde(batch, mask)  # [B, T, latent_size]

            mse_loss = mse(recon_x * mask.unsqueeze(-1), batch * mask.unsqueeze(-1))  # MSE loss on the masked elements
            
            chamfer_loss = chamfer_distance(recon_x * mask.unsqueeze(-1), batch * mask.unsqueeze(-1))

            batch_loss = mse_loss + chamfer_loss

            test_loss += batch_loss.item()

            for j in range(batch.size(0)):
                plt.plot(batch[j, mask[j].bool(), 0].cpu().detach().numpy(), batch[j, mask[j].bool(), 1].cpu().detach().numpy(), color='blue', label='Original')
                plt.plot(recon_x[j, mask[j].bool(), 0].cpu().detach().numpy(), recon_x[j, mask[j].bool(), 1].cpu().detach().numpy(), color='red', label='Reconstructed')
                plt.title(f"Subject {i} - Image {j}")
                plt.xlabel("X Coordinate")
                plt.ylabel("Y Coordinate")
                plt.legend()
                plt.savefig(f"reconstructions/reconstruction_sbj_{i}_img_{j}.png")
                plt.clf()

        test_loss /= len(test_loader)
    
    print(f"Test Loss for Subject {i}: {test_loss:.4f}")
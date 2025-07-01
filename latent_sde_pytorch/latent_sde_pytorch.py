import pickle

with open("../dataset/fixs.pkl", "rb") as f:
    fixs = pickle.load(f) # n_sbj x n_img x n_fix x n_coord x 2

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import tqdm
from torch import nn
from torch import optim
import torchsde

class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val

# ## Encoder
# Prende in input la sequenza osservata xs e ne produce un contesto ctx, che verrà usato per condizionare i termini drift/diffusione della SDE.
# - GRU: aggrega l’informazione nel tempo.
# - Linear: riduce la dimensionalità a context_size.
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp, lengths):
        # inp: [T, B, input_size]
        # lengths: [B] → lunghezze reali (senza padding)

        # Ordina per lunghezza decrescente (richiesto da pack_padded_sequence)
        lengths_sorted, perm_idx = lengths.sort(0, descending=True)
        inp_sorted = inp[:, perm_idx]

        # Impacchetta la sequenza
        packed = pack_padded_sequence(inp_sorted, lengths_sorted.cpu(), enforce_sorted=True)

        # Passa alla GRU
        packed_out, _ = self.gru(packed)

        # Deimpacchetta
        out, _ = pad_packed_sequence(packed_out)

        # Ripristina l'ordine originale del batch
        _, unperm_idx = perm_idx.sort(0)
        out = out[:, unperm_idx]

        # Passa alla linear
        out = self.lin(out)

        return out

# ## LatentSDE
# Implementa un modello generativo basato su SDE in uno spazio latente.
from torch.distributions import Normal

class LatentSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, data_size, latent_size, context_size, hidden_size):
        super(LatentSDE, self).__init__()

        # Encoder: calcola un contesto temporale ctx(t) dalla sequenza osservata xs
        self.encoder = Encoder(input_size=data_size, hidden_size=hidden_size, output_size=context_size)

        # Posterior su z0: approssima la distribuzione posteriore su z0 (il punto iniziale nello spazio latente)
        # Produce la media e log-varianza
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)

        # Decoder
        # Drift condizionato sul contesto -> usato nel forward path
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        # Drift del modello generativo -> usato per generazione/sampling
        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )

        # This needs to be an element-wise function for the SDE to satisfy diagonal noise
        # Diffusione diagonale -> ogni dimensione ha la sua rete (richiesto per noise diagonale)
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Softplus(),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid()
                )
                for _ in range(latent_size)
            ]
        )

        # Proietta il processo latente nello spazio osservabile per confrontarlo con i dati osservati
        self.projector = nn.Linear(latent_size, data_size)

        # Prior su z0: distribuzione prior su z0 (punto iniziale nello spazio latente)
        # Inizializza la distribuzione prior su z0 come una normale standard (media zero e varianza uno)
        # Parametri di un prior normale standard su z0 (trainabili)
        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self._ctx = None

    # Memorizza il contesto temporale calcolato dall'encoder (una coppia (ts, ctx) per l'accesso temporale nel drift),
    # usato per calcolare la drift condizionata sul contesto
    # e per il calcolo della distribuzione posteriore su z0.
    # Il contesto è una sequenza di vettori di dimensione [T, batch_size, context_size], T lunghezza della sequenza.
    # Viene calcolato dall'encoder e passato come input al modello SDE.
    def contextualize(self, ctx):
        self._ctx = ctx

    # Cerca il contesto ctx(t) interpolando (step-wise) e concatena a y per usare f_net.
    def f(self, t, y):
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[i]), dim=1))

    def h(self, t, y):
        return self.h_net(y)

    # Diffusione diagonale
    def g(self, t, y):
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    # Training
    def forward(self, xs, mask, ts, noise_std, adjoint=False, method="euler"):
        # Context encoding (contextualization is only needed for posterior inference)
        ctx = self.encoder(torch.flip(xs, dims=(0,)), torch.flip(mask, dims=(0,)).sum(dim=0).long()) # params: xs, lengths (sequences lengths without padding)
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))

        # Compute the posterior distribution q(z0|x0), conditioned on the context ctx(t), and sample z0
        # q(z0|x0) = N(qz0_mean, qz0_logstd.exp())
        # where qz0_mean and qz0_logstd are computed by the qz0_net.
        # The posterior is used to
        #   - sample the initial point z0 in the latent space.
        #   - compute the KL divergence term in the loss function.
        #   - compute the log probability of the path in the latent space.
        #   - compute the log probability of the initial point z0 in the latent space.
        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)

        # Simulate the SDE path in the latent space
        if adjoint:
            # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
            adjoint_params = (
                    (ctx,) +
                    tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
            )
            # Integrate the SDE using the adjoint method
            zs, log_ratio = torchsde.sdeint_adjoint(
                self, z0, ts, adjoint_params=adjoint_params, dt=1e-2, logqp=True, method=method)
        else:
            zs, log_ratio = torchsde.sdeint(self, z0, ts, dt=1e-2, logqp=True, method=method)
        # zs is a tensor of shape [T, batch_size, latent_dim], where T is the number of time steps.
        # log_ratio is a tensor of shape [T, batch_size], containing the log probability ratio of the path. 

        # Project the latent path zs into the observation space xs
        _xs = self.projector(zs)

        # Distribuzione normale con media _xs e deviazione standard noise_std.
        xs_dist = Normal(loc=_xs, scale=noise_std)

        # Confronta le osservazioni xs con le osservazioni ricostruite _xs
        # xs_dist.log_prob(xs) calcola il logaritmo della probabilità delle osservazioni xs date le osservazioni ricostruite _xs.
        log_prob = xs_dist.log_prob(xs)  # [T, B, D]
        mask = mask.unsqueeze(-1) # mask: [T, B] → [T, B, 1]
        log_prob = log_prob * mask  # [T, B, D]
        # La somma lungo le dimensioni (0, 2) calcola il logaritmo della probabilità per ogni osservazione xs.
        # La media lungo la dimensione 0 calcola la probabilità media per batch.
        log_pxs = log_prob.sum(dim=(0, 2)).mean(dim=0)

        # Calcola la divergenza KL tra la distribuzione posteriore q(z0|x0) e la distribuzione prior p(z0)
        qz0 = Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())

        # logqp0 is the log probability of the initial point z0 in the latent space, which is computed by the posterior.
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)

        # logqp_path is the log probability of the path in the latent space, which is computed by the SDE solver.
        logqp_path = log_ratio.sum(dim=0).mean(dim=0)

        return log_pxs, logqp0 + logqp_path

    # Generazione
    @torch.no_grad()
    def sample(self, batch_size, ts, bm=None):
        eps = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
        # Sample z0 from the prior p(z0)
        z0 = self.pz0_mean + self.pz0_logstd.exp() * eps

        # Context is not needed for sampling, so we can ignore it.
        # Simulate the SDE path in the latent space
        zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h'}, dt=1e-3, bm=bm)
        # Most of the times in ML, we don't sample the observation noise for visualization purposes.
        # Project the latent path zs into the observation space xs
        _xs = self.projector(zs)
        return _xs

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
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    lengths = torch.tensor([seq.shape[0] for seq in batch])
    # Padding the sequences to the maximum length in the batch
    padded = pad_sequence(batch, batch_first=False) # [T, B, 2]
    mask = torch.arange(padded.shape[0]).unsqueeze(1) < lengths.unsqueeze(0)
    mask = mask.float()  # Convert to float for compatibility with loss functions
    return padded, mask

import random

def create_dataloaders(sbj_fixs, batch_size):
    # Mescola le traiettorie
    random.shuffle(sbj_fixs)

    # Suddivisione 70% training, 15% validation, 15% test
    train_size = int(0.7 * len(sbj_fixs))
    val_size = int(0.15 * len(sbj_fixs))
    train_set = [torch.tensor(fix, dtype=torch.float) for img in sbj_fixs[:train_size] for fix in img]
    val_set = [torch.tensor(fix, dtype=torch.float) for img in sbj_fixs[train_size:train_size + val_size] for fix in img]
    test_set = [torch.tensor(fix, dtype=torch.float) for img in sbj_fixs[train_size + val_size:] for fix in img]

    # Crea i DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Parameters
    batch_size=32
    data_size=2
    latent_size=16
    context_size=64
    hidden_size=64
    lr_init=1e-3
    lr_gamma=1.0 # 0.997
    num_iters=200
    kl_anneal_iters=1000
    log_every=10
    noise_std=0.01
    adjoint=False
    method="euler"

    for i in range(1):
        print(f"Training on subject {i}")
        sbj_fixs = fixs[i]

        train_loader, val_loader, test_loader = create_dataloaders(sbj_fixs, batch_size)
        with open(f'test_loader_{i}.pkl', 'wb') as f:
            pickle.dump(test_loader, f)

        train_losses = []
        val_losses = []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        latent_sde = LatentSDE(
            data_size=data_size,
            latent_size=latent_size,
            context_size=context_size,
            hidden_size=hidden_size,
        ).to(device)
        optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
        early_stopping = EarlyStopping(patience=30, delta=1e-3) # Stop training if validation loss does not improve for "patience" epochs
        kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

        best_val_loss = float('inf')
        best_val_loss_epoch = 0
        latent_sde.train()  # Set the model to training mode

        for global_step in tqdm.tqdm(range(1, num_iters + 1)):
            # Training step
            epoch_train_loss = 0.0
            for batch, mask in train_loader:
                batch = batch.to(device)
                mask = mask.to(device)
                ts = torch.linspace(0, 1, batch.size(0), device=device)

                optimizer.zero_grad()
                log_pxs, log_ratio = latent_sde(batch, mask, ts, noise_std, adjoint, method)
                # train_loss = -log_pxs + log_ratio * kl_scheduler.val
                train_loss = -log_pxs + log_ratio
                epoch_train_loss += train_loss
                train_loss.backward()
                optimizer.step()
                scheduler.step()
                #kl_scheduler.step()
            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)

            # Validation step
            with torch.no_grad():
                epoch_val_loss = 0.0
                latent_sde.eval() # Set the model to evaluation mode
                for batch, mask in val_loader:
                    batch = batch.to(device)
                    mask = mask.to(device)
                    ts = torch.linspace(0, 1, batch.size(0), device=device)

                    log_pxs, log_ratio = latent_sde(batch, mask, ts, noise_std, adjoint, method)
                    # val_loss = -log_pxs + log_ratio * kl_scheduler.val
                    val_loss = -log_pxs + log_ratio
                    epoch_val_loss += val_loss
                epoch_val_loss /= len(val_loader)
                val_losses.append(epoch_val_loss)
                early_stopping(epoch_val_loss.item())
                latent_sde.train() # Set the model back to training mode

            # Logging
            if global_step % log_every == 0:
                tqdm.tqdm.write(
                    f"Sbj {i} | "
                    f"[{global_step:03d}] loss: {epoch_train_loss} | "
                    f"val_loss: {epoch_val_loss} | "
                )

            if (epoch_val_loss < best_val_loss):
                best_val_loss = epoch_val_loss
                best_val_loss_epoch = global_step
                torch.save({
                    'model_state_dict': latent_sde.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': best_val_loss_epoch,
                    'val_loss': best_val_loss,
                }, 'best_latent_sde_' + str(i) + '.pth')

        # Plotting training and validation losses
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='orange')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses Sbj ' + str(i))
        plt.yscale('log') # Log scale for better visibility
        plt.axhline(y=best_val_loss, color='red', linestyle='--', label='Best Validation Loss')
        plt.axvline(x=best_val_loss_epoch, color='green', linestyle='--', label='Best Validation Iteration')
        plt.xticks(range(0, num_iters + 1, log_every))
        plt.xlim(0, num_iters)
        plt.legend()
        plt.grid()
        plt.savefig('losses_sbj_' + str(i) + '.png')

    fig, axes = plt.subplots(2, 4, figsize=(16, 6))  # 2 righe, 4 colonne
    axes = axes.flatten()  # Rende l'array bidimensionale in un array 1D per iterare facilmente

    for i in range(8):
        print(f"Testing on subject {i}")
        checkpoint = torch.load('best_latent_sde_' + str(i) + '.pth')
        print(f"\tEpoch {checkpoint['epoch']} | Validation loss {checkpoint['val_loss']}")
        epoch = checkpoint['epoch']
        val_loss = checkpoint['val_loss']
        latent_sde = LatentSDE(
            data_size=data_size,
            latent_size=latent_size,
            context_size=context_size,
            hidden_size=hidden_size,
        ).to(device)
        latent_sde.load_state_dict(checkpoint['model_state_dict'])
        latent_sde.eval()  # Set the model to evaluation mode
        test_loader = pickle.load(open(f"test_loaders/test_loader_{i}.pkl", "rb"))

        # Test step
        with torch.no_grad():
            for batch, mask in test_loader:
                batch = batch.to(device)
                mask = mask.to(device)
                ts = torch.linspace(0, 1, batch.size(0), device=device)

                _xs = latent_sde.sample(batch.size(0), ts)

                x, y = _xs[:, 0], _xs[:, 1]
                ax = axes[i]
                ax.plot(x, y)
                ax.set_title("Subject " + str(i))
                ax.set_xlabel("X")
                ax.set_ylabel("Y")

    plt.tight_layout()
    plt.savefig('sampled_trajectories.png')
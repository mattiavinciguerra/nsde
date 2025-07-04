from latent_sde_pytorch import LatentSDE
from latent_sde_pytorch import collate_fn
import torch
import pickle
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
fig, axes = plt.subplots(2, 4, figsize=(16, 6))  # 2 righe, 4 colonne
axes = axes.flatten()  # Rende l'array bidimensionale in un array 1D per iterare facilmente
for i in range(1):
    print(f"Testing on subject {i}")
    checkpoint = torch.load('best_latent_sde_' + str(i) + '.pth')
    print(f"\tEpoch {checkpoint['epoch']} | Validation loss {checkpoint['val_loss']}")
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    latent_sde = LatentSDE(
        data_size=2,
        latent_size=16,
        context_size=64,
        hidden_size=64,
    ).to(device)
    latent_sde.load_state_dict(checkpoint['model_state_dict'])
    latent_sde.eval()  # Set the model to evaluation mode
    test_loader = pickle.load(open(f"test_loader_{i}.pkl", "rb"))

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
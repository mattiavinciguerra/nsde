from my_latent_sde import LatentSDE

# Testing
import matplotlib.pyplot as plt
import torch
import pickle

# parameters
input_size = 2  # Assuming 2D coordinates (x, y)
hidden_size = 64  # Size of the hidden layers in the SDE
latent_size = 16  # Size of the latent space
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_sbj = 1  # Number of subjects
mse = torch.nn.MSELoss()

for i in range(n_sbj):
    data = torch.load(f"sdes/best_sde_{i}.pth", map_location=torch.device('cpu'), weights_only=True)
    sde = LatentSDE(input_size, hidden_size, latent_size, device)
    sde.load_state_dict(data['sde'])
    sde.to(device)
    sde.eval()
    print(f"Loaded SDE for Subject {i} from epoch {data['epoch']} with validation loss {data['val_loss']:.4f}")

    test_loader = pickle.load(open(f"test_loaders/test_loader_{i}.pkl", "rb"))
    latent_states = []
    with torch.no_grad():
        for batch, mask in test_loader:
            batch = batch.to(device)
            mask = mask.to(device)

            latent_state = sde.encode(batch, mask)
            latent_states.append(latent_state.cpu())

    # PCA and plotting
    latent_states = torch.cat(latent_states, dim=0).numpy()
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced_states = pca.fit_transform(latent_states)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_states[:, 0], reduced_states[:, 1], alpha=0.5)
    plt.title(f'Latent Space Representation for Subject {i}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid()
    plt.savefig(f"latent_space_subject_{i}.png")
    plt.close()
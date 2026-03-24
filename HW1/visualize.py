import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from parameters import get_params
from models.MLP import MLP


def extract_features(model, loader, device):
    model.eval()
    features, labels_list = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            x = imgs.view(imgs.size(0), -1)
            # Get output of second to last layer
            x = model.net[:-1](x)
            features.append(x.cpu().numpy())
            labels_list.append(labels.numpy())
    return np.concatenate(features), np.concatenate(labels_list)


def run_tsne(params, device):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(params["mean"], params["std"]),
    ])

    dataset = datasets.MNIST(params["data_dir"], train=False, download=True, transform=tf)
    loader  = DataLoader(dataset, batch_size=512, shuffle=False)

    model = MLP(
        input_size   = params["input_size"],
        hidden_sizes = params["hidden_sizes"],
        num_classes  = params["num_classes"],
        dropout      = params["dropout"],
        activation   = params["activation"],
    ).to(device)

    model.load_state_dict(torch.load(params["save_path"], map_location=device))

    features, labels = extract_features(model, loader, device)

    print("Running t-SNE, this may take a moment...")
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(features[:2000])

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedded[:, 0], embedded[:, 1],
                          c=labels[:2000], cmap="tab10", s=5)
    plt.colorbar(scatter)
    plt.title("t-SNE of MLP features")
    plt.savefig(f"plots/tsne_{params['run_name']}.png")
    plt.show()
    print("t-SNE plot saved to tsne.png")


if __name__ == "__main__":
    params = get_params()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_tsne(params, device)
import argparse
import sys
import warnings
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils.attribute_hashmap import AttributeHashmap
from utils.metrics import per_class_dice_coeff
from utils.segmentation import label_hint_seg

warnings.filterwarnings("ignore")

class GCNClustering(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(GCNClustering, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def generate_gcn(shape: Tuple[int],
                 latent: np.array,
                 label_true: np.array,
                 num_workers: int = 1,
                 random_seed: int = 1) -> Tuple[float, np.array, np.array]:

    H, W, C = shape
    assert latent.shape == (H * W, C)

    seg_true = label_true > 0

    # Prepare graph data for GCN
    edge_index = build_graph_edges(latent)
    x = torch.tensor(latent, dtype=torch.float)

    # Initialize and train GCN
    model = GCNClustering(in_channels=C, out_channels=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the GCN model
    for epoch in range(100):  # Number of epochs can be adjusted
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out, torch.tensor(label_true.flatten(), dtype=torch.long))
        loss.backward()
        optimizer.step()

    # Get cluster predictions (taking the max of each node's output)
    _, cluster_pred = out.max(dim=1)
    cluster_pred = cluster_pred.detach().cpu().numpy().reshape(H, W)

    seg_pred = label_hint_seg(label_pred=cluster_pred, label_true=label_true)

    return per_class_dice_coeff(seg_pred, seg_true), cluster_pred, seg_pred

def build_graph_edges(latent: np.array) -> torch.Tensor:
    """
    Build a k-nearest neighbor graph from the latent features
    to use as the edge connections in the graph for GCN.
    """
    from sklearn.neighbors import NearestNeighbors

    # Build the K-nearest neighbors graph
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(latent)
    distances, indices = nbrs.kneighbors(latent)

    # Convert to edge_index format for PyTorch Geometric
    edge_index = []
    for i in range(latent.shape[0]):
        for j in indices[i]:
            if i != j:  # Avoid self-loops
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return edge_index

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--num_workers', type=int, required=True)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    numpy_array = np.load(args.load_path)
    image = numpy_array['image']
    label_true = numpy_array['label']
    latent = numpy_array['latent']

    image = (image + 1) / 2

    H, W = label_true.shape[:2]
    C = latent.shape[-1]
    X = latent

    dice_score, label_pred, seg_pred = generate_gcn(
        (H, W, C), latent, label_true, num_workers=args.num_workers)

    with open(args.save_path, 'wb+') as f:
        np.savez(f,
                 image=image,
                 label=label_true,
                 latent=latent,
                 label_gcn=label_pred,
                 seg_gcn=seg_pred)

    sys.stdout.write('SUCCESS! %s, dice: %s' %
                     (args.load_path.split('/')[-1], dice_score))

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
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries

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

    # Add geometric features to latent space
    boundary_features = compute_geometric_features(label_true)
    enhanced_latent = np.concatenate([latent, boundary_features], axis=1)

    # Prepare graph data for GCN
    edge_index = build_graph_edges(enhanced_latent)
    x = torch.tensor(enhanced_latent, dtype=torch.float)

    # Initialize and train GCN
    model = GCNClustering(in_channels=enhanced_latent.shape[1], out_channels=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the GCN model with geometric constraints
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(x, edge_index)

        # Classification loss
        cls_loss = F.cross_entropy(out, torch.tensor(label_true.flatten(), dtype=torch.long))

        # Boundary smoothness loss
        boundary_loss = compute_boundary_loss(out, label_true, H, W)

        # Total loss with geometric constraints
        loss = 0.5 * cls_loss + 0.5 * boundary_loss  # Weight can be adjusted

        loss.backward()
        optimizer.step()

    # Get cluster predictions
    _, cluster_pred = out.max(dim=1)
    cluster_pred = cluster_pred.detach().cpu().numpy().reshape(H, W)

    seg_pred = label_hint_seg(label_pred=cluster_pred, label_true=label_true)

    return per_class_dice_coeff(seg_pred, seg_true), cluster_pred, seg_pred


def compute_geometric_features(label_true: np.array) -> np.array:
    """
    Compute geometric features (distance transform and boundary) to enhance latent space.
    """
    H, W = label_true.shape
    features = np.zeros((H * W, 2))  # Distance transform + boundary indicator

    # Distance transform for each class
    for class_id in np.unique(label_true):
        if class_id == 0:  # Skip background
            continue
        binary_mask = (label_true == class_id).astype(np.float32)
        pos_dist = distance_transform_edt(binary_mask)
        neg_dist = distance_transform_edt(1 - binary_mask)

        # Stack features
        class_features = np.stack([pos_dist, neg_dist], axis=-1)
        features[label_true.flatten() == class_id] = class_features[label_true == class_id]

    # Boundary indicator
    boundaries = find_boundaries(label_true, mode='inner').astype(np.float32)
    features = np.concatenate([features, boundaries.reshape(-1, 1)], axis=1)

    return features


def compute_boundary_loss(out: torch.Tensor, label_true: np.array, H: int, W: int) -> torch.Tensor:
    """
    Compute boundary smoothness loss to encourage geometrically consistent segmentation.
    """
    # Get predicted probabilities
    prob = F.softmax(out, dim=1)

    # Reshape to 2D
    prob_2d = prob.view(H, W, -1).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

    # Compute gradient magnitude
    grad_x = torch.abs(prob_2d[:, :, :, 1:] - prob_2d[:, :, :, :-1])
    grad_y = torch.abs(prob_2d[:, :, 1:, :] - prob_2d[:, :, :-1, :])

    # Get boundary mask from ground truth
    boundary_mask = torch.tensor(find_boundaries(label_true, mode='inner'),
                                 dtype=torch.float32, device=out.device)
    boundary_mask = boundary_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # Apply boundary mask to gradients
    grad_x = grad_x * boundary_mask[:, :, :, :-1]
    grad_y = grad_y * boundary_mask[:, :, :-1, :]

    # Sum gradients along boundaries
    boundary_loss = grad_x.mean() + grad_y.mean()

    return boundary_loss


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
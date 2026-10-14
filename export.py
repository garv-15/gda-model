"""
Model Export Pipeline for TransGDA:
Trains or bundles a production-ready TransGDA checkpoint along with:
1. PyTorch model state dict & calibrated temperature
2. Canonical entity mappings (genes and diseases)
3. Functional graph neighborhoods
4. Model architecture configuration
"""

import argparse
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from src.utils import set_global_seed
from src.data import DataLoaderManager
from src.similarity import GraphNeighborhoodManager
from src.models.transf_gda import TransGDA
from src.train import TransGDATrainer, get_optimal_device


def export_model(
    data_dir: str = ".",
    output_dir: str = "results/saved_model",
    epochs: int = 10,
    embed_dim: int = 64,
    num_heads: int = 4,
    ffn_dim: int = 128,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    k_neighbors: int = 16,
    lr: float = 1e-3,
    lambda_strength: float = 1.0,
    seed: int = 42,
) -> Path:
    set_global_seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = get_optimal_device()
    print(f"Device: {device}")

    # 1. Load and canonically align data
    print("Loading and aligning canonical dataset...")
    data_mgr = DataLoaderManager(data_dir=data_dir)
    ext_gene_sim, ext_disease_sim = data_mgr.load_external_similarity_matrices()

    split = data_mgr.create_edge_split(val_ratio=0.05, test_ratio=0.01, seed=seed)
    print(f"Train edges: {len(split.train_edges):,} | Val edges: {len(split.val_edges):,}")

    # 2. Build leakage-free similarity graphs
    print("Computing similarity graphs...")
    graph_mgr = GraphNeighborhoodManager(
        num_genes=data_mgr.meta.num_genes,
        num_diseases=data_mgr.meta.num_diseases,
        k_gene_neighbors=k_neighbors,
        k_disease_neighbors=k_neighbors,
    )
    graph_mgr.build_from_train_matrix(split.train_matrix_binary, metric="cosine")

    # 3. Build model
    print("Initializing TransGDA...")
    model = TransGDA(
        num_genes=data_mgr.meta.num_genes,
        num_diseases=data_mgr.meta.num_diseases,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        gene_feat_dim=data_mgr.meta.num_genes,
        disease_feat_dim=data_mgr.meta.num_diseases,
    )
    model.set_node_features(
        gene_feat=torch.tensor(ext_gene_sim, dtype=torch.float32),
        disease_feat=torch.tensor(ext_disease_sim, dtype=torch.float32),
    )

    train_loader, val_loader, _ = data_mgr.build_torch_dataloaders(
        split, batch_size=256, shuffle_train=True
    )

    trainer = TransGDATrainer(
        model=model,
        graph_manager=graph_mgr,
        device=device,
        lr=lr,
        lambda_strength=lambda_strength,
    )

    print(f"Training model for {epochs} epochs...")
    trainer.fit(train_loader, val_loader, epochs=epochs, patience=4, verbose=True)

    # 4. Save model weights and calibrated temperature
    weights_path = out_path / "transgda_weights.pt"
    trainer.save_checkpoint(weights_path)

    # 5. Save architecture configuration
    config = {
        "num_genes": data_mgr.meta.num_genes,
        "num_diseases": data_mgr.meta.num_diseases,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "ffn_dim": ffn_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "k_neighbors": k_neighbors,
        "gene_feat_dim": data_mgr.meta.num_genes,
        "disease_feat_dim": data_mgr.meta.num_diseases,
        "calibrated_temperature": float(trainer.model.temperature.item()),
    }
    config_path = out_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # 6. Save metadata mappings
    meta_dict = {
        "genes": data_mgr.meta.genes,
        "diseases": data_mgr.meta.diseases,
        "gene_to_idx": data_mgr.meta.gene_to_idx,
        "disease_to_idx": data_mgr.meta.disease_to_idx,
    }
    meta_path = out_path / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta_dict, f, indent=2)

    # 7. Save graph neighborhoods
    nbr_path = out_path / "graph_neighborhoods.pt"
    torch.save({
        "gene_neighbor_idx": graph_mgr.gene_neighbor_idx.cpu(),
        "gene_neighbor_wt": graph_mgr.gene_neighbor_wt.cpu(),
        "disease_neighbor_idx": graph_mgr.disease_neighbor_idx.cpu(),
        "disease_neighbor_wt": graph_mgr.disease_neighbor_wt.cpu(),
    }, nbr_path)

    print(f"\nModel export completed successfully:")
    print(f"  - Weights:       {weights_path}")
    print(f"  - Configuration: {config_path}")
    print(f"  - Metadata:      {meta_path}")
    print(f"  - Neighborhoods: {nbr_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export trained TransGDA bundle.")
    parser.add_argument("--data_dir", type=str, default=".", help="Root dataset directory")
    parser.add_argument("--output_dir", type=str, default="results/saved_model", help="Export target directory")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    export_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
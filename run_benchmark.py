"""
Benchmark Runner: Evaluates Baselines and TransGDA across Edge, Gene-Held-Out,
and Disease-Held-Out CV splits under strict leakage-free conditions.
"""

import argparse
from pathlib import Path
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.data import DataLoaderManager
from src.similarity import GraphNeighborhoodManager, compute_cosine_similarity
from src.utils import set_global_seed
from src.models.transf_gda import TransGDA
from src.models.baselines import (
    DegreeBaseline,
    NeighborhoodCFBaseline,
    MatrixFactorizationBaseline,
    StaticMLPBaseline,
    TransGDAAblationNoCrossAttn,
)
from src.train import TransGDATrainer, get_optimal_device
from src.evaluate import evaluate_predictions, format_metrics_table


def _fit_or_load(
    trainer: TransGDATrainer,
    train_loader,
    val_loader,
    epochs: int,
    patience: int,
    verbose: bool,
    ckpt_path: Path,
    load_checkpoint: bool,
) -> None:
    """
    If *load_checkpoint* is True and *ckpt_path* exists, restores the model
    from the checkpoint (skipping all training epochs).  Otherwise trains
    normally and saves the checkpoint afterwards.
    """
    if load_checkpoint and ckpt_path.exists():
        print(f"  ↩  Loading checkpoint (skipping training): {ckpt_path}")
        trainer.load_checkpoint(ckpt_path)
    else:
        trainer.fit(train_loader, val_loader, epochs=epochs, patience=patience, verbose=verbose)
        trainer.save_checkpoint(ckpt_path)


def run_benchmark_for_split(
    split_name: str,
    data_mgr: DataLoaderManager,
    device: torch.device,
    epochs: int = 15,
    fast: bool = False,
    checkpoint_dir: Path = Path("results/checkpoints"),
    load_checkpoint: bool = False,
) -> dict:
    print(f"\n{'='*70}\nStarting Benchmark on [{split_name.upper()}] Split (Leakage-Free)\n{'='*70}")

    # Generate split
    if split_name == "edge":
        split = data_mgr.create_edge_split(val_ratio=0.10, test_ratio=0.10, seed=42)
    elif split_name == "gene":
        split = data_mgr.create_gene_held_out_split(val_ratio=0.10, test_ratio=0.15, seed=42)
    elif split_name == "disease":
        split = data_mgr.create_disease_held_out_split(val_ratio=0.10, test_ratio=0.15, seed=42)
    else:
        raise ValueError(f"Unknown split: {split_name}")

    print(f"Train edges: {len(split.train_edges):,}, Val edges: {len(split.val_edges):,}, Test edges: {len(split.test_edges):,}")

    # Load external similarity matrices for inductive cold-start features
    ext_gene_sim, ext_disease_sim = data_mgr.load_external_similarity_matrices()
    cold_genes = split.test_gene_indices if split_name == "gene" else None
    cold_diseases = split.test_disease_indices if split_name == "disease" else None

    # Build leakage-free similarity graphs
    print("Computing dynamic leakage-free similarity networks from A_train...")
    t0 = time.time()
    graph_mgr = GraphNeighborhoodManager(
        num_genes=data_mgr.meta.num_genes,
        num_diseases=data_mgr.meta.num_diseases,
        k_gene_neighbors=16,
        k_disease_neighbors=16,
    )
    graph_mgr.build_from_train_matrix(
        split.train_matrix_binary,
        metric="cosine",
        external_gene_sim=ext_gene_sim,
        external_disease_sim=ext_disease_sim,
        cold_start_genes=cold_genes,
        cold_start_diseases=cold_diseases,
    )
    print(f"Similarity computed in {time.time()-t0:.2f}s!")

    # Test set pairs
    test_pairs = np.vstack([split.test_edges, split.test_neg])
    y_test = np.concatenate([
        np.ones(len(split.test_edges), dtype=np.float32),
        np.zeros(len(split.test_neg), dtype=np.float32),
    ])
    s_test = np.concatenate([
        data_mgr.strength_matrix[split.test_edges[:, 0], split.test_edges[:, 1]],
        np.zeros(len(split.test_neg), dtype=np.float32),
    ])
    g_test, d_test = test_pairs[:, 0], test_pairs[:, 1]

    results = {}

    # 1. Degree Baseline
    print("\n--- Evaluating Baseline 1: Degree / Frequency Prior ---")
    deg_model = DegreeBaseline()
    deg_model.fit(split.train_matrix_binary, split.train_matrix_strength)
    p_deg, s_deg = deg_model.predict(g_test, d_test)
    results["Degree Prior"] = evaluate_predictions(y_test, p_deg, s_test, s_deg)

    # 2. Neighborhood Collaborative Filtering Baseline
    print("\n--- Evaluating Baseline 2: Neighborhood CF (Guilt-By-Association) ---")
    cf_model = NeighborhoodCFBaseline(k_neighbors=20)
    cf_model.fit(split.train_matrix_binary, split.train_matrix_strength, graph_mgr.gene_sim)
    p_cf, s_cf = cf_model.predict(g_test, d_test)
    results["Neighborhood CF"] = evaluate_predictions(y_test, p_cf, s_test, s_cf)

    # 3. Low-Rank Matrix Factorization (SVD)
    print("\n--- Evaluating Baseline 3: Matrix Factorization (SVD) ---")
    mf_model = MatrixFactorizationBaseline(n_components=32, seed=42)
    mf_model.fit(split.train_matrix_binary, split.train_matrix_strength)
    p_mf, s_mf = mf_model.predict(g_test, d_test)
    results["Matrix Factorization"] = evaluate_predictions(y_test, p_mf, s_test, s_mf)

    # DataLoaders for PyTorch models
    batch_size = 512 if fast else 256
    train_loader, val_loader, test_loader = data_mgr.build_torch_dataloaders(
        split, batch_size=batch_size, shuffle_train=True
    )

    # 4. Static MLP Baseline
    print("\n--- Evaluating Baseline 4: Static Embedding MLP (No Attention) ---")
    mlp_model = StaticMLPBaseline(
        num_genes=data_mgr.meta.num_genes,
        num_diseases=data_mgr.meta.num_diseases,
        embed_dim=64,
        hidden_dim=128,
        dropout=0.1,
    )
    mlp_trainer = TransGDATrainer(
        model=mlp_model,
        graph_manager=None,
        device=device,
        lr=1e-3,
        lambda_strength=1.0,
    )
    _fit_or_load(
        mlp_trainer, train_loader, val_loader,
        epochs=epochs, patience=4, verbose=False,
        ckpt_path=checkpoint_dir / f"{split_name}_mlp.pt",
        load_checkpoint=load_checkpoint,
    )
    m_mlp, _, _ = mlp_trainer.evaluate(test_loader)
    results["Static MLP"] = m_mlp

    # 5. Ablation: TransGDA without Cross-Attention
    print("\n--- Evaluating Ablation: TransGDA (w/o Cross-Attention) ---")
    abl_model = TransGDAAblationNoCrossAttn(
        num_genes=data_mgr.meta.num_genes,
        num_diseases=data_mgr.meta.num_diseases,
        embed_dim=64,
        num_heads=4,
        ffn_dim=128,
        hidden_dim=128,
        dropout=0.1,
        gene_feat_dim=data_mgr.meta.num_genes,
        disease_feat_dim=data_mgr.meta.num_diseases,
    )
    abl_model.set_node_features(
        gene_feat=torch.tensor(ext_gene_sim, dtype=torch.float32),
        disease_feat=torch.tensor(ext_disease_sim, dtype=torch.float32),
    )
    if cold_genes is not None:
        abl_model.set_cold_start_nodes(cold_start_genes=cold_genes)
    if cold_diseases is not None:
        abl_model.set_cold_start_nodes(cold_start_diseases=cold_diseases)

    abl_trainer = TransGDATrainer(
        model=abl_model,
        graph_manager=graph_mgr,
        device=device,
        lr=1e-3,
        lambda_strength=1.0,
    )
    _fit_or_load(
        abl_trainer, train_loader, val_loader,
        epochs=epochs, patience=4, verbose=False,
        ckpt_path=checkpoint_dir / f"{split_name}_abl_no_cross_attn.pt",
        load_checkpoint=load_checkpoint,
    )
    m_abl, _, _ = abl_trainer.evaluate(test_loader)
    results["TransGDA (No Cross-Attn)"] = m_abl

    # 6. TransGDA (Full Proposed Model)
    print("\n--- Evaluating Proposed Model: TransGDA (Dual Graph-Transformer + Cross-Attn) ---")
    trans_model = TransGDA(
        num_genes=data_mgr.meta.num_genes,
        num_diseases=data_mgr.meta.num_diseases,
        embed_dim=64,
        num_heads=4,
        ffn_dim=128,
        hidden_dim=128,
        dropout=0.1,
        gene_feat_dim=data_mgr.meta.num_genes,
        disease_feat_dim=data_mgr.meta.num_diseases,
    )
    trans_model.set_node_features(
        gene_feat=torch.tensor(ext_gene_sim, dtype=torch.float32),
        disease_feat=torch.tensor(ext_disease_sim, dtype=torch.float32),
    )
    if cold_genes is not None:
        trans_model.set_cold_start_nodes(cold_start_genes=cold_genes)
    if cold_diseases is not None:
        trans_model.set_cold_start_nodes(cold_start_diseases=cold_diseases)

    trans_trainer = TransGDATrainer(
        model=trans_model,
        graph_manager=graph_mgr,
        device=device,
        lr=1e-3,
        lambda_strength=1.0,
    )
    _fit_or_load(
        trans_trainer, train_loader, val_loader,
        epochs=epochs, patience=4, verbose=True,
        ckpt_path=checkpoint_dir / f"{split_name}_transgda_full.pt",
        load_checkpoint=load_checkpoint,
    )
    m_trans, _, _ = trans_trainer.evaluate(test_loader)
    results["TransGDA (Full)"] = m_trans

    print("\n" + format_metrics_table(results))
    return results


def main():
    parser = argparse.ArgumentParser(description="Run TransGDA benchmarks across CV splits.")
    parser.add_argument("--split", type=str, default="edge", choices=["edge", "gene", "disease", "all"], help="Split type")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs per neural model")
    parser.add_argument("--fast", action="store_true", help="Faster execution for quick verification")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save benchmark results")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="results/checkpoints",
        help="Directory where model checkpoints are saved/loaded",
    )
    parser.add_argument(
        "--load_checkpoint", action="store_true",
        help="Skip training and load existing checkpoints from --checkpoint_dir (if available)",
    )
    args = parser.parse_args()

    set_global_seed(42)

    device = get_optimal_device()
    print(f"Using compute device: {device}")

    data_mgr = DataLoaderManager(data_dir=".")
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "benchmark_results.json"
    all_results = {}
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}

    splits_to_run = ["edge", "gene", "disease"] if args.split == "all" else [args.split]

    for s in splits_to_run:
        res = run_benchmark_for_split(
            s, data_mgr, device,
            epochs=args.epochs,
            fast=args.fast,
            checkpoint_dir=ckpt_dir,
            load_checkpoint=args.load_checkpoint,
        )
        all_results[s] = res

    # Save to JSON
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll benchmark results saved to: {json_path}")


if __name__ == "__main__":
    main()

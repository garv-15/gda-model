"""
Interpretability Script:
Generates attention explanations and publication-quality figures for benchmark gene-disease pairs.
"""

import argparse
from pathlib import Path
import json
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import DataLoaderManager
from src.similarity import GraphNeighborhoodManager
from src.utils import set_global_seed
from src.models.transf_gda import TransGDA
from src.train import TransGDATrainer, get_optimal_device
from src.interpret import AttentionInterpreter


def run_interpretability(epochs: int = 10, output_dir: str = "results"):
    set_global_seed(42)
    out_path = Path(output_dir)
    fig_dir = out_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = get_optimal_device()
    data_mgr = DataLoaderManager(data_dir=".")
    split = data_mgr.create_edge_split(val_ratio=0.10, test_ratio=0.10, seed=42)

    print("Building dynamic leakage-free similarity graphs...")
    graph_mgr = GraphNeighborhoodManager(
        num_genes=data_mgr.meta.num_genes,
        num_diseases=data_mgr.meta.num_diseases,
        k_gene_neighbors=16,
        k_disease_neighbors=16,
    )
    graph_mgr.build_from_train_matrix(split.train_matrix_binary, metric="cosine")

    train_loader, val_loader, test_loader = data_mgr.build_torch_dataloaders(
        split, batch_size=256, shuffle_train=True
    )

    print(f"Training TransGDA for interpretability ({epochs} epochs)...")
    model = TransGDA(
        num_genes=data_mgr.meta.num_genes,
        num_diseases=data_mgr.meta.num_diseases,
        embed_dim=64,
        num_heads=4,
        ffn_dim=128,
        hidden_dim=128,
        dropout=0.1,
    )
    trainer = TransGDATrainer(
        model=model,
        graph_manager=graph_mgr,
        device=device,
        lr=1e-3,
        lambda_strength=1.0,
    )
    trainer.fit(train_loader, val_loader, epochs=epochs, patience=3, verbose=True)

    interpreter = AttentionInterpreter(model, graph_mgr, data_mgr.meta, device)

    # Select case-study benchmark pairs
    case_studies = [
        ("UNC13C", "parkinson's disease"),
        ("RIT2", "lymphoid leukemia"),
        ("AAK1", "nervous system disease"),
        ("TCF7L2", "type 2 diabetes mellitus"),
    ]

    report_lines = [
        "# TransGDA Attention-Based Interpretability Report",
        "\nThis report investigates which genes, biological pathways, and disease phenotypes drive predictions in TransGDA.\n",
    ]

    for gene, disease in case_studies:
        if gene not in data_mgr.meta.gene_to_idx or disease not in data_mgr.meta.disease_to_idx:
            continue
        print(f"\nExplaining pair: {gene} ↔ {disease}...")
        exp = interpreter.explain_pair(gene, disease, top_k=6)

        # Plot and save
        safe_g = gene.replace("/", "_")
        safe_d = disease.replace(" ", "_").replace("'", "")
        plot_file = fig_dir / f"attn_{safe_g}_{safe_d}.png"
        interpreter.plot_attention_explanation(exp, save_path=plot_file)
        print(f"Saved figure: {plot_file}")

        report_lines.append(f"## Case Study: {gene} ↔ {disease}")
        report_lines.append(f"- **Predicted Existence Probability**: `{exp['predicted_prob']:.4f}`")
        report_lines.append(f"- **Predicted Association Strength**: `{exp['predicted_strength']:.4f}`")
        report_lines.append(f"- **Figure**: `results/figures/{plot_file.name}`\n")

        report_lines.append("### Primary Causal Drivers (Integrated Gradients Attribution):")
        if exp.get("attribution_genes"):
            report_lines.append("**Top Gene Attributions:**")
            for item in exp["attribution_genes"][:5]:
                report_lines.append(f"- **{item['gene']}**: Attribution Score = `{item['attribution_score']:+.4f}`")
        if exp.get("attribution_diseases"):
            report_lines.append("\n**Top Disease Attributions:**")
            for item in exp["attribution_diseases"][:5]:
                report_lines.append(f"- **{item['disease']}**: Attribution Score = `{item['attribution_score']:+.4f}`")

        report_lines.append("\n### Structural Cross-Attention Context:")
        report_lines.append("**Gene Context Attention:**")
        for item in exp["top_contributing_genes"]:
            report_lines.append(f"- **{item['gene']}**: Attention Weight = `{item['attention_weight']:.4f}`")

        report_lines.append("\n**Disease Context Attention:**")
        for item in exp["top_contributing_diseases"]:
            report_lines.append(f"- **{item['disease']}**: Attention Weight = `{item['attention_weight']:.4f}`")
        report_lines.append("\n---\n")

    report_file = out_path / "interpretability_report.md"
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nInterpretability report saved to: {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    run_interpretability(epochs=args.epochs, output_dir=args.output_dir)

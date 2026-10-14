"""
Causal Knockout & Statistical Pathway Validation Runner:
Executes in-silico virtual knockouts, synthetic lethality evaluations,
and hypergeometric pathway over-representation analysis (ORA) across
clinical benchmark case studies and top novel candidate discoveries.
"""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import DataLoaderManager, GDADataset
from src.similarity import GraphNeighborhoodManager
from src.utils import set_global_seed
from src.models.transf_gda import TransGDA
from src.train import TransGDATrainer, get_optimal_device
from src.causal_knockout import InSilicoKnockoutEngine
from src.pathway_validation import PathwayEnrichmentEngine


def train_transgda_model(
    data_mgr: DataLoaderManager,
    split,
    graph_mgr: GraphNeighborhoodManager,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 256,
) -> TransGDA:
    print(f"\n--- Training TransGDA ({epochs} Epochs) on Leakage-Free Edge Split ---")
    model = TransGDA(
        num_genes=data_mgr.meta.num_genes,
        num_diseases=data_mgr.meta.num_diseases,
        embed_dim=64,
        num_heads=4,
        ffn_dim=128,
        hidden_dim=128,
        dropout=0.1,
    )

    train_pairs = np.vstack([split.train_edges, split.train_neg])
    train_labels = np.concatenate([
        np.ones(len(split.train_edges), dtype=np.float32),
        np.zeros(len(split.train_neg), dtype=np.float32),
    ])
    train_strengths = np.concatenate([
        data_mgr.strength_matrix[split.train_edges[:, 0], split.train_edges[:, 1]],
        np.zeros(len(split.train_neg), dtype=np.float32),
    ])
    train_loader = DataLoader(
        GDADataset(train_pairs, train_labels, train_strengths),
        batch_size=batch_size,
        shuffle=True,
    )

    val_pairs = np.vstack([split.val_edges, split.val_neg])
    val_labels = np.concatenate([
        np.ones(len(split.val_edges), dtype=np.float32),
        np.zeros(len(split.val_neg), dtype=np.float32),
    ])
    val_strengths = np.concatenate([
        data_mgr.strength_matrix[split.val_edges[:, 0], split.val_edges[:, 1]],
        np.zeros(len(split.val_neg), dtype=np.float32),
    ])
    val_loader = DataLoader(
        GDADataset(val_pairs, val_labels, val_strengths),
        batch_size=batch_size,
        shuffle=False,
    )

    trainer = TransGDATrainer(
        model=model,
        graph_manager=graph_mgr,
        device=device,
        lr=1e-3,
        lambda_strength=1.0,
    )
    trainer.fit(train_loader, val_loader, epochs=epochs, patience=4)
    return trainer.model


def main():
    parser = argparse.ArgumentParser(description="Run Causal Knockout & Pathway Validation")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs for TransGDA")
    parser.add_argument("--top_n_discoveries", type=int, default=15, help="Number of novel candidates to validate")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for output report and plots")
    args = parser.parse_args()

    set_global_seed(42)

    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = get_optimal_device()
    print(f"Device: {device}")

    # Load canonical data
    print("Loading canonical data and building leakage-free edge split...")
    data_mgr = DataLoaderManager(data_dir=".")
    split = data_mgr.create_edge_split(val_ratio=0.10, test_ratio=0.10, seed=42)

    # Similarity graphs
    graph_mgr = GraphNeighborhoodManager(
        num_genes=data_mgr.meta.num_genes,
        num_diseases=data_mgr.meta.num_diseases,
        k_gene_neighbors=16,
        k_disease_neighbors=16,
    )
    graph_mgr.build_from_train_matrix(split.train_matrix_binary, metric="cosine")

    # Train model
    model = train_transgda_model(data_mgr, split, graph_mgr, device, epochs=args.epochs)

    # Initialize engines
    ko_engine = InSilicoKnockoutEngine(
        model=model,
        graph_manager=graph_mgr,
        meta=data_mgr.meta,
        device=device,
    )
    pathway_engine = PathwayEnrichmentEngine(
        data_dir=".",
        universe_genes=set(data_mgr.meta.genes),
    )

    # 1. Benchmark Case Studies
    case_studies = [
        ("TCF7L2", "type 2 diabetes mellitus"),
        ("UNC13C", "parkinson's disease"),
        ("RIT2", "lymphoid leukemia"),
        ("AAK1", "nervous system disease"),
        ("TCF12", "breast cancer"),
    ]

    report_lines = [
        "# TransGDA Causal In-Silico Knockout & Statistical Pathway Validation Report",
        "",
        "This report details **causal epistatic attributions** (in-silico knockouts) and **hypergeometric Over-Representation Analysis (ORA)**",
        "to validate candidate gene–disease associations beyond correlation-based attention weights.",
        "",
        "---",
        "",
        "## Part 1: Benchmark Clinical Case Studies",
        "",
    ]

    print("\n" + "=" * 70)
    print("Executing In-Silico Virtual Knockouts & Pathway ORA on Case Studies")
    print("=" * 70)

    for gene, disease in case_studies:
        safe_gene = gene.replace(" ", "_")
        safe_dis = disease.replace(" ", "_").replace("'", "")
        print(f"\nEvaluating Case Study: {gene} ↔ {disease}")

        # Run Knockout
        ko_res = ko_engine.evaluate_pair_knockout(gene, disease)
        base_p = ko_res["base_prob"]
        base_s = ko_res["base_strength"]

        # Plot Waterfall
        plot_path = figures_dir / f"ko_{safe_gene}_{safe_dis}.png"
        ko_engine.plot_knockout_waterfall(ko_res, save_path=plot_path, top_n=10)
        print(f"Saved Waterfall Plot: {plot_path}")

        # Synthetic Lethality
        syn_res = ko_engine.evaluate_synthetic_lethality(gene, disease, top_k=5)

        # Extract essential & moderate neighbor drivers (excluding target gene itself)
        driver_items = [
            x for x in ko_res.get("neighbor_knockouts", ko_res["gene_knockouts"])
            if not x["is_central"] and x["role"] in ("Essential Activator", "Moderate Contributor")
        ]
        if not driver_items:
            driver_items = [x for x in ko_res.get("neighbor_knockouts", ko_res["gene_knockouts"]) if not x["is_central"]][:3]

        driver_names = [x["name"] for x in driver_items]
        driver_effects = {x["name"]: x["relative_effect"] for x in driver_items}

        # Pathway validation
        val_res = pathway_engine.validate_candidate_pair(
            candidate_gene=gene,
            target_disease=disease,
            causal_driver_genes=driver_names,
            pred_prob=base_p,
            pred_strength=base_s,
            driver_effects=driver_effects,
        )

        target_ko = ko_res.get("target_knockout")

        # Build Markdown section
        report_lines.append(f"### Case Study: **{gene}** ↔ **{disease}**")
        report_lines.append(f"- **Baseline Existence Probability**: `{base_p:.4f}`")
        report_lines.append(f"- **Baseline Association Strength**: `{base_s:.4f}`")
        if target_ko:
            report_lines.append(f"- **Target Gene Causal Impact**: `Δs = {target_ko['delta_strength']:+.4f}` (`{target_ko['relative_effect']:+.2%}`) — `{target_ko['role']}`")
        report_lines.append(f"- **Biological Plausibility Index (BPI)**: `{val_res['biological_plausibility_index']:.2f}`")
        report_lines.append(f"- **Target Disease DOID**: `{val_res['target_doid']}`")
        report_lines.append(f"- **Waterfall Plot**: `results/figures/ko_{safe_gene}_{safe_dis}.png`")
        report_lines.append("")
        report_lines.append("#### Causal Knockout Waterfall Ranking (Top 6 Neighbor Drivers):")
        report_lines.append("| Rank | Neighbor Gene | Position | Attention Weight | Knockout Strength | Causal Effect (Δs) | Relative Effect (ρ) | Epistatic Role |")
        report_lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")

        nbr_kos = ko_res.get("neighbor_knockouts", [x for x in ko_res["gene_knockouts"] if not x["is_central"]])
        for idx, item in enumerate(nbr_kos[:6], 1):
            report_lines.append(
                f"| {idx} | **{item['name']}** | "
                f"{item['position']} | {item['attention_weight']:.4f} | {item['ko_strength']:.4f} | "
                f"{item['delta_strength']:+.4f} | {item['relative_effect']:+.2%} | {item['role']} |"
            )

        report_lines.append("")
        if syn_res:
            report_lines.append("#### Second-Order Epistasis (Synthetic Lethality) Between Neighborhood Genes:")
            report_lines.append("| Gene Pair | Δs (Gene A) | Δs (Gene B) | Δs (Double Knockout) | Epistasis Score (E) | Interaction Classification |")
            report_lines.append("| :--- | :---: | :---: | :---: | :---: | :--- |")
            for pair in syn_res[:4]:
                report_lines.append(
                    f"| **{pair['gene_a']} + {pair['gene_b']}** | {pair['delta_s_a']:+.4f} | {pair['delta_s_b']:+.4f} | "
                    f"{pair['delta_s_double']:+.4f} | {pair['epistasis_score']:+.4f} | **{pair['interaction_type']}** |"
                )
            report_lines.append("")

        report_lines.append("#### Statistical Pathway Over-Representation Analysis (ORA):")
        report_lines.append(f"- **Target Pathway Overlap**: `{val_res['overlap_count']}` genes (`{', '.join(val_res['overlap_genes']) if val_res['overlap_genes'] else 'None'}`)")
        report_lines.append(f"- **Hypergeometric p-value**: `{val_res['p_value']:.4e}` | **FDR q-value**: `{val_res['fdr_q_value']:.4e}`")
        report_lines.append(f"- **Odds Ratio**: `{val_res['odds_ratio']:.2f}`")

        if val_res["top_enriched_pathways"]:
            report_lines.append("\nTop Over-Represented Biological Phenotypes:")
            report_lines.append("| Phenotype / Disease Term | Overlap | Odds Ratio | Fold Enrichment | p-value | FDR q-value |")
            report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
            for pway in val_res["top_enriched_pathways"][:4]:
                report_lines.append(
                    f"| {pway['term']} | {pway['overlap_count']}/{pway['term_size']} | "
                    f"{pway['odds_ratio']:.2f} | {pway['fold_enrichment']:.2f}x | "
                    f"{pway['p_value']:.3e} | {pway['fdr_q_value']:.3e} |"
                )

        report_lines.append("\n---\n")

    # 2. Novel Candidate Discoveries Validation
    print("\n" + "=" * 70)
    print("Validating Top Novel Candidate Discoveries via In-Silico KO & Pathway ORA")
    print("=" * 70)

    novel_csv_path = results_dir / "novel_candidates.csv"
    if novel_csv_path.exists():
        df_novel = pd.read_csv(novel_csv_path).head(args.top_n_discoveries)
    else:
        print("Running fresh screening for novel unobserved candidates...")
        from src.discovery import NovelCandidateDiscovery
        screener = NovelCandidateDiscovery(model, graph_mgr, data_mgr, device=device)
        df_novel = screener.screen_unobserved_pairs(min_prob=0.75).head(args.top_n_discoveries)

    validated_candidates = []
    report_lines.append("## Part 2: Statistical Validation of Top Novel Discoveries")
    report_lines.append("")
    report_lines.append("Every candidate hypothesis discovered among unobserved pairs is evaluated for:")
    report_lines.append("1. **Causal Necessity**: Drop in association strength upon virtual knockout of top driver genes.")
    report_lines.append("2. **Pathway Overlap**: Hypergeometric test against canonical disease gene sets.")
    report_lines.append("3. **Biological Plausibility Index (BPI)**: Composite metric weighting model confidence by causal necessity and FDR significance.")
    report_lines.append("")
    report_lines.append("| Rank | Gene | Disease | DOID | Pred Prob | Pred Strength | Essential Driver | Pathway Overlap | FDR q-value | BPI Score | Status |")
    report_lines.append("| :---: | :--- | :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- |")

    for rank, (_, row) in enumerate(df_novel.iterrows(), 1):
        cand_gene = str(row.get("gene_symbol", row.get("gene", "")))
        cand_dis = str(row.get("disease_name", row.get("disease", "")))
        cand_p = float(row.get("predicted_probability", row.get("existence_prob", 1.0)))
        cand_s = float(row.get("predicted_strength", row.get("strength", 1.0)))
        doid = str(row.get("doid", "NA"))

        # Causal knockout on novel pair
        ko_res = ko_engine.evaluate_pair_knockout(cand_gene, cand_dis)
        driver_items = [
            x for x in ko_res.get("neighbor_knockouts", ko_res["gene_knockouts"])
            if not x["is_central"] and x["role"] in ("Essential Activator", "Moderate Contributor")
        ]
        if not driver_items:
            driver_items = [x for x in ko_res.get("neighbor_knockouts", ko_res["gene_knockouts"]) if not x["is_central"]][:3]

        top_driver = driver_items[0]["name"] if driver_items else "Autonomous"
        driver_names = [x["name"] for x in driver_items]
        driver_effects = {x["name"]: x["relative_effect"] for x in driver_items}

        val_res = pathway_engine.validate_candidate_pair(
            candidate_gene=cand_gene,
            target_disease=cand_dis,
            causal_driver_genes=driver_names,
            pred_prob=cand_p,
            pred_strength=cand_s,
            driver_effects=driver_effects,
        )

        status = "✅ Statistically Validated" if val_res["is_statistically_validated"] else "🔬 Novel Hypothesis"

        report_lines.append(
            f"| {rank} | **{cand_gene}** | {cand_dis} | {doid} | {cand_p:.3f} | {cand_s:.3f} | "
            f"**{top_driver}** | {val_res['overlap_count']} genes | {val_res['fdr_q_value']:.3e} | "
            f"**{val_res['biological_plausibility_index']:.2f}** | {status} |"
        )

        validated_candidates.append({
            "rank": rank,
            "gene": cand_gene,
            "disease": cand_dis,
            "doid": doid,
            "existence_prob": cand_p,
            "predicted_strength": cand_s,
            "essential_driver": top_driver,
            "driver_overlap_count": val_res["overlap_count"],
            "hypergeometric_p_value": val_res["p_value"],
            "fdr_q_value": val_res["fdr_q_value"],
            "bpi_score": val_res["biological_plausibility_index"],
            "status": status,
        })

    report_lines.append("")
    report_lines.append("---")
    report_lines.append("*Report automatically generated by TransGDA Causal Knockout & Pathway Validation Engine.*")

    # Save Markdown report
    report_path = results_dir / "causal_and_pathway_validation_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nSaved Causal & Pathway Validation Report: {report_path}")

    # Save CSV
    df_val = pd.DataFrame(validated_candidates)
    val_csv_path = results_dir / "validated_novel_candidates.csv"
    df_val.to_csv(val_csv_path, index=False)
    print(f"Saved Validated Novel Candidates CSV: {val_csv_path}")

    print("\nCausal Knockout & Pathway Validation Suite completed successfully!")


if __name__ == "__main__":
    main()

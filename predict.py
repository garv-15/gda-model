"""
Single Pair Prediction & Interpretation CLI for TransGDA.

Allows users to query ANY arbitrary pair of gene and disease to get:
1. Association Existence Probability (0 to 1)
2. Predicted Continuous Association Strength
3. Top Causal Driver Genes (Integrated Gradients & Cross-Attention)
4. Top Disease Phenotype Drivers
5. Optional publication-quality attention plot
"""

import argparse
import difflib
import json
from pathlib import Path
import sys
import torch

from src.data import DataLoaderManager, DatasetMetadata
from src.similarity import GraphNeighborhoodManager
from src.models.transf_gda import TransGDA
from src.interpret import AttentionInterpreter
from src.train import get_optimal_device


def load_inference_pipeline(model_dir: str = "results/saved_model", data_dir: str = "."):
    model_path = Path(model_dir)
    device = get_optimal_device()

    # Load metadata and config
    with open(model_path / "config.json", "r") as f:
        config = json.load(f)

    data_mgr = DataLoaderManager(data_dir=data_dir)
    ext_gene_sim, ext_disease_sim = data_mgr.load_external_similarity_matrices()

    # Initialize model
    model = TransGDA(
        num_genes=config["num_genes"],
        num_diseases=config["num_diseases"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        ffn_dim=config["ffn_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config.get("dropout", 0.1),
        gene_feat_dim=config.get("gene_feat_dim", config["num_genes"]),
        disease_feat_dim=config.get("disease_feat_dim", config["num_diseases"]),
    )
    model.set_node_features(
        gene_feat=torch.tensor(ext_gene_sim, dtype=torch.float32),
        disease_feat=torch.tensor(ext_disease_sim, dtype=torch.float32),
    )

    weights_file = model_path / "transgda_weights.pt"
    if weights_file.exists():
        payload = torch.load(weights_file, map_location=device, weights_only=False)
        state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
        model.load_state_dict(state_dict)
        if "temperature" in payload and hasattr(model, "temperature"):
            t = payload["temperature"]
            if isinstance(t, (float, int)):
                model.temperature.data.fill_(t)
            elif isinstance(t, torch.Tensor):
                model.temperature.data.copy_(t)
    model.to(device)
    model.eval()

    # Load graph neighborhoods
    graph_mgr = GraphNeighborhoodManager(
        num_genes=config["num_genes"],
        num_diseases=config["num_diseases"],
        k_gene_neighbors=config.get("k_neighbors", 16),
        k_disease_neighbors=config.get("k_neighbors", 16),
    )
    nbr_file = model_path / "graph_neighborhoods.pt"
    if nbr_file.exists():
        nbr_dict = torch.load(nbr_file, map_location=device, weights_only=False)
        graph_mgr.gene_neighbor_idx = nbr_dict["gene_neighbor_idx"].to(device)
        graph_mgr.gene_neighbor_wt = nbr_dict["gene_neighbor_wt"].to(device)
        graph_mgr.disease_neighbor_idx = nbr_dict["disease_neighbor_idx"].to(device)
        graph_mgr.disease_neighbor_wt = nbr_dict["disease_neighbor_wt"].to(device)
    else:
        # Fallback to train matrix
        graph_mgr.build_from_train_matrix(data_mgr.binary_matrix, metric="cosine")

    interpreter = AttentionInterpreter(model, graph_mgr, data_mgr.meta, device)
    return interpreter, data_mgr


def find_entity_match(query: str, candidates: list, entity_type: str = "gene"):
    q_norm = query.strip().lower()
    # 1. Exact match (case-insensitive)
    for c in candidates:
        if c.lower() == q_norm:
            return c
    # 2. Substring match
    matches = [c for c in candidates if q_norm in c.lower()]
    if matches:
        return matches[0]
    # 3. Fuzzy match
    close = difflib.get_close_matches(query, candidates, n=3, cutoff=0.5)
    if close:
        return close[0]
    return None


def predict_pair(
    gene: str,
    disease: str,
    model_dir: str = "results/saved_model",
    top_k: int = 5,
    save_plot: bool = False,
    plot_path: str = None,
):
    print("=" * 65)
    print(" TransGDA Inference Engine: Arbitrary Pair Prediction")
    print("=" * 65)

    interpreter, data_mgr = load_inference_pipeline(model_dir=model_dir)

    # Resolve Gene
    matched_gene = find_entity_match(gene, data_mgr.meta.genes, "gene")
    if not matched_gene:
        print(f"[-] Error: Gene '{gene}' not found in canonical dataset (4,055 genes).")
        suggestions = difflib.get_close_matches(gene, data_mgr.meta.genes, n=5, cutoff=0.4)
        if suggestions:
            print(f"    Did you mean: {', '.join(suggestions)}?")
        sys.exit(1)

    # Resolve Disease
    matched_disease = find_entity_match(disease, data_mgr.meta.diseases, "disease")
    if not matched_disease:
        print(f"[-] Error: Disease '{disease}' not found in canonical dataset (350 diseases).")
        suggestions = difflib.get_close_matches(disease, data_mgr.meta.diseases, n=5, cutoff=0.4)
        if suggestions:
            print(f"    Did you mean: {', '.join(suggestions)}?")
        sys.exit(1)

    g_idx = data_mgr.meta.gene_to_idx[matched_gene]
    d_idx = data_mgr.meta.disease_to_idx[matched_disease]
    is_known = bool(data_mgr.binary_matrix[g_idx, d_idx] > 0)
    doid = ""
    try:
        doid_row = data_mgr.meta.disease_df[data_mgr.meta.disease_df["Disease"] == matched_disease]
        if not doid_row.empty and "DOID" in doid_row.columns:
            doid = doid_row["DOID"].values[0]
    except Exception:
        pass

    print(f"\n[+] Query Pair:")
    print(f"    - Gene:            {matched_gene} (Index: {g_idx})")
    print(f"    - Disease:         {matched_disease} {f'({doid})' if doid else ''} (Index: {d_idx})")
    print(f"    - Training Status: {'Known Association in Dataset' if is_known else 'Unobserved Pair (Novel Candidate)'}")

    # Run explanation and prediction
    exp = interpreter.explain_pair(matched_gene, matched_disease, top_k=top_k)

    prob = exp["predicted_prob"]
    strength = exp["predicted_strength"]

    confidence = "Very High" if prob >= 0.90 else "Moderate-High" if prob >= 0.70 else "Moderate" if prob >= 0.50 else "Low / Non-Associated"

    print("\n[+] Model Predictions:")
    print(f"    - Association Probability: {prob:.4f} ({confidence})")
    print(f"    - Association Strength:    {strength:.4f}")

    print(f"\n[+] Top {top_k} Biological Gene Drivers (Integrated Gradients):")
    for rank, item in enumerate(exp.get("attribution_genes", [])[:top_k], 1):
        score = item.get("attribution_score", item.get("attribution", 0.0))
        sign = "+" if score >= 0 else ""
        print(f"    {rank}. {item['gene']:15s} | Attribution: {sign}{score:.4f}")

    print(f"\n[+] Top {top_k} Phenotypic Disease Drivers (Integrated Gradients):")
    for rank, item in enumerate(exp.get("attribution_diseases", [])[:top_k], 1):
        score = item.get("attribution_score", item.get("attribution", 0.0))
        sign = "+" if score >= 0 else ""
        print(f"    {rank}. {item['disease']:35s} | Attribution: {sign}{score:.4f}")

    if save_plot:
        out_fig = Path(plot_path) if plot_path else Path(f"attn_{matched_gene}_{matched_disease.replace(' ', '_')}.png")
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        interpreter.plot_attention_explanation(exp, save_path=out_fig)
        print(f"\n[+] Attention explanation plot saved to: {out_fig}")

    print("=" * 65 + "\n")
    return exp


def main():
    parser = argparse.ArgumentParser(description="Predict and interpret any arbitrary Gene-Disease pair with TransGDA.")
    parser.add_argument("--gene", type=str, required=False, help="Gene symbol (e.g., TCF7L2, UNC13C, BRCA1)")
    parser.add_argument("--disease", type=str, required=False, help="Disease name (e.g., 'type 2 diabetes mellitus', 'parkinson\'s disease')")
    parser.add_argument("--model_dir", type=str, default="results/saved_model", help="Path to exported model folder")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top drivers to display")
    parser.add_argument("--plot", action="store_true", help="Generate and save an attention visualization plot")
    parser.add_argument("--output_fig", type=str, default=None, help="Custom output path for visualization plot")
    parser.add_argument("--search_gene", type=str, default=None, help="Search for available genes matching a query")
    parser.add_argument("--search_disease", type=str, default=None, help="Search for available diseases matching a query")

    args = parser.parse_args()

    # Search options
    if args.search_gene:
        data_mgr = DataLoaderManager(data_dir=".")
        matches = [g for g in data_mgr.meta.genes if args.search_gene.lower() in g.lower()]
        print(f"Found {len(matches)} matching genes: {matches[:25]}")
        return

    if args.search_disease:
        data_mgr = DataLoaderManager(data_dir=".")
        matches = [d for d in data_mgr.meta.diseases if args.search_disease.lower() in d.lower()]
        print(f"Found {len(matches)} matching diseases: {matches[:25]}")
        return

    if not args.gene or not args.disease:
        parser.print_help()
        print("\nExample command:")
        print('  python predict.py --gene TCF7L2 --disease "type 2 diabetes mellitus"')
        sys.exit(1)

    predict_pair(
        gene=args.gene,
        disease=args.disease,
        model_dir=args.model_dir,
        top_k=args.top_k,
        save_plot=args.plot,
        plot_path=args.output_fig,
    )


if __name__ == "__main__":
    main()

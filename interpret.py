"""
Attention-based interpretability and explanation generator for TransGDA:
- Extracts bilateral cross-attention and neighborhood self-attention maps
- Identifies key contributing neighbor genes and diseases for any prediction
- Visualizes attention weight distributions as publication-quality figures
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from .models.transf_gda import TransGDA
from .similarity import GraphNeighborhoodManager
from .data import DatasetMetadata
from .attribution import GradientAttributionEngine


class AttentionInterpreter:
    """Interprets predictions of TransGDA through unified gradient attribution and attention analysis."""

    def __init__(
        self,
        model: TransGDA,
        graph_manager: GraphNeighborhoodManager,
        meta: DatasetMetadata,
        device: Optional[torch.device] = None,
    ):
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.graph_manager = graph_manager
        self.meta = meta
        self.attr_engine = GradientAttributionEngine(
            model=self.model,
            graph_manager=self.graph_manager,
            meta=self.meta,
            device=self.device,
        )

    def explain_pair(
        self,
        gene: Union[str, int],
        disease: Union[str, int],
        top_k: int = 8,
    ) -> Dict[str, Union[str, float, list]]:
        """
        Generates full attention-based explanation for a specific gene-disease pair.
        """
        self.model.eval()

        # Resolve indices
        if isinstance(gene, str):
            g_idx = self.meta.gene_to_idx[gene]
            gene_name = gene
        else:
            g_idx = gene
            gene_name = self.meta.idx_to_gene[g_idx]

        if isinstance(disease, str):
            d_idx = self.meta.disease_to_idx[disease]
            disease_name = disease
        else:
            d_idx = disease
            disease_name = self.meta.idx_to_disease[d_idx]

        # Prepare tensors
        t_g = torch.tensor([g_idx], dtype=torch.long, device=self.device)
        t_d = torch.tensor([d_idx], dtype=torch.long, device=self.device)
        g_nbr_idx = self.graph_manager.gene_neighbor_idx[t_g]
        g_nbr_wt = self.graph_manager.gene_neighbor_wt[t_g]
        d_nbr_idx = self.graph_manager.disease_neighbor_idx[t_d]
        d_nbr_wt = self.graph_manager.disease_neighbor_wt[t_d]

        with torch.no_grad():
            preds = self.model(
                gene_idx=t_g,
                disease_idx=t_d,
                gene_nbr_idx=g_nbr_idx,
                gene_nbr_wt=g_nbr_wt,
                disease_nbr_idx=d_nbr_idx,
                disease_nbr_wt=d_nbr_wt,
                return_attention=True,
            )

        prob = float(preds["probs"][0].cpu().numpy())
        strength = float(preds["strengths"][0].cpu().numpy())

        # Extract Cross-Attention weights (averaged over attention heads)
        # attn_g2d: (1, num_heads, 1, 1+K_d) -> query=gene, keys=[disease, d_nbr_0, ..., d_nbr_{K-1}]
        attn_g2d = preds["attn_g2d"][0].mean(dim=0).squeeze(0).cpu().numpy()  # (1+K_d,)
        # attn_d2g: (1, num_heads, 1, 1+K_g) -> query=disease, keys=[gene, g_nbr_0, ..., g_nbr_{K-1}]
        attn_d2g = preds["attn_d2g"][0].mean(dim=0).squeeze(0).cpu().numpy()  # (1+K_g,)

        # Disease context labels
        d_nbr_indices = d_nbr_idx[0].cpu().numpy()
        d_labels = [disease_name] + [self.meta.idx_to_disease[int(idx)] for idx in d_nbr_indices]

        # Gene context labels
        g_nbr_indices = g_nbr_idx[0].cpu().numpy()
        g_labels = [gene_name] + [self.meta.idx_to_gene[int(idx)] for idx in g_nbr_indices]

        # Top contributing diseases to gene representation
        d_top_order = np.argsort(-attn_g2d)[:top_k]
        top_diseases = [
            {"disease": d_labels[i], "attention_weight": float(attn_g2d[i])}
            for i in d_top_order
        ]

        # Top contributing genes to disease representation
        g_top_order = np.argsort(-attn_d2g)[:top_k]
        top_genes = [
            {"gene": g_labels[i], "attention_weight": float(attn_d2g[i])}
            for i in g_top_order
        ]

        # Gradient Attribution (Integrated Gradients)
        try:
            attr_res = self.attr_engine.attribute_pair(gene, disease, target="strength", steps=15)
            attr_genes = attr_res["gene_drivers"][:top_k]
            attr_diseases = attr_res["disease_drivers"][:top_k]
        except Exception:
            attr_genes = []
            attr_diseases = []

        return {
            "gene": gene_name,
            "disease": disease_name,
            "predicted_prob": prob,
            "predicted_strength": strength,
            "top_contributing_genes": top_genes,
            "top_contributing_diseases": top_diseases,
            "attribution_genes": attr_genes,
            "attribution_diseases": attr_diseases,
            "raw_attn_g2d": attn_g2d,
            "raw_attn_d2g": attn_d2g,
            "d_labels": d_labels,
            "g_labels": g_labels,
        }

    def plot_attention_explanation(
        self,
        explanation: Dict,
        save_path: Optional[Union[str, Path]] = None,
    ):
        """
        Creates a side-by-side bar chart of the attention weights driving the prediction.
        """
        gene_name = explanation["gene"]
        disease_name = explanation["disease"]
        prob = explanation["predicted_prob"]
        strength = explanation["predicted_strength"]

        top_g = explanation["top_contributing_genes"]
        top_d = explanation["top_contributing_diseases"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"TransGDA Attention Explanation: {gene_name} ↔ {disease_name}\n"
            f"(Existence Prob: {prob:.3f} | Predicted Strength: {strength:.3f})",
            fontsize=14,
            fontweight="bold",
        )

        # Plot 1: Top genes contributing to disease connection
        g_names = [item["gene"] for item in reversed(top_g)]
        g_weights = [item["attention_weight"] for item in reversed(top_g)]
        colors_g = ["#1f77b4" if g != gene_name else "#ff7f0e" for g in g_names]

        ax1.barh(g_names, g_weights, color=colors_g, edgecolor="black", alpha=0.85)
        ax1.set_xlabel("Cross-Attention Weight (Disease → Gene Context)")
        ax1.set_title("Biological Gene Drivers")
        ax1.grid(axis="x", linestyle="--", alpha=0.5)

        # Plot 2: Top diseases contributing to gene connection
        d_names = [item["disease"] for item in reversed(top_d)]
        d_weights = [item["attention_weight"] for item in reversed(top_d)]
        colors_d = ["#2ca02c" if d != disease_name else "#d62728" for d in d_names]

        ax2.barh(d_names, d_weights, color=colors_d, edgecolor="black", alpha=0.85)
        ax2.set_xlabel("Cross-Attention Weight (Gene → Disease Context)")
        ax2.set_title("Phenotypic Disease Drivers")
        ax2.grid(axis="x", linestyle="--", alpha=0.5)

        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

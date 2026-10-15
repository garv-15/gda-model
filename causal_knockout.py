"""
In-Silico Virtual Gene Knockout & Causal Epistatic Attribution Engine for TransGDA:
- simulate virtual CRISPR/knockdowns by systematically masking neighbor gene & disease tokens
- evaluate counterfactual shifts in predicted association strength (Delta_s) and probability (Delta_p)
- classify epistatic roles: Essential Activator, Moderate Contributor, Passive Bystander, Inhibitory Buffer
- Detects second-order synthetic lethality / epistasis between gene pairs
- Generates publication-ready causal knockout waterfall and interaction plots
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .models.transf_gda import TransGDA
from .similarity import GraphNeighborhoodManager
from .data import DatasetMetadata


class InSilicoKnockoutEngine:

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

    def _prepare_pair_tensors(
        self,
        gene: Union[str, int],
        disease: Union[str, int],
    ) -> Tuple[int, int, str, str, Dict[str, torch.Tensor]]:
        """Resolves indices, labels, and neighborhood tensors for a pair."""
        if isinstance(gene, str):
            g_idx = self.meta.gene_to_idx[gene]
            gene_name = gene
        else:
            g_idx = int(gene)
            gene_name = self.meta.idx_to_gene[g_idx]

        if isinstance(disease, str):
            d_idx = self.meta.disease_to_idx[disease]
            disease_name = disease
        else:
            d_idx = int(disease)
            disease_name = self.meta.idx_to_disease[d_idx]

        t_g = torch.tensor([g_idx], dtype=torch.long, device=self.device)
        t_d = torch.tensor([d_idx], dtype=torch.long, device=self.device)

        g_nbr_idx = self.graph_manager.gene_neighbor_idx[t_g]
        g_nbr_wt = self.graph_manager.gene_neighbor_wt[t_g]
        d_nbr_idx = self.graph_manager.disease_neighbor_idx[t_d]
        d_nbr_wt = self.graph_manager.disease_neighbor_wt[t_d]

        tensors = {
            "gene_idx": t_g,
            "disease_idx": t_d,
            "gene_nbr_idx": g_nbr_idx,
            "gene_nbr_wt": g_nbr_wt,
            "disease_nbr_idx": d_nbr_idx,
            "disease_nbr_wt": d_nbr_wt,
        }
        return g_idx, d_idx, gene_name, disease_name, tensors

    @torch.no_grad()
    def evaluate_pair_knockout(
        self,
        gene: Union[str, int],
        disease: Union[str, int],
    ) -> Dict[str, Union[str, float, list, dict]]:
        """
        Executes a full systematic in-silico knockout sweep across:
        1. Baseline (unperturbed)
        2. Central gene knockout
        3. All individual gene neighborhood knockouts
        4. All individual disease neighborhood knockouts
        """
        self.model.eval()
        g_idx, d_idx, gene_name, disease_name, tensors = self._prepare_pair_tensors(gene, disease)

        # Baseline prediction
        base_preds = self.model(**tensors, return_attention=True)
        base_prob = float(base_preds["probs"][0].cpu().item())
        base_strength = float(base_preds["strengths"][0].cpu().item())

        # Baseline attention distributions
        # attn_d2g: (num_heads, 1, 1+K_g) -> query=disease, keys=[gene, g_nbr_0, ..., g_nbr_{K-1}]
        attn_d2g = base_preds["attn_d2g"][0].mean(dim=0).squeeze(0).cpu().numpy()
        # attn_g2d: (num_heads, 1, 1+K_d) -> query=gene, keys=[disease, d_nbr_0, ..., d_nbr_{K-1}]
        attn_g2d = base_preds["attn_g2d"][0].mean(dim=0).squeeze(0).cpu().numpy()

        k_g = tensors["gene_nbr_idx"].size(1)
        k_d = tensors["disease_nbr_idx"].size(1)

        gene_nbr_indices = tensors["gene_nbr_idx"][0].cpu().numpy()
        gene_nbr_names = [self.meta.idx_to_gene[int(idx)] for idx in gene_nbr_indices]

        disease_nbr_indices = tensors["disease_nbr_idx"][0].cpu().numpy()
        disease_nbr_names = [self.meta.idx_to_disease[int(idx)] for idx in disease_nbr_indices]

        eps = 1e-8
        gene_knockout_results = []

        # 1. Knockout each gene in neighborhood (including central gene at pos 0)
        for pos in range(1 + k_g):
            mask_g = torch.zeros((1, 1 + k_g), dtype=torch.bool, device=self.device)
            mask_g[0, pos] = True

            ko_preds = self.model(**tensors, gene_mask=mask_g)
            ko_p = float(ko_preds["probs"][0].cpu().item())
            ko_s = float(ko_preds["strengths"][0].cpu().item())

            delta_s = base_strength - ko_s
            rel_effect = delta_s / (base_strength + eps)
            delta_p = base_prob - ko_p

            if pos == 0:
                name = gene_name
                is_central = True
            else:
                name = gene_nbr_names[pos - 1]
                is_central = False

            attn_w = float(attn_d2g[pos]) if pos < len(attn_d2g) else 0.0

            if rel_effect >= 0.10:
                role = "Essential Activator"
            elif rel_effect >= 0.03:
                role = "Moderate Contributor"
            elif rel_effect <= -0.03:
                role = "Inhibitory Buffer"
            else:
                role = "Passive Bystander"

            gene_knockout_results.append({
                "name": name,
                "position": pos,
                "is_central": is_central,
                "attention_weight": attn_w,
                "ko_strength": ko_s,
                "delta_strength": delta_s,
                "relative_effect": rel_effect,
                "ko_prob": ko_p,
                "delta_prob": delta_p,
                "role": role,
            })

        # 2. Knockout each disease neighbor (pos 1 to K_d)
        disease_knockout_results = []
        for pos in range(1, 1 + k_d):
            mask_d = torch.zeros((1, 1 + k_d), dtype=torch.bool, device=self.device)
            mask_d[0, pos] = True

            ko_preds = self.model(**tensors, disease_mask=mask_d)
            ko_p = float(ko_preds["probs"][0].cpu().item())
            ko_s = float(ko_preds["strengths"][0].cpu().item())

            delta_s = base_strength - ko_s
            rel_effect = delta_s / (base_strength + eps)
            delta_p = base_prob - ko_p

            name = disease_nbr_names[pos - 1]
            attn_w = float(attn_g2d[pos]) if pos < len(attn_g2d) else 0.0

            if rel_effect >= 0.10:
                role = "Essential Phenotype Driver"
            elif rel_effect >= 0.03:
                role = "Moderate Phenotype Contributor"
            elif rel_effect <= -0.03:
                role = "Inhibitory Phenotype Buffer"
            else:
                role = "Passive Phenotype Bystander"

            disease_knockout_results.append({
                "name": name,
                "position": pos,
                "attention_weight": attn_w,
                "ko_strength": ko_s,
                "delta_strength": delta_s,
                "relative_effect": rel_effect,
                "ko_prob": ko_p,
                "delta_prob": delta_p,
                "role": role,
            })

        # Rank results by delta_strength descending
        gene_knockout_results.sort(key=lambda x: x["delta_strength"], reverse=True)
        disease_knockout_results.sort(key=lambda x: x["delta_strength"], reverse=True)

        central_item = next((item for item in gene_knockout_results if item["is_central"]), None)
        neighbor_knockouts = [item for item in gene_knockout_results if not item["is_central"]]
        neighbor_knockouts.sort(key=lambda x: x["delta_strength"], reverse=True)

        return {
            "gene": gene_name,
            "disease": disease_name,
            "base_prob": base_prob,
            "base_strength": base_strength,
            "target_knockout": central_item,
            "neighbor_knockouts": neighbor_knockouts,
            "gene_knockouts": gene_knockout_results,
            "disease_knockouts": disease_knockout_results,
        }

    @torch.no_grad()
    def evaluate_synthetic_lethality(
        self,
        gene: Union[str, int],
        disease: Union[str, int],
        top_k: int = 6,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Evaluates pairwise combinatorial knockouts to detect second-order epistasis (synthetic lethality):
        E(i, j) = Delta_s({i, j}) - (Delta_s(i) + Delta_s(j))
        A positive E(i, j) indicates synergistic necessity / synthetic lethality.
        """
        self.model.eval()
        g_idx, d_idx, gene_name, disease_name, tensors = self._prepare_pair_tensors(gene, disease)

        # Baseline
        base_preds = self.model(**tensors)
        base_s = float(base_preds["strengths"][0].cpu().item())
        k_g = tensors["gene_nbr_idx"].size(1)

        # Get single knockouts
        single_res = self.evaluate_pair_knockout(gene, disease)
        # Select top candidates (excluding central gene to focus on neighborhood epistasis)
        candidate_genes = [
            item for item in single_res["gene_knockouts"]
            if not item["is_central"]
        ][:top_k]

        comb_results = []
        for i in range(len(candidate_genes)):
            for j in range(i + 1, len(candidate_genes)):
                g1 = candidate_genes[i]
                g2 = candidate_genes[j]
                pos1, pos2 = g1["position"], g2["position"]

                # Double knockout mask
                double_mask = torch.zeros((1, 1 + k_g), dtype=torch.bool, device=self.device)
                double_mask[0, pos1] = True
                double_mask[0, pos2] = True

                ko_preds = self.model(**tensors, gene_mask=double_mask)
                ko_s = float(ko_preds["strengths"][0].cpu().item())
                delta_s_pair = base_s - ko_s

                # Expected additive effect
                expected_additive = g1["delta_strength"] + g2["delta_strength"]
                # Epistasis score
                epistasis_score = delta_s_pair - expected_additive

                # Calibrate interaction using relative epistasis ratio over additive expectation
                denom = max(abs(expected_additive), 1e-4)
                rel_epistasis = epistasis_score / denom

                if epistasis_score > 0.001 and rel_epistasis >= 0.10:
                    interaction = "Synergistic / Synthetic Lethal"
                elif epistasis_score < -0.001 and rel_epistasis <= -0.10:
                    interaction = "Antagonistic / Redundant"
                else:
                    interaction = "Additive"

                comb_results.append({
                    "gene_a": g1["name"],
                    "gene_b": g2["name"],
                    "delta_s_a": g1["delta_strength"],
                    "delta_s_b": g2["delta_strength"],
                    "delta_s_double": delta_s_pair,
                    "expected_additive": expected_additive,
                    "epistasis_score": epistasis_score,
                    "relative_epistasis": rel_epistasis,
                    "interaction_type": interaction,
                })

        comb_results.sort(key=lambda x: x["epistasis_score"], reverse=True)
        return comb_results

    def plot_knockout_waterfall(
        self,
        ko_results: Dict,
        save_path: Optional[Union[str, Path]] = None,
        top_n: int = 10,
    ):
        """
        Visualizes causal knockout delta strength (Delta s) as a publication-ready waterfall plot,
        color-coded by epistatic role.
        """
        gene_name = ko_results["gene"]
        disease_name = ko_results["disease"]
        base_p = ko_results["base_prob"]
        base_s = ko_results["base_strength"]

        g_kos = ko_results["gene_knockouts"][:top_n]

        names = [item["name"] + (" (Target)" if item["is_central"] else "") for item in reversed(g_kos)]
        deltas = [item["delta_strength"] for item in reversed(g_kos)]
        roles = [item["role"] for item in reversed(g_kos)]

        # Color mapping
        color_map = {
            "Essential Activator": "#2ca02c",      # Forest green
            "Moderate Contributor": "#1f77b4",     # Steel blue
            "Passive Bystander": "#7f7f7f",        # Slate gray
            "Inhibitory Buffer": "#d62728",        # Crimson red
        }
        bar_colors = [color_map.get(r, "#1f77b4") for r in roles]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(names, deltas, color=bar_colors, edgecolor="black", alpha=0.88)

        # Baseline zero line
        ax.axvline(0, color="black", linestyle="-", linewidth=1.0)
        ax.set_xlabel(r"Causal Effect $\Delta \hat{s} = \hat{s}_{\mathrm{baseline}} - \hat{s}_{\mathrm{knockout}}$", fontsize=12)
        ax.set_title(
            f"In-Silico Virtual Knockout Attribution: {gene_name} ↔ {disease_name}\n"
            f"(Baseline Prob: {base_p:.3f} | Baseline Strength: {base_s:.3f})",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        # Legend
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=color_map["Essential Activator"], ec="black", label="Essential Activator (Δs ≥ 10%)"),
            plt.Rectangle((0, 0), 1, 1, color=color_map["Moderate Contributor"], ec="black", label="Moderate Contributor (Δs ≥ 3%)"),
            plt.Rectangle((0, 0), 1, 1, color=color_map["Passive Bystander"], ec="black", label="Passive Bystander (|Δs| < 3%)"),
            plt.Rectangle((0, 0), 1, 1, color=color_map["Inhibitory Buffer"], ec="black", label="Inhibitory Buffer (Δs ≤ -3%)"),
        ]
        ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=10)

        plt.tight_layout()
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

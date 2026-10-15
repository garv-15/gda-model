"""
Gradient-based attribution engine for TransGDA:
Computes Integrated Gradients and Input x Gradient attribution scores over
functional gene and disease neighbor tokens to provide faithful, causal
feature importance beyond correlation-based attention maps.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from .models.transf_gda import TransGDA
from .similarity import GraphNeighborhoodManager
from .data import DatasetMetadata


class GradientAttributionEngine:
    """
    Computes token-level gradient attribution (Integrated Gradients / Input x Gradient)
    over neighborhood tokens for specific gene-disease pairs.
    """

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

    def attribute_pair(
        self,
        gene: Union[str, int],
        disease: Union[str, int],
        target: str = "strength",
        method: str = "integrated_gradients",
        steps: int = 20,
    ) -> Dict[str, Union[str, float, list, np.ndarray]]:
        """
        Computes attribution scores for all neighbor tokens of a gene-disease pair.

        Args:
            gene: Gene symbol or index.
            disease: Disease name or index.
            target: 'strength' (association strength) or 'prob' (existence logit).
            method: 'integrated_gradients' or 'input_x_gradient'.
            steps: Number of interpolation steps for Integrated Gradients.

        Returns:
            Dictionary containing gene and disease neighbor attribution scores and rankings.
        """
        self.model.eval()
        g_idx, d_idx, gene_name, disease_name, tensors = self._prepare_pair_tensors(gene, disease)

        # Baseline unperturbed forward pass
        with torch.no_grad():
            base_preds = self.model(**tensors)
            base_prob = float(base_preds["probs"][0].cpu().item())
            base_strength = float(base_preds["strengths"][0].cpu().item())

        # Extract embeddings
        with torch.no_grad():
            e_g = self.model._embed_gene(tensors["gene_idx"])
            e_g_nbrs = self.model._embed_gene(tensors["gene_nbr_idx"])
            e_d = self.model._embed_disease(tensors["disease_idx"])
            e_d_nbrs = self.model._embed_disease(tensors["disease_nbr_idx"])

        k_g = tensors["gene_nbr_idx"].size(1)
        k_d = tensors["disease_nbr_idx"].size(1)

        gene_nbr_indices = tensors["gene_nbr_idx"][0].cpu().numpy()
        gene_nbr_names = [self.meta.idx_to_gene[int(idx)] for idx in gene_nbr_indices]

        disease_nbr_indices = tensors["disease_nbr_idx"][0].cpu().numpy()
        disease_nbr_names = [self.meta.idx_to_disease[int(idx)] for idx in disease_nbr_indices]

        # Attribution computation
        if method == "integrated_gradients" and steps > 1:
            alphas = np.linspace(1.0 / steps, 1.0, steps)
        else:
            alphas = [1.0]

        grad_g_nbrs_sum = torch.zeros_like(e_g_nbrs)
        grad_d_nbrs_sum = torch.zeros_like(e_d_nbrs)

        for alpha in alphas:
            e_g_nbrs_step = (alpha * e_g_nbrs).detach().clone().requires_grad_(True)
            e_d_nbrs_step = (alpha * e_d_nbrs).detach().clone().requires_grad_(True)

            out = self.model(
                gene_idx=tensors["gene_idx"],
                disease_idx=tensors["disease_idx"],
                gene_nbr_idx=tensors["gene_nbr_idx"],
                gene_nbr_wt=tensors["gene_nbr_wt"],
                disease_nbr_idx=tensors["disease_nbr_idx"],
                disease_nbr_wt=tensors["disease_nbr_wt"],
                e_g=e_g,
                e_g_nbrs=e_g_nbrs_step,
                e_d=e_d,
                e_d_nbrs=e_d_nbrs_step,
            )

            if target == "strength":
                obj = out["strengths"][0]
            else:
                obj = out["logits"][0]

            grads = torch.autograd.grad(obj, [e_g_nbrs_step, e_d_nbrs_step])
            grad_g_nbrs_sum += grads[0]
            grad_d_nbrs_sum += grads[1]

        # Integrated Gradients: (input - baseline) * mean_grad
        attr_g_nbrs = (e_g_nbrs * (grad_g_nbrs_sum / len(alphas)))[0]  # (K_g, D)
        attr_d_nbrs = (e_d_nbrs * (grad_d_nbrs_sum / len(alphas)))[0]  # (K_d, D)

        # Sum across embedding dimensions to obtain scalar importance per token
        score_g = attr_g_nbrs.sum(dim=-1).detach().cpu().numpy()  # (K_g,)
        score_d = attr_d_nbrs.sum(dim=-1).detach().cpu().numpy()  # (K_d,)

        # Package gene drivers
        gene_drivers = []
        for i, (name, sc) in enumerate(zip(gene_nbr_names, score_g)):
            gene_drivers.append({
                "gene": name,
                "position": i + 1,
                "attribution_score": float(sc),
            })
        gene_drivers.sort(key=lambda x: x["attribution_score"], reverse=True)

        # Package disease drivers
        disease_drivers = []
        for i, (name, sc) in enumerate(zip(disease_nbr_names, score_d)):
            disease_drivers.append({
                "disease": name,
                "position": i + 1,
                "attribution_score": float(sc),
            })
        disease_drivers.sort(key=lambda x: x["attribution_score"], reverse=True)

        return {
            "gene": gene_name,
            "disease": disease_name,
            "target": target,
            "method": method,
            "predicted_prob": base_prob,
            "predicted_strength": base_strength,
            "gene_drivers": gene_drivers,
            "disease_drivers": disease_drivers,
            "raw_gene_scores": score_g,
            "raw_disease_scores": score_d,
        }

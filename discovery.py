"""
Screens unobserved pairs, ranks candidates by composite association score,
and annotates potential novel candidates for biological investigation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch

from .models.transf_gda import TransGDA
from .similarity import GraphNeighborhoodManager
from .data import DataLoaderManager
from .interpret import AttentionInterpreter


class NovelCandidateDiscovery:
    """Discovers and ranks candidate gene-disease associations with high confidence."""

    def __init__(
        self,
        model: TransGDA,
        graph_manager: GraphNeighborhoodManager,
        data_manager: DataLoaderManager,
        device: Optional[torch.device] = None,
    ):
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.graph_manager = graph_manager
        self.data_manager = data_manager
        self.meta = data_manager.meta
        self.interpreter = AttentionInterpreter(model, graph_manager, self.meta, self.device)

    @torch.no_grad()
    def screen_unobserved_pairs(
        self,
        min_prob: float = 0.70,
        batch_size: int = 1024,
        target_diseases: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Screens all unobserved pairs (A_gd == 0) and returns candidates exceeding min_prob.
        """
        self.model.eval()
        full_bin = self.data_manager.binary_matrix

        # Target disease indices
        if target_diseases is not None:
            d_indices = [self.meta.disease_to_idx[d] for d in target_diseases if d in self.meta.disease_to_idx]
        else:
            d_indices = list(range(self.meta.num_diseases))

        # Collect unobserved coordinates
        unobs_g = []
        unobs_d = []
        for d in d_indices:
            unobs_genes = np.where(full_bin[:, d] == 0)[0]
            unobs_g.extend(unobs_genes)
            unobs_d.extend([d] * len(unobs_genes))

        unobs_g = np.array(unobs_g, dtype=np.int64)
        unobs_d = np.array(unobs_d, dtype=np.int64)

        n_pairs = len(unobs_g)
        print(f"Screening {n_pairs:,} unobserved pairs for novel association hypotheses...")

        all_probs = []
        all_strengths = []

        for i in range(0, n_pairs, batch_size):
            end_i = min(i + batch_size, n_pairs)
            g_batch = torch.tensor(unobs_g[i:end_i], dtype=torch.long, device=self.device)
            d_batch = torch.tensor(unobs_d[i:end_i], dtype=torch.long, device=self.device)

            g_nbr_idx = self.graph_manager.gene_neighbor_idx[g_batch]
            g_nbr_wt = self.graph_manager.gene_neighbor_wt[g_batch]
            d_nbr_idx = self.graph_manager.disease_neighbor_idx[d_batch]
            d_nbr_wt = self.graph_manager.disease_neighbor_wt[d_batch]

            preds = self.model(
                gene_idx=g_batch,
                disease_idx=d_batch,
                gene_nbr_idx=g_nbr_idx,
                gene_nbr_wt=g_nbr_wt,
                disease_nbr_idx=d_nbr_idx,
                disease_nbr_wt=d_nbr_wt,
            )

            all_probs.append(preds["probs"].detach().cpu().numpy())
            all_strengths.append(preds["strengths"].detach().cpu().numpy())

        probs = np.concatenate(all_probs)
        strengths = np.concatenate(all_strengths)

        # Filter by threshold
        mask = probs >= min_prob
        cand_g = unobs_g[mask]
        cand_d = unobs_d[mask]
        cand_probs = probs[mask]
        cand_str = strengths[mask]

        composite_score = cand_probs * cand_str

        records = []
        for g_idx, d_idx, p, s, comp in zip(cand_g, cand_d, cand_probs, cand_str, composite_score):
            gene_name = self.meta.idx_to_gene[g_idx]
            disease_name = self.meta.idx_to_disease[d_idx]
            g_row = self.meta.gene_df.iloc[g_idx]
            d_row = self.meta.disease_df.iloc[d_idx]

            records.append({
                "gene_symbol": gene_name,
                "disease_name": disease_name,
                "doid": d_row.get("DOID", "N/A"),
                "gene_id": g_row.get("GeneID", "N/A"),
                "ensemble_acc": g_row.get("Ensemble Acc", "N/A"),
                "predicted_probability": float(p),
                "predicted_strength": float(s),
                "composite_score": float(comp),
            })

        df_candidates = pd.DataFrame(records)
        if not df_candidates.empty:
            df_candidates = df_candidates.sort_values(by="composite_score", ascending=False).reset_index(drop=True)

        return df_candidates

    def generate_discovery_report(
        self,
        df_candidates: pd.DataFrame,
        top_n: int = 20,
        output_csv: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Exports top novel candidates and generates markdown summary table with top attention drivers.
        """
        top_df = df_candidates.head(top_n).copy()

        if output_csv is not None:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            top_df.to_csv(output_csv, index=False)

        lines = [
            f"| Rank | Gene Symbol | Disease | DOID | Existence Prob | Pred Strength | Composite Score | Top Explaining Gene |",
            f"| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :--- |",
        ]

        for rank, (_, row) in enumerate(top_df.iterrows(), 1):
            # Get quick explanation
            exp = self.interpreter.explain_pair(row["gene_symbol"], row["disease_name"], top_k=2)
            top_gene_driver = exp["top_contributing_genes"][0]["gene"] if exp["top_contributing_genes"] else "N/A"
            lines.append(
                f"| {rank} | **{row['gene_symbol']}** | {row['disease_name']} | {row['doid']} "
                f"| {row['predicted_probability']:.3f} | {row['predicted_strength']:.3f} "
                f"| {row['composite_score']:.3f} | {top_gene_driver} |"
            )

        return "\n".join(lines)

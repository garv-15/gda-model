"""
Leakage-free similarity network computation, top-K graph neighborhood extraction,
and leaked similarity matrix loader for audit benchmarks.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch


def compute_cosine_similarity(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Vectorized cosine similarity with zero-norm safeguard.
    matrix: Shape (N, D).
    returns: Shape (N, N).
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    normed = matrix / norms
    sim = np.matmul(normed, normed.T)
    # Numerical clip to [-1, 1] then clamp negative values to 0 for similarity network
    sim = np.clip(sim, 0.0, 1.0)
    # Ensure diagonal is 1.0
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float32)


def compute_jaccard_similarity(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Vectorized Jaccard similarity for binary bipartite networks.
    matrix: Shape (N, D).
    returns: Shape (N, N).
    """
    binary = (matrix > 0).astype(np.float32)
    intersection = np.matmul(binary, binary.T)
    degrees = binary.sum(axis=1, keepdims=True)
    union = degrees + degrees.T - intersection
    union = np.where(union < eps, 1.0, union)
    jaccard = intersection / union
    jaccard = np.clip(jaccard, 0.0, 1.0)
    np.fill_diagonal(jaccard, 1.0)
    return jaccard.astype(np.float32)


class GraphNeighborhoodManager:
    """
    Extracts and manages top-K functional neighbors for genes and diseases.
    Guarantees zero biological/data leakage when initialized with A_train.
    """

    def __init__(
        self,
        num_genes: int,
        num_diseases: int,
        k_gene_neighbors: int = 16,
        k_disease_neighbors: int = 16,
    ):
        self.num_genes = num_genes
        self.num_diseases = num_diseases
        self.k_gene = k_gene_neighbors
        self.k_disease = k_disease_neighbors

        self.gene_sim: Optional[np.ndarray] = None
        self.disease_sim: Optional[np.ndarray] = None

        self.gene_neighbor_idx: Optional[torch.Tensor] = None
        self.gene_neighbor_wt: Optional[torch.Tensor] = None
        self.disease_neighbor_idx: Optional[torch.Tensor] = None
        self.disease_neighbor_wt: Optional[torch.Tensor] = None

    def build_from_train_matrix(
        self,
        train_matrix_binary: np.ndarray,
        metric: str = "cosine",
        external_gene_sim: Optional[np.ndarray] = None,
        external_disease_sim: Optional[np.ndarray] = None,
        cold_start_genes: Optional[Union[set, list, np.ndarray]] = None,
        cold_start_diseases: Optional[Union[set, list, np.ndarray]] = None,
    ):
        """
        Dynamically computes leakage-free similarity strictly from train associations A_train.
        When cold_start_genes or cold_start_diseases are provided in inductive splits,
        their neighborhood similarities are sourced from external similarity matrices
        constrained to training neighbors.
        """
        # Gene similarity from rows (genes) across training disease edges
        if metric == "cosine":
            self.gene_sim = compute_cosine_similarity(train_matrix_binary)
            # Disease similarity from columns (diseases) across training gene edges
            self.disease_sim = compute_cosine_similarity(train_matrix_binary.T)
        elif metric == "jaccard":
            self.gene_sim = compute_jaccard_similarity(train_matrix_binary)
            self.disease_sim = compute_jaccard_similarity(train_matrix_binary.T)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if cold_start_genes is not None and external_gene_sim is not None:
            cold_genes_arr = np.asarray(list(cold_start_genes))
            if len(cold_genes_arr) > 0:
                for cg in cold_genes_arr:
                    sim_row = external_gene_sim[cg].copy()
                    sim_row[cold_genes_arr] = 0.0  # connect to training nodes only
                    self.gene_sim[cg] = sim_row

        if cold_start_diseases is not None and external_disease_sim is not None:
            cold_dis_arr = np.asarray(list(cold_start_diseases))
            if len(cold_dis_arr) > 0:
                for cd in cold_dis_arr:
                    sim_row = external_disease_sim[cd].copy()
                    sim_row[cold_dis_arr] = 0.0  # connect to training nodes only
                    self.disease_sim[cd] = sim_row

        self._extract_top_k_neighbors()

    def build_from_precomputed_leaked(
        self,
        data_dir: Union[str, Path],
        canonical_genes: list,
        canonical_diseases: list,
    ):
        """
        Loads the precomputed similarity matrices from disk (which were derived from full A).
        Used exclusively for the Leakage Audit experiment.
        """
        data_dir = Path(data_dir)
        gene_file = data_dir / "gene_similarity_matrix_cosine.txt"
        disease_file = data_dir / "attribute_similarity_matrix_cosine.txt"

        # Load gene similarity
        with open(gene_file, "r") as f:
            g_cols = f.readline().strip().split("\t")[3:]
            f.readline()
            f.readline()
            rows = []
            genes = []
            for line in f:
                parts = line.strip().split("\t")
                genes.append(parts[0])
                rows.append([float(x) for x in parts[3:]])
        df_g_sim = pd.DataFrame(rows, index=genes, columns=g_cols)
        df_g_sim_aligned = df_g_sim.reindex(index=canonical_genes, columns=canonical_genes).fillna(0.0)
        self.gene_sim = df_g_sim_aligned.values.astype(np.float32)

        # Load disease similarity
        with open(disease_file, "r") as f:
            d_cols = f.readline().strip().split("\t")[3:]
            f.readline()
            f.readline()
            rows = []
            diseases = []
            for line in f:
                parts = line.strip().split("\t")
                diseases.append(parts[0])
                rows.append([float(x) for x in parts[3:]])
        df_d_sim = pd.DataFrame(rows, index=diseases, columns=d_cols)
        df_d_sim_aligned = df_d_sim.reindex(index=canonical_diseases, columns=canonical_diseases).fillna(0.0)
        self.disease_sim = df_d_sim_aligned.values.astype(np.float32)

        self._extract_top_k_neighbors()

    def _extract_top_k_neighbors(self):
        """
        Extracts top-K *distinct* neighbor indices and similarity weights for each node.

        The diagonal (self-similarity = 1.0) is excluded so the central node is not
        duplicated as its own neighbor (the central token already occupies position 0).
        """
        # --- Genes: shape (N_g, N_g) ---
        k_g = min(self.k_gene, self.num_genes - 1)
        g_sim = self.gene_sim.copy()
        np.fill_diagonal(g_sim, -np.inf)
        g_top_idx = np.argsort(-g_sim, axis=1)[:, :k_g]
        g_top_wt = np.take_along_axis(self.gene_sim, g_top_idx, axis=1)

        # --- Diseases: shape (N_d, N_d) ---
        k_d = min(self.k_disease, self.num_diseases - 1)
        d_sim = self.disease_sim.copy()
        np.fill_diagonal(d_sim, -np.inf)
        d_top_idx = np.argsort(-d_sim, axis=1)[:, :k_d]
        d_top_wt = np.take_along_axis(self.disease_sim, d_top_idx, axis=1)

        self.gene_neighbor_idx = torch.tensor(g_top_idx, dtype=torch.long)
        self.gene_neighbor_wt = torch.tensor(g_top_wt, dtype=torch.float32)
        self.disease_neighbor_idx = torch.tensor(d_top_idx, dtype=torch.long)
        self.disease_neighbor_wt = torch.tensor(d_top_wt, dtype=torch.float32)

    def to_device(self, device: torch.device):
        """Moves neighbor tensors to device."""
        if self.gene_neighbor_idx is not None:
            self.gene_neighbor_idx = self.gene_neighbor_idx.to(device)
            self.gene_neighbor_wt = self.gene_neighbor_wt.to(device)
            self.disease_neighbor_idx = self.disease_neighbor_idx.to(device)
            self.disease_neighbor_wt = self.disease_neighbor_wt.to(device)
        return self

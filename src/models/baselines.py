"""
Simpler biological and machine learning baselines for comparison against TransGDA:
1. DegreeBaseline: Degree / Marginal frequency prior
2. NeighborhoodCFBaseline: Guilt-By-Association Collaborative Filtering
3. MatrixFactorizationBaseline: Low-rank Truncated SVD Matrix Factorization
4. StaticMLPBaseline: Standard embedding MLP without cross-attention or graph neighborhood
5. TransGDAAblationNoCrossAttn: TransGDA ablation without bilateral cross-attention
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphTransformerEncoderLayer


class DegreeBaseline:
    """
    Biological Degree Prior Baseline:
    Predicts association probability proportional to gene degree * disease frequency.
    """

    def __init__(self):
        self.gene_degrees = None
        self.disease_degrees = None
        self.total_edges = 1.0
        self.mean_strength = 0.0

    def fit(self, train_matrix_binary: np.ndarray, train_matrix_strength: np.ndarray):
        self.gene_degrees = train_matrix_binary.sum(axis=1)  # (N_g,)
        self.disease_degrees = train_matrix_binary.sum(axis=0)  # (N_d,)
        self.total_edges = max(train_matrix_binary.sum(), 1.0)
        pos_mask = train_matrix_binary > 0
        if pos_mask.sum() > 0:
            self.mean_strength = float(train_matrix_strength[pos_mask].mean())
        else:
            self.mean_strength = 0.0

    def predict(self, gene_idx: np.ndarray, disease_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g_deg = self.gene_degrees[gene_idx]
        d_deg = self.disease_degrees[disease_idx]
        # Prior score normalized to [0, 1]
        prior_score = (g_deg * d_deg) / (self.total_edges + 1e-8)
        prior_prob = 1.0 - np.exp(-prior_score)  # Poisson probability of >= 1 edge
        prior_strength = np.full_like(prior_prob, self.mean_strength)
        return prior_prob, prior_strength


class NeighborhoodCFBaseline:
    """
    Guilt-By-Association Collaborative Filtering Baseline:
    Scores pair (g, d) by aggregating known associations of g's nearest neighbors.
    """

    def __init__(self, k_neighbors: int = 20):
        self.k_neighbors = k_neighbors
        self.train_matrix_binary = None
        self.train_matrix_strength = None
        self.gene_sim = None
        self.top_nbr_idx = None
        self.top_nbr_wt = None

    def fit(
        self,
        train_matrix_binary: np.ndarray,
        train_matrix_strength: np.ndarray,
        gene_similarity: np.ndarray,
    ):
        self.train_matrix_binary = train_matrix_binary
        self.train_matrix_strength = train_matrix_strength
        self.gene_sim = gene_similarity

        # Precompute top-k neighbors for each gene
        k = min(self.k_neighbors, gene_similarity.shape[0])
        self.top_nbr_idx = np.argsort(-gene_similarity, axis=1)[:, :k]
        self.top_nbr_wt = np.take_along_axis(gene_similarity, self.top_nbr_idx, axis=1)

    def predict(self, gene_idx: np.ndarray, disease_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scores = []
        strengths = []
        for g, d in zip(gene_idx, disease_idx):
            nbrs = self.top_nbr_idx[g]
            wts = self.top_nbr_wt[g]

            # Neighbor associations to disease d
            nbr_bin = self.train_matrix_binary[nbrs, d]
            nbr_str = self.train_matrix_strength[nbrs, d]

            sum_wt = wts.sum() + 1e-8
            cf_score = np.dot(wts, nbr_bin) / sum_wt
            cf_strength = np.dot(wts, nbr_str) / (np.dot(wts, (nbr_bin > 0).astype(float)) + 1e-8)

            scores.append(cf_score)
            strengths.append(cf_strength if not np.isnan(cf_strength) else 0.0)

        probs = np.clip(np.array(scores, dtype=np.float32), 0.0, 1.0)
        strengths = np.array(strengths, dtype=np.float32)
        return probs, strengths


class MatrixFactorizationBaseline:
    """
    Low-rank Truncated SVD Matrix Factorization Baseline on the bipartite association matrix.
    """

    def __init__(self, n_components: int = 32, seed: int = 42):
        self.n_components = n_components
        self.seed = seed
        self.svd_bin = TruncatedSVD(n_components=n_components, random_state=seed)
        self.svd_str = TruncatedSVD(n_components=n_components, random_state=seed)
        self.U_bin = None
        self.V_bin = None
        self.U_str = None
        self.V_str = None

    def fit(self, train_matrix_binary: np.ndarray, train_matrix_strength: np.ndarray):
        k = min(self.n_components, train_matrix_binary.shape[1] - 1)
        self.svd_bin.set_params(n_components=k)
        self.svd_str.set_params(n_components=k)

        self.U_bin = self.svd_bin.fit_transform(train_matrix_binary)  # (N_g, k)
        self.V_bin = self.svd_bin.components_.T                     # (N_d, k)

        self.U_str = self.svd_str.fit_transform(train_matrix_strength)
        self.V_str = self.svd_str.components_.T

    def predict(self, gene_idx: np.ndarray, disease_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u_b = self.U_bin[gene_idx]
        v_b = self.V_bin[disease_idx]
        dot_b = np.sum(u_b * v_b, axis=1)
        # Sigmoid mapping
        probs = 1.0 / (1.0 + np.exp(-np.clip(dot_b, -15.0, 15.0)))

        u_s = self.U_str[gene_idx]
        v_s = self.V_str[disease_idx]
        strengths = np.maximum(np.sum(u_s * v_s, axis=1), 0.0)
        return probs.astype(np.float32), strengths.astype(np.float32)


class StaticMLPBaseline(nn.Module):
    """
    Static Embedding MLP Baseline:
    Takes static gene and disease embeddings directly into MLP without cross-attention
    or graph neighborhood context.
    """

    def __init__(
        self,
        num_genes: int,
        num_diseases: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gene_embed = nn.Embedding(num_genes, embed_dim)
        self.disease_embed = nn.Embedding(num_diseases, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.exist_head = nn.Linear(hidden_dim, 1)
        self.strength_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

        nn.init.xavier_uniform_(self.gene_embed.weight)
        nn.init.xavier_uniform_(self.disease_embed.weight)

    def forward(
        self,
        gene_idx: torch.Tensor,
        disease_idx: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        e_g = self.gene_embed(gene_idx)
        e_d = self.disease_embed(disease_idx)
        cat = torch.cat([e_g, e_d], dim=-1)
        h = self.mlp(cat)

        logits = self.exist_head(h).squeeze(-1)
        probs = torch.sigmoid(logits)
        strengths = self.strength_head(h).squeeze(-1)

        return {
            "logits": logits,
            "probs": probs,
            "strengths": strengths,
        }

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        strengths: torch.Tensor,
        lambda_strength: float = 1.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bce_loss = F.binary_cross_entropy_with_logits(predictions["logits"], labels)
        pos_mask = labels > 0.5
        if pos_mask.sum() > 0:
            str_loss = F.smooth_l1_loss(predictions["strengths"][pos_mask], strengths[pos_mask])
        else:
            str_loss = torch.tensor(0.0, device=labels.device)
        return bce_loss + lambda_strength * str_loss, bce_loss, str_loss


class TransGDAAblationNoCrossAttn(nn.Module):
    """
    Ablation: TransGDA without Bilateral Cross-Attention.
    Contextualizes nodes with neighborhood Graph Transformer, but compares embeddings
    simply via concatenation and MLP.
    """

    def __init__(
        self,
        num_genes: int,
        num_diseases: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        node_feat_dim: int = 0,
        gene_feat_dim: Optional[int] = None,
        disease_feat_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.num_diseases = num_diseases
        self.embed_dim = embed_dim
        self.gene_feat_dim = gene_feat_dim if gene_feat_dim is not None else node_feat_dim
        self.disease_feat_dim = disease_feat_dim if disease_feat_dim is not None else node_feat_dim

        self.gene_embed = nn.Embedding(num_genes, embed_dim)
        self.disease_embed = nn.Embedding(num_diseases, embed_dim)

        if self.gene_feat_dim > 0:
            self.gene_feat_proj = nn.Linear(self.gene_feat_dim, embed_dim)
        else:
            self.gene_feat_proj = None

        if self.disease_feat_dim > 0:
            self.disease_feat_proj = nn.Linear(self.disease_feat_dim, embed_dim)
        else:
            self.disease_feat_proj = None

        self.register_buffer("gene_feat", None, persistent=False)
        self.register_buffer("disease_feat", None, persistent=False)
        self.register_buffer("cold_start_gene_mask", None, persistent=False)
        self.register_buffer("cold_start_disease_mask", None, persistent=False)

        self.gene_encoder = GraphTransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout)
        self.disease_encoder = GraphTransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.exist_head = nn.Linear(hidden_dim, 1)
        self.strength_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def set_node_features(
        self,
        gene_feat: Optional[torch.Tensor] = None,
        disease_feat: Optional[torch.Tensor] = None,
    ) -> None:
        self.gene_feat = gene_feat
        self.disease_feat = disease_feat

    def set_cold_start_nodes(
        self,
        cold_start_genes: Optional[Union[set, list, np.ndarray, torch.Tensor]] = None,
        cold_start_diseases: Optional[Union[set, list, np.ndarray, torch.Tensor]] = None,
    ) -> None:
        if cold_start_genes is not None:
            mask = torch.zeros(self.num_genes, dtype=torch.bool, device=self.gene_embed.weight.device)
            mask[list(cold_start_genes)] = True
            self.cold_start_gene_mask = mask
        else:
            self.cold_start_gene_mask = None

        if cold_start_diseases is not None:
            mask = torch.zeros(self.num_diseases, dtype=torch.bool, device=self.disease_embed.weight.device)
            mask[list(cold_start_diseases)] = True
            self.cold_start_disease_mask = mask
        else:
            self.cold_start_disease_mask = None

    def _embed_gene(self, idx: torch.Tensor) -> torch.Tensor:
        if self.cold_start_gene_mask is not None:
            is_cold = self.cold_start_gene_mask[idx].unsqueeze(-1)
            id_emb = torch.where(is_cold, torch.zeros_like(self.gene_embed(idx)), self.gene_embed(idx))
        else:
            id_emb = self.gene_embed(idx)
        if self.gene_feat_proj is not None and self.gene_feat is not None:
            id_emb = id_emb + self.gene_feat_proj(self.gene_feat[idx])
        return id_emb

    def _embed_disease(self, idx: torch.Tensor) -> torch.Tensor:
        if self.cold_start_disease_mask is not None:
            is_cold = self.cold_start_disease_mask[idx].unsqueeze(-1)
            id_emb = torch.where(is_cold, torch.zeros_like(self.disease_embed(idx)), self.disease_embed(idx))
        else:
            id_emb = self.disease_embed(idx)
        if self.disease_feat_proj is not None and self.disease_feat is not None:
            id_emb = id_emb + self.disease_feat_proj(self.disease_feat[idx])
        return id_emb

    def forward(
        self,
        gene_idx: torch.Tensor,
        disease_idx: torch.Tensor,
        gene_nbr_idx: torch.Tensor,
        gene_nbr_wt: torch.Tensor,
        disease_nbr_idx: torch.Tensor,
        disease_nbr_wt: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        e_g = self._embed_gene(gene_idx)
        e_g_nbrs = self._embed_gene(gene_nbr_idx)
        e_d = self._embed_disease(disease_idx)
        e_d_nbrs = self._embed_disease(disease_nbr_idx)

        # Encoder returns the full contextualized sequence; take the central node.
        seq_g, _ = self.gene_encoder(e_g, e_g_nbrs, gene_nbr_wt)
        seq_d, _ = self.disease_encoder(e_d, e_d_nbrs, disease_nbr_wt)
        z_g, z_d = seq_g[:, 0, :], seq_d[:, 0, :]

        cat = torch.cat([z_g, z_d], dim=-1)
        h = self.mlp(cat)

        logits = self.exist_head(h).squeeze(-1)
        probs = torch.sigmoid(logits)
        strengths = self.strength_head(h).squeeze(-1)

        return {"logits": logits, "probs": probs, "strengths": strengths}

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        strengths: torch.Tensor,
        lambda_strength: float = 1.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bce = F.binary_cross_entropy_with_logits(predictions["logits"], labels)
        pos_mask = labels > 0.5
        if pos_mask.sum() > 0:
            str_l = F.smooth_l1_loss(predictions["strengths"][pos_mask], strengths[pos_mask])
        else:
            str_l = torch.tensor(0.0, device=labels.device)
        return bce + lambda_strength * str_l, bce, str_l

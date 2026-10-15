"""
TransGDA: Full Dual Graph-Transformer with Bilateral Cross-Attention and Multi-Task Heads.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    GraphTransformerEncoderLayer,
    BilateralCrossAttentionLayer,
    InteractiveFusionLayer,
    attention_entropy,
)


class TransGDA(nn.Module):
    """
    TransGDA Architecture:
    1. Gene & Disease Graph-Transformer Encoders (Neighborhood Self-Attention)
    2. Bilateral Cross-Attention Interaction Module
    3. Interactive Fusion Layer (Bilinear + Multi-Modal Combination)
    4. Multi-Task Heads: Existence Classifier (BCE) & Strength Regressor (SmoothL1)
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
        self.node_feat_dim = node_feat_dim

        # Initial learnable embeddings
        self.gene_embed = nn.Embedding(num_genes, embed_dim)
        self.disease_embed = nn.Embedding(num_diseases, embed_dim)

        # Optional inductive feature path: when feature dim > 0, each node's
        # representation is augmented with a projection of an external feature
        # vector (e.g. a similarity profile). This lets cold-start nodes that were
        # never seen during training receive a meaningful representation.
        if self.gene_feat_dim > 0:
            self.gene_feat_proj = nn.Linear(self.gene_feat_dim, embed_dim)
        else:
            self.gene_feat_proj = None

        if self.disease_feat_dim > 0:
            self.disease_feat_proj = nn.Linear(self.disease_feat_dim, embed_dim)
        else:
            self.disease_feat_proj = None

        # Feature buffers (populated via set_node_features); non-persistent so they
        # are not written into checkpoints.
        self.register_buffer("gene_feat", None, persistent=False)
        self.register_buffer("disease_feat", None, persistent=False)
        self.register_buffer("cold_start_gene_mask", None, persistent=False)
        self.register_buffer("cold_start_disease_mask", None, persistent=False)

        # Graph-Transformer Encoders
        self.gene_encoder = GraphTransformerEncoderLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.disease_encoder = GraphTransformerEncoderLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        # Bilateral Cross-Attention
        self.cross_attn = BilateralCrossAttentionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Interactive Fusion
        self.fusion = InteractiveFusionLayer(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Multi-Task Heads
        self.exist_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.strength_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Enforces non-negative association strength
        )

        # Calibrated temperature scaling parameter (default: 1.0)
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))

        self._init_weights()

    def set_temperature(self, temp: float):
        """Sets temperature scaling factor for logit calibration."""
        self.temperature.data.fill_(max(0.1, float(temp)))

    def _init_weights(self):
        nn.init.xavier_uniform_(self.gene_embed.weight)
        nn.init.xavier_uniform_(self.disease_embed.weight)
        if self.gene_feat_proj is not None:
            nn.init.xavier_uniform_(self.gene_feat_proj.weight)
            nn.init.zeros_(self.gene_feat_proj.bias)
        if self.disease_feat_proj is not None:
            nn.init.xavier_uniform_(self.disease_feat_proj.weight)
            nn.init.zeros_(self.disease_feat_proj.bias)

    def set_node_features(
        self,
        gene_feat: Optional[torch.Tensor] = None,
        disease_feat: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Registers external per-node feature matrices for the inductive path.
        """
        if gene_feat is not None:
            gene_feat = torch.as_tensor(gene_feat, dtype=torch.float32)
            if self.gene_feat_dim > 0 and gene_feat.shape != (self.num_genes, self.gene_feat_dim):
                raise ValueError(
                    f"gene_feat must be ({self.num_genes}, {self.gene_feat_dim}), got {tuple(gene_feat.shape)}"
                )
        if disease_feat is not None:
            disease_feat = torch.as_tensor(disease_feat, dtype=torch.float32)
            if self.disease_feat_dim > 0 and disease_feat.shape != (self.num_diseases, self.disease_feat_dim):
                raise ValueError(
                    f"disease_feat must be ({self.num_diseases}, {self.disease_feat_dim}), got {tuple(disease_feat.shape)}"
                )
        self.gene_feat = gene_feat
        self.disease_feat = disease_feat

    def set_cold_start_nodes(
        self,
        cold_start_genes: Optional[Union[set, list, np.ndarray, torch.Tensor]] = None,
        cold_start_diseases: Optional[Union[set, list, np.ndarray, torch.Tensor]] = None,
    ) -> None:
        """
        Flags nodes as cold-start so their untrained ID embeddings are zeroed out,
        allowing feature projection to purely represent them without random noise.
        """
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
        """Gene embedding = learnable id embedding (for seen nodes) + projected external features."""
        if self.cold_start_gene_mask is not None:
            is_cold = self.cold_start_gene_mask[idx].unsqueeze(-1)
            id_emb = torch.where(is_cold, torch.zeros_like(self.gene_embed(idx)), self.gene_embed(idx))
        else:
            id_emb = self.gene_embed(idx)
        if self.gene_feat_proj is not None and self.gene_feat is not None:
            id_emb = id_emb + self.gene_feat_proj(self.gene_feat[idx])
        return id_emb

    def _embed_disease(self, idx: torch.Tensor) -> torch.Tensor:
        """Disease embedding = learnable id embedding (for seen nodes) + projected external features."""
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
        gene_idx: torch.Tensor,               # (B,)
        disease_idx: torch.Tensor,            # (B,)
        gene_nbr_idx: torch.Tensor,           # (B, K_g)
        gene_nbr_wt: torch.Tensor,            # (B, K_g)
        disease_nbr_idx: torch.Tensor,        # (B, K_d)
        disease_nbr_wt: torch.Tensor,         # (B, K_d)
        return_attention: bool = False,
        gene_mask: Optional[torch.Tensor] = None,       # (B, 1+K_g) True if masked
        disease_mask: Optional[torch.Tensor] = None,    # (B, 1+K_d) True if masked
        e_g: Optional[torch.Tensor] = None,             # (B, D) optional precomputed/interpolated
        e_g_nbrs: Optional[torch.Tensor] = None,        # (B, K_g, D)
        e_d: Optional[torch.Tensor] = None,             # (B, D)
        e_d_nbrs: Optional[torch.Tensor] = None,        # (B, K_d, D)
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a batch of gene-disease pairs with their functional graph neighborhoods.
        """
        # Lookup embeddings (feature-aware for the inductive path) if not explicitly provided
        if e_g is None:
            e_g = self._embed_gene(gene_idx)                 # (B, D)
        if e_g_nbrs is None:
            e_g_nbrs = self._embed_gene(gene_nbr_idx)        # (B, K_g, D)

        if e_d is None:
            e_d = self._embed_disease(disease_idx)           # (B, D)
        if e_d_nbrs is None:
            e_d_nbrs = self._embed_disease(disease_nbr_idx)  # (B, K_d, D)

        # 1. Neighborhood Graph Transformer Encoding
        # The encoder returns the FULL contextualized sequence: position 0 is the
        # central node, positions 1..K are the contextualized neighbors.
        seq_g, attn_g_self = self.gene_encoder(e_g, e_g_nbrs, gene_nbr_wt, mask=gene_mask)
        seq_d, attn_d_self = self.disease_encoder(e_d, e_d_nbrs, disease_nbr_wt, mask=disease_mask)

        z_g, z_g_nbrs = seq_g[:, 0, :], seq_g[:, 1:, :]
        z_d, z_d_nbrs = seq_d[:, 0, :], seq_d[:, 1:, :]

        # 2. Bilateral Cross-Attention Interaction
        h_g2d, h_d2g, attn_g2d, attn_d2g = self.cross_attn(
            z_g, z_d, z_g_nbrs, z_d_nbrs,
            gene_mask=gene_mask,
            disease_mask=disease_mask,
        )

        # 3. Interactive Fusion
        fused = self.fusion(z_g, z_d, h_g2d, h_d2g)

        # 4. Multi-Task Predictions
        exist_logits = self.exist_head(fused).squeeze(-1)       # (B,)
        # Calibrated existence probability scaled by temperature
        cal_temp = torch.clamp(self.temperature, min=0.1)
        exist_probs = torch.sigmoid(exist_logits / cal_temp)
        strength_preds = self.strength_head(fused).squeeze(-1)  # (B,)
        effective_strengths = exist_probs * strength_preds

        out = {
            "logits": exist_logits,
            "probs": exist_probs,
            "strengths": strength_preds,
            "effective_strengths": effective_strengths,
            # Attention is always returned (cheap) so the loss can apply an optional
            # entropy regularizer and downstream tools can inspect it.
            "attn_gene_self": attn_g_self,
            "attn_disease_self": attn_d_self,
            "attn_g2d": attn_g2d,
            "attn_d2g": attn_d2g,
        }

        return out

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        strengths: torch.Tensor,
        lambda_strength: float = 1.0,
        label_smoothing: float = 0.05,
        lambda_neg_strength: float = 0.2,
        lambda_attn_entropy: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes joint Multi-Task Loss with Calibration & Associativity Binding:
        1. BCE with label smoothing to prevent logit explosion and overconfident saturation.
        2. Positive strength regression (SmoothL1 on true positives).
        3. Negative strength regularization (SmoothL1 towards 0 on negatives) to preserve associativity.
        """
        # Existence Classification Loss with Label Smoothing
        if label_smoothing > 0.0:
            smoothed_targets = labels * (1.0 - 2.0 * label_smoothing) + label_smoothing
            bce_loss = F.binary_cross_entropy_with_logits(predictions["logits"], smoothed_targets)
        else:
            bce_loss = F.binary_cross_entropy_with_logits(predictions["logits"], labels)

        # Strength Regression Loss
        pos_mask = (labels > 0.5)
        neg_mask = ~pos_mask

        if pos_mask.sum() > 0:
            pred_s_pos = predictions["strengths"][pos_mask]
            true_s_pos = strengths[pos_mask]
            pos_strength_loss = F.smooth_l1_loss(pred_s_pos, true_s_pos)
        else:
            pos_strength_loss = torch.tensor(0.0, device=labels.device)

        if neg_mask.sum() > 0 and lambda_neg_strength > 0.0:
            pred_s_neg = predictions["strengths"][neg_mask]
            neg_strength_loss = F.smooth_l1_loss(pred_s_neg, torch.zeros_like(pred_s_neg))
        else:
            neg_strength_loss = torch.tensor(0.0, device=labels.device)

        strength_loss = pos_strength_loss + lambda_neg_strength * neg_strength_loss
        total_loss = bce_loss + lambda_strength * strength_loss

        # Optional attention-entropy regularizer: mildly discourages uniform
        # (uninformative) attention distributions. Disabled by default.
        if lambda_attn_entropy > 0.0 and "attn_g2d" in predictions:
            ent = attention_entropy(predictions["attn_g2d"]) + attention_entropy(predictions["attn_d2g"])
            total_loss = total_loss + lambda_attn_entropy * ent

        return total_loss, bce_loss, strength_loss

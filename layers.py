"""
Transformer neural network layers:
- GraphTransformerEncoderLayer: Neighborhood Multi-Head Self-Attention with Edge Bias
- BilateralCrossAttentionLayer: Asymmetric Gene-to-Disease and Disease-to-Gene Cross-Attention
- InteractiveFusionLayer: Multi-Modal Bilinear & Symmetric-Asymmetric Feature Interaction
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention_entropy(attn: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Mean Shannon entropy of attention distributions.

    Args:
        attn: Attention weights of shape (..., K) that sum to 1 along the last dim.
    Returns:
        Scalar tensor: the mean entropy across all distributions.
    """
    p = attn.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1).mean()


class GraphTransformerEncoderLayer(nn.Module):
    """
    Graph Transformer Encoder Layer that aggregates local neighborhood context
    using Multi-Head Self-Attention with continuous edge similarity bias.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        ffn_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.edge_bias_proj = nn.Linear(1, num_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        central_nodes: torch.Tensor,     # Shape: (B, embed_dim)
        neighbors: torch.Tensor,         # Shape: (B, K, embed_dim)
        neighbor_weights: torch.Tensor,  # Shape: (B, K)
        mask: Optional[torch.Tensor] = None, # Shape: (B, 1+K) True if masked
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            contextualized_seq: Shape (B, 1+K, embed_dim) — position 0 is the central node.
            attn_weights: Shape (B, num_heads, 1+K, 1+K)
        """
        B, K, D = neighbors.shape
        # Sequence: [central_node, neighbor_0, ..., neighbor_{K-1}]
        seq = torch.cat([central_nodes.unsqueeze(1), neighbors], dim=1)  # (B, 1+K, D)

        # Build edge bias: central to self is 1.0, central to neighbors is weights
        edge_w = torch.cat([torch.ones(B, 1, device=neighbors.device), neighbor_weights], dim=1)  # (B, 1+K)
        # Pairwise bias matrix: (B, 1+K, 1+K, 1) -> (B, num_heads, 1+K, 1+K)
        # Using broadcasted multiplication of edge weights
        pair_edge = torch.bmm(edge_w.unsqueeze(2), edge_w.unsqueeze(1)).unsqueeze(-1)  # (B, 1+K, 1+K, 1)
        edge_bias = self.edge_bias_proj(pair_edge).permute(0, 3, 1, 2)  # (B, num_heads, 1+K, 1+K)

        # Multi-Head Attention
        q = self.q_proj(seq).view(B, 1 + K, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(seq).view(B, 1 + K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(seq).view(B, 1 + K, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + edge_bias
        if mask is not None:
            # Broadcast mask (B, 1+K) to (B, 1, 1, 1+K) for key masking
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(self.dropout(attn_weights), v)  # (B, num_heads, 1+K, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1 + K, D)
        attn_out = self.out_proj(attn_out)

        # Residual + Norm
        seq = self.norm1(seq + self.dropout(attn_out))
        seq = self.norm2(seq + self.ffn(seq))

        if mask is not None:
            # Zero out representations of knocked-out tokens for counterfactual knockout interventions
            seq = seq.masked_fill(mask.unsqueeze(-1), 0.0)

        # Return the full contextualized sequence so callers can use both the
        # central node (position 0) and the contextualized neighbors (positions 1..K).
        return seq, attn_weights


class BilateralCrossAttentionLayer(nn.Module):
    """
    Asymmetric Bilateral Cross-Attention:
    - Gene queries disease neighborhood context to find disease-specific drivers
    - Disease queries gene neighborhood context to find gene-specific drivers
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Gene queries disease context
        self.g2d_q = nn.Linear(embed_dim, embed_dim)
        self.g2d_k = nn.Linear(embed_dim, embed_dim)
        self.g2d_v = nn.Linear(embed_dim, embed_dim)
        self.g2d_out = nn.Linear(embed_dim, embed_dim)

        # Disease queries gene context
        self.d2g_q = nn.Linear(embed_dim, embed_dim)
        self.d2g_k = nn.Linear(embed_dim, embed_dim)
        self.d2g_v = nn.Linear(embed_dim, embed_dim)
        self.d2g_out = nn.Linear(embed_dim, embed_dim)

        self.norm_g = nn.LayerNorm(embed_dim)
        self.norm_d = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        gene_ctx: torch.Tensor,        # (B, embed_dim)
        disease_ctx: torch.Tensor,     # (B, embed_dim)
        gene_neighbors: torch.Tensor,  # (B, K_g, embed_dim)
        disease_neighbors: torch.Tensor, # (B, K_d, embed_dim)
        gene_mask: Optional[torch.Tensor] = None,    # (B, 1+K_g) True if masked
        disease_mask: Optional[torch.Tensor] = None, # (B, 1+K_d) True if masked
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_g2d: Gene cross-attended on disease context (B, embed_dim)
            h_d2g: Disease cross-attended on gene context (B, embed_dim)
            attn_g2d: Attention weights (B, num_heads, 1, 1+K_d)
            attn_d2g: Attention weights (B, num_heads, 1, 1+K_g)
        """
        B, D = gene_ctx.shape
        # Disease key-value memory: disease central + disease neighbors
        d_mem = torch.cat([disease_ctx.unsqueeze(1), disease_neighbors], dim=1)  # (B, 1+K_d, D)
        # Gene key-value memory: gene central + gene neighbors
        g_mem = torch.cat([gene_ctx.unsqueeze(1), gene_neighbors], dim=1)  # (B, 1+K_g, D)

        scale = math.sqrt(self.head_dim)

        # 1. Gene queries disease
        q_g = self.g2d_q(gene_ctx.unsqueeze(1)).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k_d = self.g2d_k(d_mem).view(B, d_mem.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v_d = self.g2d_v(d_mem).view(B, d_mem.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        scores_g2d = torch.matmul(q_g, k_d.transpose(-2, -1)) / scale
        if disease_mask is not None:
            # Broadcast disease_mask (B, 1+K_d) to (B, 1, 1, 1+K_d)
            scores_g2d = scores_g2d.masked_fill(disease_mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn_g2d = F.softmax(scores_g2d, dim=-1)
        out_g2d = torch.matmul(self.dropout(attn_g2d), v_d).transpose(1, 2).contiguous().view(B, 1, D).squeeze(1)
        h_g2d = self.norm_g(gene_ctx + self.dropout(self.g2d_out(out_g2d)))

        # 2. Disease queries gene
        q_d = self.d2g_q(disease_ctx.unsqueeze(1)).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k_g = self.d2g_k(g_mem).view(B, g_mem.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v_g = self.d2g_v(g_mem).view(B, g_mem.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        scores_d2g = torch.matmul(q_d, k_g.transpose(-2, -1)) / scale
        if gene_mask is not None:
            # Broadcast gene_mask (B, 1+K_g) to (B, 1, 1, 1+K_g)
            scores_d2g = scores_d2g.masked_fill(gene_mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn_d2g = F.softmax(scores_d2g, dim=-1)
        out_d2g = torch.matmul(self.dropout(attn_d2g), v_g).transpose(1, 2).contiguous().view(B, 1, D).squeeze(1)
        h_d2g = self.norm_d(disease_ctx + self.dropout(self.d2g_out(out_d2g)))

        if gene_mask is not None:
            # If central gene is knocked out (position 0), ablate h_g2d
            h_g2d = h_g2d.masked_fill(gene_mask[:, 0].unsqueeze(-1), 0.0)

        if disease_mask is not None:
            # If central disease is knocked out (position 0), ablate h_d2g
            h_d2g = h_d2g.masked_fill(disease_mask[:, 0].unsqueeze(-1), 0.0)

        return h_g2d, h_d2g, attn_g2d, attn_d2g


class InteractiveFusionLayer(nn.Module):
    """
    Fuses multiple interactive features between gene and disease representations:
    [z_g, z_d, h_g2d, h_d2g, z_g * z_d, |z_g - z_d|, bilinear(z_g, z_d)]
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bilinear = nn.Bilinear(embed_dim, embed_dim, embed_dim)
        in_dim = embed_dim * 7
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        z_g: torch.Tensor,
        z_d: torch.Tensor,
        h_g2d: torch.Tensor,
        h_d2g: torch.Tensor,
    ) -> torch.Tensor:
        hadamard = z_g * z_d
        diff = torch.abs(z_g - z_d)
        bilinear = self.bilinear(z_g, z_d)
        fused = torch.cat([z_g, z_d, h_g2d, h_d2g, hadamard, diff, bilinear], dim=-1)
        h = F.gelu(self.fc1(fused))
        h = self.norm(h)
        h = self.dropout(h)
        h = F.gelu(self.fc2(h))
        return h

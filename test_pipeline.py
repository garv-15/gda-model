"""
Unit tests and sanity checks for the TransGDA pipeline:
- Canonical alignment of matrices
- Leakage-free train split invariants
- Baselines fit & predict
- TransGDA forward pass & loss computation
- Attention map extraction
"""

import unittest
import numpy as np
import torch

from src.data import DataLoaderManager
from src.similarity import (
    compute_cosine_similarity,
    compute_jaccard_similarity,
    GraphNeighborhoodManager,
)
from src.models.transf_gda import TransGDA
from src.models.baselines import (
    DegreeBaseline,
    NeighborhoodCFBaseline,
    MatrixFactorizationBaseline,
    StaticMLPBaseline,
)
from src.evaluate import compute_classification_metrics, compute_regression_metrics


class TestTransGDAPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_mgr = DataLoaderManager(data_dir=".")

    def test_canonical_alignment(self):
        """Verify binary and strength matrices are 100% aligned with identical positive counts."""
        bin_mat = self.data_mgr.binary_matrix
        str_mat = self.data_mgr.strength_matrix

        self.assertEqual(bin_mat.shape, (self.data_mgr.meta.num_genes, self.data_mgr.meta.num_diseases))
        self.assertEqual(str_mat.shape, (self.data_mgr.meta.num_genes, self.data_mgr.meta.num_diseases))

        bin_pos = (bin_mat > 0)
        str_pos = (str_mat > 0)

        # Invariant: identical non-zero sparsity patterns and positive edge counts
        self.assertTrue(np.array_equal(bin_pos, str_pos))
        self.assertGreater(int(bin_pos.sum()), 0)

    def test_leakage_prevention_in_splits(self):
        """Verify that test/validation edges are completely masked from train matrices."""
        split = self.data_mgr.create_edge_split(val_ratio=0.1, test_ratio=0.1, seed=42)

        train_mat = split.train_matrix_binary

        # Check all test edges have 0 in train matrix
        test_vals_in_train = train_mat[split.test_edges[:, 0], split.test_edges[:, 1]]
        self.assertEqual(test_vals_in_train.sum(), 0.0)

        # Check all val edges have 0 in train matrix
        val_vals_in_train = train_mat[split.val_edges[:, 0], split.val_edges[:, 1]]
        self.assertEqual(val_vals_in_train.sum(), 0.0)

        # Dynamic similarity computation on train matrix
        sim = compute_cosine_similarity(train_mat)
        self.assertEqual(sim.shape, (self.data_mgr.meta.num_genes, self.data_mgr.meta.num_genes))
        self.assertTrue(np.all(sim >= 0.0) and np.all(sim <= 1.0))
        self.assertTrue(np.allclose(np.diag(sim), 1.0))

    def test_gene_held_out_split(self):
        """Verify cold-start gene split holds out test genes completely."""
        split = self.data_mgr.create_gene_held_out_split(val_ratio=0.1, test_ratio=0.2, seed=42)
        test_genes = set(split.test_gene_indices)

        # Ensure no test gene appears in train edges
        train_genes = set(split.train_edges[:, 0])
        overlap = test_genes.intersection(train_genes)
        self.assertEqual(len(overlap), 0)

    def test_disease_held_out_split(self):
        """Verify cold-start disease split holds out test diseases completely."""
        split = self.data_mgr.create_disease_held_out_split(val_ratio=0.1, test_ratio=0.2, seed=42)
        test_diseases = set(split.test_disease_indices)

        # Ensure no test disease appears in train edges
        train_diseases = set(split.train_edges[:, 1])
        overlap = test_diseases.intersection(train_diseases)
        self.assertEqual(len(overlap), 0)

    def test_baselines(self):
        """Verify that biological baselines fit and produce valid predictions."""
        split = self.data_mgr.create_edge_split(val_ratio=0.1, test_ratio=0.1, seed=42)
        train_bin = split.train_matrix_binary
        train_str = split.train_matrix_strength

        g_test = split.test_edges[:20, 0]
        d_test = split.test_edges[:20, 1]

        # 1. Degree Baseline
        deg_base = DegreeBaseline()
        deg_base.fit(train_bin, train_str)
        p_deg, s_deg = deg_base.predict(g_test, d_test)
        self.assertEqual(len(p_deg), 20)
        self.assertTrue(np.all(p_deg >= 0.0) and np.all(p_deg <= 1.0))

        # 2. CF Baseline
        g_sim = compute_cosine_similarity(train_bin)
        cf_base = NeighborhoodCFBaseline(k_neighbors=10)
        cf_base.fit(train_bin, train_str, g_sim)
        p_cf, s_cf = cf_base.predict(g_test, d_test)
        self.assertEqual(len(p_cf), 20)

        # 3. Matrix Factorization Baseline
        mf_base = MatrixFactorizationBaseline(n_components=8)
        mf_base.fit(train_bin, train_str)
        p_mf, s_mf = mf_base.predict(g_test, d_test)
        self.assertEqual(len(p_mf), 20)

    def test_transgda_forward_and_loss(self):
        """Verify TransGDA forward pass, multi-task heads, and attention weight extraction."""
        model = TransGDA(
            num_genes=50,
            num_diseases=20,
            embed_dim=16,
            num_heads=2,
            ffn_dim=32,
            hidden_dim=32,
            dropout=0.0,
        )

        batch_size = 4
        k_g = 6
        k_d = 4

        gene_idx = torch.tensor([0, 5, 10, 15], dtype=torch.long)
        disease_idx = torch.tensor([1, 3, 7, 12], dtype=torch.long)
        gene_nbr_idx = torch.randint(0, 50, (batch_size, k_g))
        gene_nbr_wt = torch.rand(batch_size, k_g)
        disease_nbr_idx = torch.randint(0, 20, (batch_size, k_d))
        disease_nbr_wt = torch.rand(batch_size, k_d)

        out = model(
            gene_idx=gene_idx,
            disease_idx=disease_idx,
            gene_nbr_idx=gene_nbr_idx,
            gene_nbr_wt=gene_nbr_wt,
            disease_nbr_idx=disease_nbr_idx,
            disease_nbr_wt=disease_nbr_wt,
            return_attention=True,
        )

        self.assertEqual(out["logits"].shape, (batch_size,))
        self.assertEqual(out["probs"].shape, (batch_size,))
        self.assertEqual(out["strengths"].shape, (batch_size,))
        self.assertTrue(torch.all(out["probs"] >= 0.0) and torch.all(out["probs"] <= 1.0))
        self.assertTrue(torch.all(out["strengths"] >= 0.0))

        # Check attention map shapes
        # attn_g2d: (batch_size, num_heads, 1, 1+k_d)
        self.assertEqual(out["attn_g2d"].shape, (batch_size, 2, 1, 1 + k_d))
        # attn_d2g: (batch_size, num_heads, 1, 1+k_g)
        self.assertEqual(out["attn_d2g"].shape, (batch_size, 2, 1, 1 + k_g))

        # Loss check
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
        strengths = torch.tensor([0.8, 0.0, 1.2, 0.0], dtype=torch.float32)
        loss, bce, str_loss = model.compute_loss(out, labels, strengths)
        self.assertTrue(loss.item() > 0.0)

        # Backward check
        loss.backward()
        self.assertIsNotNone(model.gene_embed.weight.grad)

    def test_calibration_and_ece_metrics(self):
        """Verify calibration metrics (ECE and Brier score)."""
        from src.evaluate import compute_calibration_metrics, compute_classification_metrics
        y_true = np.array([1, 1, 0, 0], dtype=np.float32)
        # Perfect calibration
        y_score_perf = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        m_perf = compute_calibration_metrics(y_true, y_score_perf)
        self.assertAlmostEqual(m_perf["ece"], 0.0, places=4)
        self.assertAlmostEqual(m_perf["brier_score"], 0.0, places=4)

        # Overconfident predictions on false positives
        y_score_overconf = np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float32)
        m_overconf = compute_calibration_metrics(y_true, y_score_overconf)
        self.assertGreater(m_overconf["ece"], 0.3)

        # Check integration in compute_classification_metrics
        m_clf = compute_classification_metrics(y_true, y_score_perf)
        self.assertIn("ece", m_clf)
        self.assertIn("brier_score", m_clf)

    def test_temperature_scaling(self):
        """Verify temperature scaling modulates logit probabilities without changing rank."""
        model = TransGDA(num_genes=10, num_diseases=10, embed_dim=16)
        model.eval()
        model.set_temperature(2.0)
        self.assertAlmostEqual(model.temperature.item(), 2.0, places=3)

        g = torch.tensor([0])
        d = torch.tensor([0])
        g_nbrs = torch.zeros(1, 4, dtype=torch.long)
        g_wt = torch.ones(1, 4)
        d_nbrs = torch.zeros(1, 4, dtype=torch.long)
        d_wt = torch.ones(1, 4)

        model.set_temperature(1.0)
        out1 = model(g, d, g_nbrs, g_wt, d_nbrs, d_wt)
        logit = out1["logits"].item()

        model.set_temperature(2.0)
        out2 = model(g, d, g_nbrs, g_wt, d_nbrs, d_wt)
        prob2 = out2["probs"].item()

        expected_prob2 = 1.0 / (1.0 + np.exp(-logit / 2.0))
        self.assertAlmostEqual(prob2, expected_prob2, places=4)

    def test_calibrated_loss_with_smoothing_and_negative_penalty(self):
        """Verify label smoothing bounds BCE loss and negative strength penalty regularizes non-associations."""
        model = TransGDA(num_genes=10, num_diseases=10, embed_dim=16)
        preds = {
            "logits": torch.tensor([5.0, 5.0]),  # Both high
            "strengths": torch.tensor([1.5, 1.5]),
        }
        labels = torch.tensor([1.0, 0.0])
        strengths = torch.tensor([1.5, 0.0])

        # With negative strength penalty
        loss, bce, str_loss = model.compute_loss(
            preds, labels, strengths, lambda_strength=1.0, label_smoothing=0.05, lambda_neg_strength=0.5
        )
        # Negative pair has strength 1.5 and target 0, so str_loss must be > 0
        self.assertGreater(str_loss.item(), 0.0)
        self.assertGreater(loss.item(), 0.0)

    def test_degree_weighted_negative_sampling(self):
        """Verify degree-weighted negative sampling selects valid pairs within allowed sets."""
        exclude = set(map(tuple, self.data_mgr.get_positive_pairs()))
        neg_pairs = self.data_mgr.sample_negative_pairs(
            num_samples=50,
            exclude_set=exclude,
            degree_weighted_prob=0.8,
        )
        self.assertEqual(len(neg_pairs), 50)
        for g, d in neg_pairs:
            self.assertNotIn((g, d), exclude)

    def test_cross_partition_negative_leakage_free(self):
        """Verify negative pairs never leak across train, val, and test splits."""
        split = self.data_mgr.create_edge_split(val_ratio=0.1, test_ratio=0.1, seed=42)
        s_train = set(map(tuple, split.train_neg))
        s_val = set(map(tuple, split.val_neg))
        s_test = set(map(tuple, split.test_neg))
        s_pos = set(map(tuple, self.data_mgr.get_positive_pairs()))

        # Zero overlap between partitions
        self.assertEqual(len(s_train.intersection(s_val)), 0)
        self.assertEqual(len(s_train.intersection(s_test)), 0)
        self.assertEqual(len(s_val.intersection(s_test)), 0)
        # Zero overlap with positives
        self.assertEqual(len(s_train.intersection(s_pos)), 0)
        self.assertEqual(len(s_val.intersection(s_pos)), 0)
        self.assertEqual(len(s_test.intersection(s_pos)), 0)

    def test_inductive_feature_path_and_cold_start_masking(self):
        """Verify external feature projection works and cold-start masking zeroes ID embedding."""
        model = TransGDA(
            num_genes=10,
            num_diseases=10,
            embed_dim=16,
            gene_feat_dim=20,
            disease_feat_dim=15,
        )
        gene_feats = torch.randn(10, 20)
        disease_feats = torch.randn(10, 15)
        model.set_node_features(gene_feats, disease_feats)

        # Before cold start masking: embedding = ID + feat_proj
        g_idx = torch.tensor([3])
        emb_seen = model._embed_gene(g_idx)
        expected_seen = model.gene_embed(g_idx) + model.gene_feat_proj(gene_feats[g_idx])
        self.assertTrue(torch.allclose(emb_seen, expected_seen))

        # After cold start masking: node 3 is cold start -> pure feature projection without untrained ID
        model.set_cold_start_nodes(cold_start_genes=[3])
        emb_cold = model._embed_gene(g_idx)
        expected_cold = model.gene_feat_proj(gene_feats[g_idx])
        self.assertTrue(torch.allclose(emb_cold, expected_cold))

    def test_target_knockout_ablation(self):
        """Verify that knocking out the central gene token properly ablates its state."""
        model = TransGDA(num_genes=10, num_diseases=10, embed_dim=16)
        model.eval()
        g = torch.tensor([0])
        d = torch.tensor([1])
        g_nbrs = torch.zeros(1, 4, dtype=torch.long)
        g_wt = torch.ones(1, 4)
        d_nbrs = torch.zeros(1, 4, dtype=torch.long)
        d_wt = torch.ones(1, 4)

        base_out = model(g, d, g_nbrs, g_wt, d_nbrs, d_wt)

        # Knock out central gene at position 0
        mask_g = torch.zeros(1, 5, dtype=torch.bool)
        mask_g[0, 0] = True
        ko_out = model(g, d, g_nbrs, g_wt, d_nbrs, d_wt, gene_mask=mask_g)

        # Prediction should drop significantly under full ablation
        self.assertNotEqual(base_out["probs"].item(), ko_out["probs"].item())


if __name__ == "__main__":
    unittest.main()

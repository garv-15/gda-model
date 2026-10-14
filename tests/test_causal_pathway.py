"""
Unit tests for In-Silico Virtual Gene Knockout Engine and Statistical Pathway Validation Engine.
"""

import unittest
from pathlib import Path
import numpy as np
import torch

from src.data import DataLoaderManager
from src.similarity import GraphNeighborhoodManager
from src.models.transf_gda import TransGDA
from src.causal_knockout import InSilicoKnockoutEngine
from src.pathway_validation import GMTParser, PathwayEnrichmentEngine, benjamini_hochberg_fdr


class TestCausalAndPathwayValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_mgr = DataLoaderManager(data_dir=".")
        cls.split = cls.data_mgr.create_edge_split(val_ratio=0.1, test_ratio=0.1, seed=42)

        cls.graph_mgr = GraphNeighborhoodManager(
            num_genes=cls.data_mgr.meta.num_genes,
            num_diseases=cls.data_mgr.meta.num_diseases,
            k_gene_neighbors=8,
            k_disease_neighbors=8,
        )
        cls.graph_mgr.build_from_train_matrix(cls.split.train_matrix_binary, metric="cosine")

        cls.device = torch.device("cpu")
        cls.graph_mgr.to_device(cls.device)

        cls.model = TransGDA(
            num_genes=cls.data_mgr.meta.num_genes,
            num_diseases=cls.data_mgr.meta.num_diseases,
            embed_dim=32,
            num_heads=2,
            ffn_dim=64,
            hidden_dim=64,
            dropout=0.0,
        ).to(cls.device)
        cls.model.eval()

        cls.ko_engine = InSilicoKnockoutEngine(
            model=cls.model,
            graph_manager=cls.graph_mgr,
            meta=cls.data_mgr.meta,
            device=cls.device,
        )

        cls.pathway_engine = PathwayEnrichmentEngine(
            data_dir=".",
            universe_genes=set(cls.data_mgr.meta.genes),
        )

    def test_gmt_parser(self):
        """Test GMT parsing correctness."""
        parser = self.pathway_engine.parser
        self.assertGreater(len(parser.disease_to_genes), 100)
        self.assertGreater(len(parser.all_genes), 1000)
        self.assertIn("type 2 diabetes mellitus", parser.disease_to_genes)
        t2d_genes = parser.disease_to_genes["type 2 diabetes mellitus"]
        self.assertIn("TCF7L2", t2d_genes)

    def test_benjamini_hochberg_monotonicity(self):
        """Test BH-FDR adjusts p-values correctly and maintains monotonicity."""
        p_vals = np.array([0.001, 0.01, 0.02, 0.05, 0.10, 0.50])
        q_vals = benjamini_hochberg_fdr(p_vals)
        self.assertEqual(len(q_vals), len(p_vals))
        # Check sorted monotonicity
        self.assertTrue(np.all(np.diff(q_vals) >= -1e-12))
        self.assertTrue(np.all(q_vals >= p_vals - 1e-12))
        self.assertTrue(np.all(q_vals <= 1.0))

    def test_pathway_ora(self):
        """Test hypergeometric ORA for known disease-gene set."""
        # Query with known T2D genes
        t2d_known = ["TCF7L2", "PPARG", "KCNJ11", "SLC30A8", "GCK"]
        ora_df = self.pathway_engine.run_ora(t2d_known, min_overlap=2)
        self.assertFalse(ora_df.empty)

        # Check column presence
        expected_cols = {"term", "overlap_count", "p_value", "fdr_q_value", "odds_ratio"}
        self.assertTrue(expected_cols.issubset(set(ora_df.columns)))

        # Target term should have significant p-value
        t2d_rows = ora_df[ora_df["term"] == "type 2 diabetes mellitus"]
        self.assertFalse(t2d_rows.empty)
        p_val = t2d_rows.iloc[0]["p_value"]
        self.assertLess(p_val, 0.01)

    def test_insilico_knockout_execution(self):
        """Test systematic in-silico knockout of gene and disease neighbors."""
        gene = "TCF7L2"
        disease = "type 2 diabetes mellitus"

        ko_res = self.ko_engine.evaluate_pair_knockout(gene, disease)

        self.assertEqual(ko_res["gene"], gene)
        self.assertEqual(ko_res["disease"], disease)
        self.assertIn("base_prob", ko_res)
        self.assertIn("base_strength", ko_res)
        self.assertGreater(len(ko_res["gene_knockouts"]), 0)

        # First entry in raw sweep was central gene
        central_item = [x for x in ko_res["gene_knockouts"] if x["is_central"]][0]
        self.assertEqual(central_item["name"], gene)
        self.assertIn("delta_strength", central_item)
        self.assertIn("role", central_item)

    def test_synthetic_lethality_evaluation(self):
        """Test combinatorial double knockout for second-order epistasis."""
        gene = "TCF7L2"
        disease = "type 2 diabetes mellitus"

        comb_res = self.ko_engine.evaluate_synthetic_lethality(gene, disease, top_k=4)
        self.assertIsInstance(comb_res, list)
        if len(comb_res) > 0:
            first = comb_res[0]
            self.assertIn("gene_a", first)
            self.assertIn("gene_b", first)
            self.assertIn("epistasis_score", first)
            self.assertIn("interaction_type", first)

    def test_candidate_pair_validation(self):
        """Test candidate validation combining causal knockout effects with pathway ORA."""
        gene = "TCF7L2"
        disease = "type 2 diabetes mellitus"
        drivers = ["PPARG", "KCNJ11", "SLC30A8"]
        effects = {"PPARG": 0.15, "KCNJ11": 0.08, "SLC30A8": 0.05}

        val_res = self.pathway_engine.validate_candidate_pair(
            candidate_gene=gene,
            target_disease=disease,
            causal_driver_genes=drivers,
            pred_prob=0.99,
            pred_strength=2.5,
            driver_effects=effects,
        )

        self.assertEqual(val_res["candidate_gene"], gene)
        self.assertEqual(val_res["target_disease"], disease)
        self.assertGreater(val_res["biological_plausibility_index"], 0.0)
        self.assertTrue(val_res["is_statistically_validated"])

    def test_gradient_attribution_execution(self):
        """Test Integrated Gradients attribution on gene-disease pair."""
        from src.attribution import GradientAttributionEngine
        attr_engine = GradientAttributionEngine(
            model=self.model,
            graph_manager=self.graph_mgr,
            meta=self.data_mgr.meta,
            device=self.device,
        )
        res = attr_engine.attribute_pair("TCF7L2", "type 2 diabetes mellitus", target="strength", steps=5)
        self.assertEqual(res["gene"], "TCF7L2")
        self.assertEqual(res["disease"], "type 2 diabetes mellitus")
        self.assertGreater(len(res["gene_drivers"]), 0)
        self.assertGreater(len(res["disease_drivers"]), 0)
        self.assertIn("attribution_score", res["gene_drivers"][0])

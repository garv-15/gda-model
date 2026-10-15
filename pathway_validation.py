"""
Statistical Biological Pathway Validation Engine for TransGDA:
- Parses canonical GMT libraries (Disease -> Genes and Gene -> Diseases)
- Executes exact Hypergeometric Over-Representation Analysis (ORA) with Fisher's Exact Test
- Performs Benjamini-Hochberg False Discovery Rate (FDR) multiple testing correction
- Computes Odds Ratio, Fold Enrichment, and Biological Plausibility Index (BPI)
- Annotates novel candidate gene-disease discoveries with statistical validation metrics
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from .data import DataLoaderManager


class GMTParser:
    """Parses and indexes standard GMT (Gene Matrix Transposed) functional library files."""

    def __init__(self, data_dir: Union[str, Path] = "."):
        self.data_dir = Path(data_dir)
        self.disease_to_genes: Dict[str, Set[str]] = {}
        self.disease_to_doid: Dict[str, str] = {}
        self.gene_to_diseases: Dict[str, Set[str]] = {}
        self.all_genes: Set[str] = set()
        self.all_diseases: Set[str] = set()
        self._load_gmt_files()

    def _load_gmt_files(self):
        # 1. Parse gene_set_library_crisp.gmt (Disease \t DOID \t gene1 \t gene2 ...)
        gene_set_file = self.data_dir / "gene_set_library_crisp.gmt"
        if gene_set_file.exists():
            with open(gene_set_file, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        disease = parts[0].strip()
                        doid = parts[1].strip()
                        genes = set(p.strip() for p in parts[2:] if p.strip())
                        self.disease_to_genes[disease] = genes
                        self.disease_to_doid[disease] = doid
                        self.all_diseases.add(disease)
                        self.all_genes.update(genes)

        # 2. Parse attribute_set_library_crisp.gmt (Gene \t Ensembl_GeneID \t disease1 \t disease2 ...)
        attr_set_file = self.data_dir / "attribute_set_library_crisp.gmt"
        if attr_set_file.exists():
            with open(attr_set_file, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        gene = parts[0].strip()
                        diseases = set(p.strip() for p in parts[2:] if p.strip())
                        self.gene_to_diseases[gene] = diseases
                        self.all_genes.add(gene)
                        self.all_diseases.update(diseases)


def benjamini_hochberg_fdr(p_values: np.ndarray) -> np.ndarray:
    """
    Computes Benjamini-Hochberg False Discovery Rate (FDR) adjusted q-values.
    Guarantees monotonicity: q_(i) <= q_(i+1).
    """
    p_vals = np.asarray(p_values, dtype=np.float64)
    n = len(p_vals)
    if n == 0:
        return np.array([], dtype=np.float64)

    order = np.argsort(p_vals)
    sorted_p = p_vals[order]

    # q_i = min_{j >= i} (p_j * n / (j + 1))
    ranks = np.arange(1, n + 1, dtype=np.float64)
    raw_q = sorted_p * (n / ranks)

    # Monotonicity enforcement from back to front
    adjusted_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    adjusted_q = np.clip(adjusted_q, 0.0, 1.0)

    # Invert sorting back to original order
    orig_order = np.empty_like(order)
    orig_order[order] = np.arange(n)
    return adjusted_q[orig_order]


class PathwayEnrichmentEngine:
    """
    Performs Over-Representation Analysis (ORA) to statistically validate
    gene-disease associations and candidate driver modules.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = ".",
        universe_genes: Optional[Set[str]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.parser = GMTParser(data_dir=data_dir)
        if universe_genes is not None:
            self.universe = set(universe_genes)
        else:
            self.universe = self.parser.all_genes

        self.universe_size = len(self.universe)

    def run_ora(
        self,
        query_genes: Union[List[str], Set[str]],
        background_universe: Optional[Set[str]] = None,
        min_overlap: int = 1,
    ) -> pd.DataFrame:
        """
        Runs hypergeometric Over-Representation Analysis (Fisher's exact test)
        for a set of query genes against all curated disease/pathway sets in the GMT library.
        """
        universe = background_universe or self.universe
        n_universe = len(universe)

        # Filter query genes to background universe
        filtered_query = set(query_genes).intersection(universe)
        n_query = len(filtered_query)

        if n_query == 0:
            return pd.DataFrame(columns=[
                "term", "doid", "overlap_count", "query_size", "term_size",
                "odds_ratio", "fold_enrichment", "p_value", "fdr_q_value", "overlap_genes",
            ])

        records = []
        p_values = []

        for disease_term, term_genes in self.parser.disease_to_genes.items():
            valid_term_genes = term_genes.intersection(universe)
            k_term = len(valid_term_genes)
            if k_term == 0:
                continue

            overlap = filtered_query.intersection(valid_term_genes)
            k_overlap = len(overlap)

            if k_overlap < min_overlap:
                continue

            # Hypergeometric test:
            # P(X >= k_overlap) where population = n_universe, successes in pop = k_term, draws = n_query
            p_val = float(hypergeom.sf(k_overlap - 1, n_universe, k_term, n_query))
            p_values.append(p_val)

            # Odds Ratio = (k * (N - K - n + k)) / ((K - k) * (n - k))
            num = k_overlap * (n_universe - k_term - n_query + k_overlap)
            denom = max(1, (k_term - k_overlap) * (n_query - k_overlap))
            odds_ratio = float(num / denom)

            # Fold Enrichment = (k / n) / (K / N)
            expected = (k_term / n_universe) * n_query
            fold_enrichment = float(k_overlap / expected) if expected > 0 else 0.0

            doid = self.parser.disease_to_doid.get(disease_term, "NA")

            records.append({
                "term": disease_term,
                "doid": doid,
                "overlap_count": k_overlap,
                "query_size": n_query,
                "term_size": k_term,
                "odds_ratio": odds_ratio,
                "fold_enrichment": fold_enrichment,
                "p_value": p_val,
                "overlap_genes": sorted(list(overlap)),
            })

        if not records:
            return pd.DataFrame(columns=[
                "term", "doid", "overlap_count", "query_size", "term_size",
                "odds_ratio", "fold_enrichment", "p_value", "fdr_q_value", "overlap_genes",
            ])

        # Apply Benjamini-Hochberg FDR
        q_values = benjamini_hochberg_fdr(np.array(p_values))
        for rec, q_val in zip(records, q_values):
            rec["fdr_q_value"] = float(q_val)

        df = pd.DataFrame(records)
        df.sort_values(by=["fdr_q_value", "p_value"], ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def validate_candidate_pair(
        self,
        candidate_gene: str,
        target_disease: str,
        causal_driver_genes: List[str],
        pred_prob: float = 1.0,
        pred_strength: float = 1.0,
        driver_effects: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Union[str, float, int, list]]:
        """
        Validates a candidate novel gene-disease association by:
        1. Checking direct overlap between causal driver module and target disease's curated genes.
        2. Computing target disease hypergeometric p-value & FDR q-value.
        3. Quantifying Biological Plausibility Index (BPI).
        """
        # Exclude candidate gene itself to evaluate genuine external driver module overlap
        driver_set = set(causal_driver_genes) - {candidate_gene}
        if len(driver_set) == 0:
            driver_set = set(causal_driver_genes)

        target_disease_genes = self.parser.disease_to_genes.get(target_disease, set())

        overlap = driver_set.intersection(target_disease_genes)
        k_overlap = len(overlap)

        n_universe = self.universe_size
        k_term = len(target_disease_genes.intersection(self.universe))
        n_query = len(driver_set.intersection(self.universe))

        if k_term > 0 and n_query > 0:
            p_val = float(hypergeom.sf(k_overlap - 1, n_universe, k_term, n_query))
        else:
            p_val = 1.0

        # Run broader ORA across all pathways to find FDR adjusted q-value
        ora_df = self.run_ora(driver_set, min_overlap=1)
        matched_row = ora_df[ora_df["term"] == target_disease]

        if not matched_row.empty:
            q_val = float(matched_row.iloc[0]["fdr_q_value"])
            odds_ratio = float(matched_row.iloc[0]["odds_ratio"])
        else:
            q_val = p_val
            odds_ratio = 1.0

        # Compute BPI: Calibrated probability x strength x (1 + causal impact) x (1 + significance)
        sum_pos_effect = 0.0
        if driver_effects:
            sum_pos_effect = sum(max(0.0, eff) for eff in driver_effects.values())

        # Negative log q-value capped at 5 to prevent single-feature dominance
        log_q_boost = min(5.0, -np.log10(max(1e-10, q_val))) if k_overlap > 0 else 0.0
        bpi = float(pred_prob * pred_strength * (1.0 + sum_pos_effect) * (1.0 + log_q_boost))

        is_validated = (q_val < 0.05 and k_overlap >= 1) or (p_val < 0.01 and k_overlap >= 1)

        return {
            "candidate_gene": candidate_gene,
            "target_disease": target_disease,
            "target_doid": self.parser.disease_to_doid.get(target_disease, "NA"),
            "pred_prob": pred_prob,
            "pred_strength": pred_strength,
            "driver_module_size": len(driver_set),
            "overlap_count": k_overlap,
            "overlap_genes": sorted(list(overlap)),
            "p_value": p_val,
            "fdr_q_value": q_val,
            "odds_ratio": odds_ratio,
            "biological_plausibility_index": bpi,
            "is_statistically_validated": is_validated,
            "top_enriched_pathways": ora_df.head(5).to_dict(orient="records"),
        }

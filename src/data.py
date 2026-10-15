"""
Data loading, canonical index alignment, negative sampling, and CV split generation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class DatasetMetadata:
    """Canonical mapping for genes and diseases."""
    num_genes: int
    num_diseases: int
    genes: List[str]
    gene_to_idx: Dict[str, int]
    idx_to_gene: Dict[int, str]
    gene_df: pd.DataFrame
    diseases: List[str]
    disease_to_idx: Dict[str, int]
    idx_to_disease: Dict[int, str]
    disease_df: pd.DataFrame


@dataclass
class SplitData:
    """Train/Val/Test split container with leakage-free train matrices."""
    split_type: str
    train_edges: np.ndarray  # Shape: (N_train, 2) [gene_idx, disease_idx]
    val_edges: np.ndarray    # Shape: (N_val, 2)
    test_edges: np.ndarray   # Shape: (N_test, 2)
    train_neg: np.ndarray    # Shape: (N_train_neg, 2)
    val_neg: np.ndarray      # Shape: (N_val_neg, 2)
    test_neg: np.ndarray     # Shape: (N_test_neg, 2)
    train_matrix_binary: np.ndarray     # Leakage-free A_train (N_g, N_d)
    train_matrix_strength: np.ndarray   # Leakage-free S_train (N_g, N_d)
    val_matrix_binary: np.ndarray
    test_matrix_binary: np.ndarray
    test_gene_indices: Optional[np.ndarray] = None
    test_disease_indices: Optional[np.ndarray] = None


class GDADataset(Dataset):
    """PyTorch Dataset for Gene-Disease pairs, existence labels, and strength targets."""

    def __init__(
        self,
        pairs: np.ndarray,
        labels: np.ndarray,
        strengths: np.ndarray,
    ):
        self.pairs = torch.tensor(pairs, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.strengths = torch.tensor(strengths, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.pairs[idx, 0], self.pairs[idx, 1], self.labels[idx], self.strengths[idx]


class DataLoaderManager:
    """Manages raw data loading, canonical alignment, and split construction."""

    def __init__(self, data_dir: Union[str, Path] = "."):
        self.data_dir = Path(data_dir)
        self.meta: Optional[DatasetMetadata] = None
        self.binary_matrix: Optional[np.ndarray] = None
        self.strength_matrix: Optional[np.ndarray] = None
        self._load_and_align_all()

    def _load_metadata(self):
        gene_file = self.data_dir / "gene_list_terms.csv"
        # gene_list_terms.csv is tab-separated with columns: GeneSym, Ensemble Acc, GeneID
        df_gene = pd.read_csv(gene_file, sep="\t")
        genes = df_gene["GeneSym"].tolist()
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        idx_to_gene = {i: g for i, g in enumerate(genes)}

        disease_file = self.data_dir / "attribute_list_entries.txt"
        df_disease = pd.read_csv(disease_file, sep="\t")
        diseases = df_disease["Disease"].tolist()
        disease_to_idx = {d: i for i, d in enumerate(diseases)}
        idx_to_disease = {i: d for i, d in enumerate(diseases)}

        self.meta = DatasetMetadata(
            num_genes=len(genes),
            num_diseases=len(diseases),
            genes=genes,
            gene_to_idx=gene_to_idx,
            idx_to_gene=idx_to_gene,
            gene_df=df_gene,
            diseases=diseases,
            disease_to_idx=disease_to_idx,
            idx_to_disease=idx_to_disease,
            disease_df=df_disease,
        )

    def _load_matrix_file(self, filepath: Path) -> pd.DataFrame:
        """Loads a matrix file with 3 metadata header rows and 3 metadata columns."""
        with open(filepath, "r") as f:
            d_names = f.readline().strip().split("\t")[3:]
            f.readline()  # DOID row
            f.readline()  # GeneID/NA row
            rows = []
            genes = []
            for line in f:
                parts = line.strip().split("\t")
                genes.append(parts[0])
                rows.append([float(x) for x in parts[3:]])
        df = pd.DataFrame(rows, index=genes, columns=d_names)
        return df

    def _load_and_align_all(self):
        """Loads all matrices and aligns them strictly to canonical order."""
        self._load_metadata()

        df_bin = self._load_matrix_file(self.data_dir / "gene_attribute_matrix.txt")
        # Raw association scores (e.g. -log10(p)) are the regression target.
        df_clean = self._load_matrix_file(self.data_dir / "gene_attribute_matrix_cleaned.txt")

        # Canonical alignment
        canonical_genes = self.meta.genes
        canonical_diseases = self.meta.diseases

        df_bin_aligned = df_bin.reindex(index=canonical_genes, columns=canonical_diseases).fillna(0.0)
        df_clean_aligned = df_clean.reindex(index=canonical_genes, columns=canonical_diseases).fillna(0.0)

        self.binary_matrix = df_bin_aligned.values.astype(np.float32)
        # log1p tames the heavy right tail (raw scores span ~0.2..304) while keeping
        # the target non-negative so the Softplus strength head remains valid.
        self.strength_matrix = np.log1p(df_clean_aligned.values.astype(np.float32))

        # Consistency: binary and strength matrices must share the exact same support.
        bin_pos = self.binary_matrix > 0
        str_pos = self.strength_matrix > 0
        if not np.array_equal(bin_pos, str_pos):
            n_bin, n_str = int(bin_pos.sum()), int(str_pos.sum())
            raise ValueError(
                f"Support mismatch between binary ({n_bin}) and strength ({n_str}) matrices."
            )
        if int(bin_pos.sum()) == 0:
            raise ValueError("No positive gene-disease associations found after alignment.")

    def get_positive_pairs(self) -> np.ndarray:
        """Returns array of shape (N_pos, 2) containing [gene_idx, disease_idx]."""
        rows, cols = np.nonzero(self.binary_matrix > 0)
        return np.column_stack([rows, cols])

    def load_external_similarity_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads precomputed external similarity matrices from disk aligned with canonical genes and diseases.
        Returns:
            gene_sim: (num_genes, num_genes) float32 array
            disease_sim: (num_diseases, num_diseases) float32 array
        """
        gene_file = self.data_dir / "gene_similarity_matrix_cosine.txt"
        disease_file = self.data_dir / "attribute_similarity_matrix_cosine.txt"

        # Load and align gene similarity
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
        df_g = pd.DataFrame(rows, index=genes, columns=g_cols)
        df_g_aligned = df_g.reindex(index=self.meta.genes, columns=self.meta.genes).fillna(0.0)
        gene_sim = df_g_aligned.values.astype(np.float32)

        # Load and align disease similarity
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
        df_d = pd.DataFrame(rows, index=diseases, columns=d_cols)
        df_d_aligned = df_d.reindex(index=self.meta.diseases, columns=self.meta.diseases).fillna(0.0)
        disease_sim = df_d_aligned.values.astype(np.float32)

        return gene_sim, disease_sim

    def sample_negative_pairs(
        self,
        num_samples: int,
        exclude_set: set,
        degree_matrix: Optional[np.ndarray] = None,
        allowed_genes: Optional[np.ndarray] = None,
        allowed_diseases: Optional[np.ndarray] = None,
        rng: Optional[np.random.RandomState] = None,
        degree_weighted_prob: float = 0.5,
    ) -> np.ndarray:
        """
        Rejection samples negative pairs not in exclude_set.
        Uses a hybrid sampling strategy: uniform negatives mixed with degree-weighted negatives
        (proportional to deg^0.75) to counter degree assortativity bias and eliminate hub overconfidence.

        Degree weights are computed from ``degree_matrix`` (the *training* association matrix)
        so that held-out edge structure never leaks into the negative distribution. If
        ``degree_matrix`` is None the full binary matrix is used (transductive use only).
        """
        if rng is None:
            rng = np.random.RandomState(42)

        if allowed_genes is None:
            allowed_genes = np.arange(self.meta.num_genes)
        if allowed_diseases is None:
            allowed_diseases = np.arange(self.meta.num_diseases)

        deg_source = degree_matrix if degree_matrix is not None else self.binary_matrix

        # Precompute degree weights on allowed entities
        if degree_weighted_prob > 0.0 and deg_source is not None:
            gene_degs = deg_source[allowed_genes].sum(axis=1) + 1.0
            p_genes = (gene_degs ** 0.75) / (gene_degs ** 0.75).sum()

            disease_degs = deg_source[:, allowed_diseases].sum(axis=0) + 1.0
            p_diseases = (disease_degs ** 0.75) / (disease_degs ** 0.75).sum()
        else:
            p_genes = None
            p_diseases = None

        neg_pairs = []
        sampled_set = set()

        max_attempts = num_samples * 25
        attempts = 0

        while len(neg_pairs) < num_samples and attempts < max_attempts:
            attempts += 1
            if degree_weighted_prob > 0.0 and p_genes is not None and rng.rand() < degree_weighted_prob:
                g = allowed_genes[rng.choice(len(allowed_genes), p=p_genes)]
                d = allowed_diseases[rng.choice(len(allowed_diseases), p=p_diseases)]
            else:
                g = rng.choice(allowed_genes)
                d = rng.choice(allowed_diseases)
            pair = (int(g), int(d))
            if pair not in exclude_set and pair not in sampled_set:
                neg_pairs.append([g, d])
                sampled_set.add(pair)

        if len(neg_pairs) < num_samples:
            # Fallback for dense spaces
            all_pairs = [
                [g, d]
                for g in allowed_genes
                for d in allowed_diseases
                if (g, d) not in exclude_set and (g, d) not in sampled_set
            ]
            if len(all_pairs) > 0:
                indices = rng.choice(len(all_pairs), size=min(num_samples - len(neg_pairs), len(all_pairs)), replace=False)
                for idx in indices:
                    neg_pairs.append(all_pairs[idx])

        return np.array(neg_pairs, dtype=np.int64)

    def create_edge_split(
        self,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        neg_ratio_train: int = 1,
        neg_ratio_eval: int = 1,
        seed: int = 42,
    ) -> SplitData:
        """
        Creates Transductive Edge Split (Cell CV).
        Splits existing positive edges randomly into train, val, and test.
        Leakage-free train matrices mask out val and test edges.
        """
        rng = np.random.RandomState(seed)
        pos_pairs = self.get_positive_pairs()
        num_pos = len(pos_pairs)

        perm = rng.permutation(num_pos)
        pos_shuffled = pos_pairs[perm]

        n_test = int(num_pos * test_ratio)
        n_val = int(num_pos * val_ratio)
        n_train = num_pos - n_val - n_test

        train_pos = pos_shuffled[:n_train]
        val_pos = pos_shuffled[n_train:n_train + n_val]
        test_pos = pos_shuffled[n_train + n_val:]

        # Build leakage-free train matrices
        train_mat_bin = np.zeros_like(self.binary_matrix)
        train_mat_str = np.zeros_like(self.strength_matrix)
        train_mat_bin[train_pos[:, 0], train_pos[:, 1]] = 1.0
        train_mat_str[train_pos[:, 0], train_pos[:, 1]] = self.strength_matrix[train_pos[:, 0], train_pos[:, 1]]

        val_mat_bin = np.zeros_like(self.binary_matrix)
        val_mat_bin[val_pos[:, 0], val_pos[:, 1]] = 1.0

        test_mat_bin = np.zeros_like(self.binary_matrix)
        test_mat_bin[test_pos[:, 0], test_pos[:, 1]] = 1.0

        all_known_pos = set(map(tuple, pos_pairs))

        # Sample negatives with cross-partition exclusion to eliminate negative leakage
        exclude_train = set(all_known_pos)
        train_neg = self.sample_negative_pairs(
            len(train_pos) * neg_ratio_train, exclude_train, degree_matrix=train_mat_bin, rng=rng,
        )
        exclude_val = exclude_train | set(map(tuple, train_neg))
        val_neg = self.sample_negative_pairs(
            len(val_pos) * neg_ratio_eval, exclude_val, degree_matrix=train_mat_bin, rng=rng,
        )
        exclude_test = exclude_val | set(map(tuple, val_neg))
        test_neg = self.sample_negative_pairs(
            len(test_pos) * neg_ratio_eval, exclude_test, degree_matrix=train_mat_bin, rng=rng,
        )

        return SplitData(
            split_type="edge",
            train_edges=train_pos,
            val_edges=val_pos,
            test_edges=test_pos,
            train_neg=train_neg,
            val_neg=val_neg,
            test_neg=test_neg,
            train_matrix_binary=train_mat_bin,
            train_matrix_strength=train_mat_str,
            val_matrix_binary=val_mat_bin,
            test_matrix_binary=test_mat_bin,
        )

    def create_gene_held_out_split(
        self,
        val_ratio: float = 0.10,
        test_ratio: float = 0.20,
        neg_ratio_train: int = 1,
        neg_ratio_eval: int = 1,
        seed: int = 42,
    ) -> SplitData:
        """
        Creates Inductive Gene-Held-Out Split (Cold-Start Genes).
        All associations of test genes are completely hidden during training.
        """
        rng = np.random.RandomState(seed)
        num_genes = self.meta.num_genes
        perm_genes = rng.permutation(num_genes)

        n_test_genes = int(num_genes * test_ratio)
        n_val_genes = int(num_genes * val_ratio)

        test_genes = set(perm_genes[:n_test_genes])
        val_genes = set(perm_genes[n_test_genes:n_test_genes + n_val_genes])
        train_genes = set(perm_genes[n_test_genes + n_val_genes:])

        pos_pairs = self.get_positive_pairs()
        train_pos_list = []
        val_pos_list = []
        test_pos_list = []

        for g, d in pos_pairs:
            if g in test_genes:
                test_pos_list.append([g, d])
            elif g in val_genes:
                val_pos_list.append([g, d])
            else:
                train_pos_list.append([g, d])

        train_pos = np.array(train_pos_list, dtype=np.int64)
        val_pos = np.array(val_pos_list, dtype=np.int64)
        test_pos = np.array(test_pos_list, dtype=np.int64)

        train_mat_bin = np.zeros_like(self.binary_matrix)
        train_mat_str = np.zeros_like(self.strength_matrix)
        train_mat_bin[train_pos[:, 0], train_pos[:, 1]] = 1.0
        train_mat_str[train_pos[:, 0], train_pos[:, 1]] = self.strength_matrix[train_pos[:, 0], train_pos[:, 1]]

        val_mat_bin = np.zeros_like(self.binary_matrix)
        val_mat_bin[val_pos[:, 0], val_pos[:, 1]] = 1.0

        test_mat_bin = np.zeros_like(self.binary_matrix)
        test_mat_bin[test_pos[:, 0], test_pos[:, 1]] = 1.0

        all_known_pos = set(map(tuple, pos_pairs))

        # Sample negatives respecting gene split partition with cross-partition exclusion
        exclude_train = set(all_known_pos)
        train_neg = self.sample_negative_pairs(
            len(train_pos) * neg_ratio_train,
            exclude_train,
            degree_matrix=train_mat_bin,
            allowed_genes=np.array(list(train_genes)),
            rng=rng,
        )
        exclude_val = exclude_train | set(map(tuple, train_neg))
        val_neg = self.sample_negative_pairs(
            len(val_pos) * neg_ratio_eval,
            exclude_val,
            degree_matrix=train_mat_bin,
            allowed_genes=np.array(list(val_genes)),
            rng=rng,
        )
        exclude_test = exclude_val | set(map(tuple, val_neg))
        test_neg = self.sample_negative_pairs(
            len(test_pos) * neg_ratio_eval,
            exclude_test,
            degree_matrix=train_mat_bin,
            allowed_genes=np.array(list(test_genes)),
            rng=rng,
        )

        return SplitData(
            split_type="gene_held_out",
            train_edges=train_pos,
            val_edges=val_pos,
            test_edges=test_pos,
            train_neg=train_neg,
            val_neg=val_neg,
            test_neg=test_neg,
            train_matrix_binary=train_mat_bin,
            train_matrix_strength=train_mat_str,
            val_matrix_binary=val_mat_bin,
            test_matrix_binary=test_mat_bin,
            test_gene_indices=np.array(list(test_genes)),
        )

    def create_disease_held_out_split(
        self,
        val_ratio: float = 0.10,
        test_ratio: float = 0.20,
        neg_ratio_train: int = 1,
        neg_ratio_eval: int = 1,
        seed: int = 42,
    ) -> SplitData:
        """
        Creates Inductive Disease-Held-Out Split (Cold-Start Diseases).
        All associations of test diseases are completely hidden during training.
        """
        rng = np.random.RandomState(seed)
        num_diseases = self.meta.num_diseases
        perm_diseases = rng.permutation(num_diseases)

        n_test_d = int(num_diseases * test_ratio)
        n_val_d = int(num_diseases * val_ratio)

        test_d = set(perm_diseases[:n_test_d])
        val_d = set(perm_diseases[n_test_d:n_test_d + n_val_d])
        train_d = set(perm_diseases[n_test_d + n_val_d:])

        pos_pairs = self.get_positive_pairs()
        train_pos_list = []
        val_pos_list = []
        test_pos_list = []

        for g, d in pos_pairs:
            if d in test_d:
                test_pos_list.append([g, d])
            elif d in val_d:
                val_pos_list.append([g, d])
            else:
                train_pos_list.append([g, d])

        train_pos = np.array(train_pos_list, dtype=np.int64)
        val_pos = np.array(val_pos_list, dtype=np.int64)
        test_pos = np.array(test_pos_list, dtype=np.int64)

        train_mat_bin = np.zeros_like(self.binary_matrix)
        train_mat_str = np.zeros_like(self.strength_matrix)
        train_mat_bin[train_pos[:, 0], train_pos[:, 1]] = 1.0
        train_mat_str[train_pos[:, 0], train_pos[:, 1]] = self.strength_matrix[train_pos[:, 0], train_pos[:, 1]]

        val_mat_bin = np.zeros_like(self.binary_matrix)
        val_mat_bin[val_pos[:, 0], val_pos[:, 1]] = 1.0

        test_mat_bin = np.zeros_like(self.binary_matrix)
        test_mat_bin[test_pos[:, 0], test_pos[:, 1]] = 1.0

        all_known_pos = set(map(tuple, pos_pairs))

        # Sample negatives respecting disease split partition with cross-partition exclusion
        exclude_train = set(all_known_pos)
        train_neg = self.sample_negative_pairs(
            len(train_pos) * neg_ratio_train,
            exclude_train,
            degree_matrix=train_mat_bin,
            allowed_diseases=np.array(list(train_d)),
            rng=rng,
        )
        exclude_val = exclude_train | set(map(tuple, train_neg))
        val_neg = self.sample_negative_pairs(
            len(val_pos) * neg_ratio_eval,
            exclude_val,
            degree_matrix=train_mat_bin,
            allowed_diseases=np.array(list(val_d)),
            rng=rng,
        )
        exclude_test = exclude_val | set(map(tuple, val_neg))
        test_neg = self.sample_negative_pairs(
            len(test_pos) * neg_ratio_eval,
            exclude_test,
            degree_matrix=train_mat_bin,
            allowed_diseases=np.array(list(test_d)),
            rng=rng,
        )

        return SplitData(
            split_type="disease_held_out",
            train_edges=train_pos,
            val_edges=val_pos,
            test_edges=test_pos,
            train_neg=train_neg,
            val_neg=val_neg,
            test_neg=test_neg,
            train_matrix_binary=train_mat_bin,
            train_matrix_strength=train_mat_str,
            val_matrix_binary=val_mat_bin,
            test_matrix_binary=test_mat_bin,
            test_disease_indices=np.array(list(test_d)),
        )

    def build_torch_dataloaders(
        self,
        split: SplitData,
        batch_size: int = 256,
        shuffle_train: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Constructs PyTorch DataLoaders for train, val, and test partitions."""
        # Train dataset
        train_pairs = np.vstack([split.train_edges, split.train_neg])
        train_labels = np.concatenate([
            np.ones(len(split.train_edges), dtype=np.float32),
            np.zeros(len(split.train_neg), dtype=np.float32),
        ])
        train_strengths = np.concatenate([
            self.strength_matrix[split.train_edges[:, 0], split.train_edges[:, 1]],
            np.zeros(len(split.train_neg), dtype=np.float32),
        ])
        train_ds = GDADataset(train_pairs, train_labels, train_strengths)

        # Val dataset
        val_pairs = np.vstack([split.val_edges, split.val_neg])
        val_labels = np.concatenate([
            np.ones(len(split.val_edges), dtype=np.float32),
            np.zeros(len(split.val_neg), dtype=np.float32),
        ])
        val_strengths = np.concatenate([
            self.strength_matrix[split.val_edges[:, 0], split.val_edges[:, 1]],
            np.zeros(len(split.val_neg), dtype=np.float32),
        ])
        val_ds = GDADataset(val_pairs, val_labels, val_strengths)

        # Test dataset
        test_pairs = np.vstack([split.test_edges, split.test_neg])
        test_labels = np.concatenate([
            np.ones(len(split.test_edges), dtype=np.float32),
            np.zeros(len(split.test_neg), dtype=np.float32),
        ])
        test_strengths = np.concatenate([
            self.strength_matrix[split.test_edges[:, 0], split.test_edges[:, 1]],
            np.zeros(len(split.test_neg), dtype=np.float32),
        ])
        test_ds = GDADataset(test_pairs, test_labels, test_strengths)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

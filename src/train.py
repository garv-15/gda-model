"""
Training pipeline with multi-task loss, validation tracking, early stopping,
checkpoint save/load, and GPU/MPS/CPU support.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models.transf_gda import TransGDA
from .similarity import GraphNeighborhoodManager
from .evaluate import evaluate_predictions


def get_optimal_device() -> torch.device:
    """Returns Apple Silicon MPS if available, otherwise CUDA or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class TransGDATrainer:
    """Trainer for TransGDA and neural baseline models."""

    def __init__(
        self,
        model: nn.Module,
        graph_manager: Optional[GraphNeighborhoodManager] = None,
        device: Optional[torch.device] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_strength: float = 1.0,
        label_smoothing: float = 0.05,
        lambda_neg_strength: float = 0.2,
        lambda_attn_entropy: float = 0.005,
    ):
        self.device = device or get_optimal_device()
        self.model = model.to(self.device)
        self.graph_manager = graph_manager
        if self.graph_manager is not None:
            self.graph_manager.to_device(self.device)

        self.lambda_strength = lambda_strength
        self.label_smoothing = label_smoothing
        self.lambda_neg_strength = lambda_neg_strength
        self.lambda_attn_entropy = lambda_attn_entropy
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3
        )
        self.best_state = None
        self.best_val_metric = -float("inf")

    def _prepare_batch_inputs(
        self,
        g_idx: torch.Tensor,
        d_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Slices neighbor indices and weights for the current batch."""
        g_idx = g_idx.to(self.device)
        d_idx = d_idx.to(self.device)
        inputs = {
            "gene_idx": g_idx,
            "disease_idx": d_idx,
        }

        if self.graph_manager is not None and self.graph_manager.gene_neighbor_idx is not None:
            inputs["gene_nbr_idx"] = self.graph_manager.gene_neighbor_idx[g_idx]
            inputs["gene_nbr_wt"] = self.graph_manager.gene_neighbor_wt[g_idx]
            inputs["disease_nbr_idx"] = self.graph_manager.disease_neighbor_idx[d_idx]
            inputs["disease_nbr_wt"] = self.graph_manager.disease_neighbor_wt[d_idx]

        return inputs

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_str = 0.0
        batches = 0

        for g_idx, d_idx, labels, strengths in dataloader:
            inputs = self._prepare_batch_inputs(g_idx, d_idx)
            labels = labels.to(self.device)
            strengths = strengths.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(**inputs)
            loss, bce, str_loss = self.model.compute_loss(
                preds,
                labels,
                strengths,
                lambda_strength=self.lambda_strength,
                label_smoothing=self.label_smoothing,
                lambda_neg_strength=self.lambda_neg_strength,
                lambda_attn_entropy=self.lambda_attn_entropy,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_bce += bce.item()
            total_str += str_loss.item()
            batches += 1

        return {
            "loss": total_loss / max(batches, 1),
            "bce": total_bce / max(batches, 1),
            "strength_loss": total_str / max(batches, 1),
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        self.model.eval()
        all_labels = []
        all_strengths = []
        all_probs = []
        all_pred_strengths = []

        for g_idx, d_idx, labels, strengths in dataloader:
            inputs = self._prepare_batch_inputs(g_idx, d_idx)
            preds = self.model(**inputs)

            all_labels.append(labels.numpy())
            all_strengths.append(strengths.numpy())
            all_probs.append(preds["probs"].detach().cpu().numpy())
            all_pred_strengths.append(preds["strengths"].detach().cpu().numpy())

        y_true = np.concatenate(all_labels)
        s_true = np.concatenate(all_strengths)
        y_score = np.concatenate(all_probs)
        s_pred = np.concatenate(all_pred_strengths)

        metrics = evaluate_predictions(y_true, y_score, s_true, s_pred)
        return metrics, y_score, s_pred

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        patience: int = 5,
        verbose: bool = True,
    ) -> Dict[str, list]:
        history = {"train_loss": [], "val_auroc": [], "val_auprc": [], "val_rmse": []}
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics, _, _ = self.evaluate(val_loader)
            elapsed = time.time() - t0

            val_target = val_metrics.get("auroc", 0.0)
            self.scheduler.step(val_target)

            history["train_loss"].append(train_metrics["loss"])
            history["val_auroc"].append(val_metrics.get("auroc", 0.0))
            history["val_auprc"].append(val_metrics.get("auprc", 0.0))
            history["val_rmse"].append(val_metrics.get("rmse", 0.0))

            if verbose and (epoch % 2 == 0 or epoch == 1 or epoch == epochs):
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] "
                    f"Loss: {train_metrics['loss']:.4f} (BCE: {train_metrics['bce']:.4f}, Str: {train_metrics['strength_loss']:.4f}) | "
                    f"Val AUROC: {val_metrics.get('auroc', 0.0):.4f} | "
                    f"Val AUPRC: {val_metrics.get('auprc', 0.0):.4f} | "
                    f"Val RMSE: {val_metrics.get('rmse', 0.0):.4f}"
                )

            if val_target > self.best_val_metric:
                self.best_val_metric = val_target
                self.best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch}. Best Val AUROC: {self.best_val_metric:.4f}")
                    break

        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        # Post-hoc temperature calibration to eliminate overconfidence
        cal_t = self.calibrate_temperature(val_loader, verbose=verbose)
        history["calibrated_temperature"] = cal_t

        return history

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Saves the model state dict, optimizer state, best validation metric,
        and temperature (if present) to *path*.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_metric": self.best_val_metric,
        }
        # Persist calibrated temperature when available
        if hasattr(self.model, "temperature"):
            payload["temperature"] = float(self.model.temperature.item())
        torch.save(payload, path)
        print(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Loads model weights (and optional temperature) from *path*.
        After this call the model is ready for inference — no training needed.

        Raises:
            FileNotFoundError: if *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "best_val_metric" in payload:
            self.best_val_metric = payload["best_val_metric"]
        if "temperature" in payload:
            t = payload["temperature"]
            if hasattr(self.model, "set_temperature"):
                self.model.set_temperature(t)
            elif hasattr(self.model, "temperature"):
                self.model.temperature.data.fill_(t)
        print(f"Checkpoint loaded ← {path}  (best val metric: {self.best_val_metric:.4f})")

    def calibrate_temperature(self, val_loader: DataLoader, verbose: bool = False) -> float:
        """
        Calibrates the model temperature on validation set via NLL minimization (Platt / Temperature scaling).
        Directly reduces Expected Calibration Error (ECE) and eliminates overconfidence.
        """
        self.model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for g_idx, d_idx, labels, _ in val_loader:
                inputs = self._prepare_batch_inputs(g_idx, d_idx)
                preds = self.model(**inputs)
                all_logits.append(preds["logits"])
                all_labels.append(labels.to(self.device))

        if not all_logits:
            return 1.0

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)

        # Optimize temperature T > 0
        log_temp = torch.nn.Parameter(torch.zeros(1, device=self.device))
        optimizer = torch.optim.LBFGS([log_temp], lr=0.05, max_iter=50)

        def eval_nll():
            optimizer.zero_grad()
            t = torch.exp(log_temp)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits / t, labels)
            loss.backward()
            return loss

        try:
            optimizer.step(eval_nll)
            best_t = float(torch.exp(log_temp).item())
            best_t = max(0.5, min(best_t, 10.0))
        except Exception:
            best_t = 1.0

        if hasattr(self.model, "set_temperature"):
            self.model.set_temperature(best_t)
        elif hasattr(self.model, "temperature"):
            self.model.temperature.data.fill_(best_t)

        if verbose:
            print(f"Optimal temperature calibrated: T = {best_t:.4f}")

        return best_t

    @torch.no_grad()
    def predict_pairs(
        self,
        gene_indices: np.ndarray,
        disease_indices: np.ndarray,
        batch_size: int = 512,
        return_attention: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Runs batch inference on arbitrary gene-disease index pairs."""
        self.model.eval()
        n_pairs = len(gene_indices)
        all_probs = []
        all_strengths = []
        all_eff_strengths = []

        attn_list = [] if return_attention else None

        for start_idx in range(0, n_pairs, batch_size):
            end_idx = min(start_idx + batch_size, n_pairs)
            g_batch = torch.tensor(gene_indices[start_idx:end_idx], dtype=torch.long)
            d_batch = torch.tensor(disease_indices[start_idx:end_idx], dtype=torch.long)

            inputs = self._prepare_batch_inputs(g_batch, d_batch)
            if return_attention:
                inputs["return_attention"] = True

            preds = self.model(**inputs)
            all_probs.append(preds["probs"].detach().cpu().numpy())
            all_strengths.append(preds["strengths"].detach().cpu().numpy())
            if "effective_strengths" in preds:
                all_eff_strengths.append(preds["effective_strengths"].detach().cpu().numpy())
            else:
                all_eff_strengths.append((preds["probs"] * preds["strengths"]).detach().cpu().numpy())

            if return_attention:
                # Capture attention weights for the batch
                attn_dict = {
                    "g2d": preds["attn_g2d"].detach().cpu().numpy(),
                    "d2g": preds["attn_d2g"].detach().cpu().numpy(),
                    "gene_self": preds["attn_gene_self"].detach().cpu().numpy(),
                    "disease_self": preds["attn_disease_self"].detach().cpu().numpy(),
                }
                attn_list.append(attn_dict)

        results = {
            "probs": np.concatenate(all_probs),
            "strengths": np.concatenate(all_strengths),
            "effective_strengths": np.concatenate(all_eff_strengths),
        }
        if return_attention:
            results["attentions"] = attn_list
        return results

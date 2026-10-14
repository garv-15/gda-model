# Mathematical Equations in TransGDA

A complete mathematical reference of all formulas used throughout the TransGDA pipeline, with a 1-line explanation for each.

---

### 1. Graph Construction & Similarity

1. **Cosine Similarity Matrix:**
   $$S_{ij} = \frac{u_i \cdot u_j}{\|u_i\|_2 \|u_j\|_2}$$
   Measures pairwise functional/phenotypic profile alignment between two genes or two diseases.

2. **Top-$K$ Neighborhood Extraction (Self-Loop Excluded):**
   $$\mathcal{N}_k(i) = \operatorname{arg\,topK}_{j \ne i} (S_{ij})$$
   Selects the $K$ most similar distinct neighbors for node $i$ to construct localized graph attention contexts.

3. **External Feature-Based Inductive Projection (Seen vs. Cold-Start):**
   $$h_u = (1 - m_u) E_{\text{ID}}(u) + W_{\text{feat}} x_u$$
   Combines learned ID embeddings for seen nodes with projected external similarity features, zeroing out ID embeddings ($m_u = 1$) for cold-start entities.

---

### 2. Dual Graph-Transformer Encoders

4. **Multi-Head Scaled Dot-Product Self-Attention with Edge Bias:**
   $$\text{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + W_{\text{edge}} S_{\text{nbr}}\right) V$$
   Aggregates neighborhood feature representations scaled by embedding dimension and weighted by similarity edge strengths.

5. **Multi-Head Aggregation:**
   $$\text{MHA}(X) = \left[ \text{head}_1 \,\|\, \dots \,\|\, \text{head}_H \right] W_O$$
   Concatenates projections from $H$ parallel attention heads and maps them back to the model hidden dimension.

6. **Graph Transformer Residual & Normalization Layer:**
   $$\tilde{H} = \operatorname{LayerNorm}(H + \operatorname{Dropout}(\operatorname{MHA}(H)))$$
   Adds a skip-connection around self-attention followed by layer normalization to stabilize deep representations.

7. **Position-Wise Feed-Forward Network (FFN):**
   $$\operatorname{FFN}(x) = W_2 \operatorname{GELU}(W_1 x + b_1) + b_2$$
   Applies non-linear token-level transformations using the Gaussian Error Linear Unit (GELU) activation.

---

### 3. Bilateral Cross-Attention & Interactive Fusion

8. **Disease-to-Gene Cross-Attention:**
   $$h_{g \to d} = \operatorname{softmax}\left(\frac{(W_q^g h_g)(W_k^d H_d)^T}{\sqrt{d_k}}\right) (W_v^d H_d)$$
   Queries the disease neighborhood using the gene representation to identify symptom-specific gene mechanisms.

9. **Gene-to-Disease Cross-Attention:**
   $$h_{d \to g} = \operatorname{softmax}\left(\frac{(W_q^d h_d)(W_k^g H_g)^T}{\sqrt{d_k}}\right) (W_v^g H_g)$$
   Queries the gene neighborhood using the disease representation to identify gene-specific disease manifestations.

10. **Bilinear Interaction:**
    $$z_{\text{bilinear}} = h_g^T W_{\text{bi}} h_d + b_{\text{bi}}$$
    Captures multiplicative second-order compatibility directly between gene and disease representations.

11. **Interactive Fusion Feature Vector:**
    $$h_{\text{fuse}} = \left[ h_g \,\|\, h_d \,\|\, h_{g \to d} \,\|\, h_{d \to g} \,\|\, (h_g \odot h_d) \,\|\, |h_g - h_d| \right]$$
    Concatenates individual states, cross-attended contexts, element-wise products, and absolute differences into a unified interaction vector.

12. **Fused Latent Representation:**
    $$h_{\text{final}} = \operatorname{LayerNorm}\left(\operatorname{GELU}(W_f h_{\text{fuse}} + b_f)\right) + z_{\text{bilinear}}$$
    Projects and normalizes the concatenated features with an additive bilinear interaction shortcut.

---

### 4. Multi-Task Prediction Heads

13. **Association Existence Probability (Classifier Head):**
    $$\hat{p} = \sigma\left(W_{\text{exist}} h_{\text{final}} + b_{\text{exist}}\right) = \frac{1}{1 + e^{-(W_{\text{exist}} h_{\text{final}} + b_{\text{exist}})}}$$
    Maps the latent interaction representation through a Sigmoid function to output binary association probability in $[0, 1]$.

14. **Temperature-Scaled (Calibrated) Probability:**
    $$\hat{p}_{\text{calib}} = \sigma\left(\frac{z_{\text{logit}}}{T}\right)$$
    Rescales logit predictions by an optimized temperature parameter $T > 0$ to eliminate model overconfidence and minimize calibration error.

15. **Association Strength Prediction (Regressor Head):**
    $$\hat{s} = \operatorname{softplus}\left(W_{\text{str}} h_{\text{final}} + b_{\text{str}}\right) = \ln\left(1 + e^{W_{\text{str}} h_{\text{final}} + b_{\text{str}}}\right)$$
    Ensures continuous, strictly non-negative association strength predictions via the smooth softplus activation.

---

### 5. Training Objectives & Loss Functions

16. **Binary Cross-Entropy with Label Smoothing:**
    $$\mathcal{L}_{\text{exist}} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i^* \ln(\hat{p}_i) + (1 - y_i^*) \ln(1 - \hat{p}_i) \right], \quad y_i^* = y_i (1 - \alpha) + 0.5 \alpha$$
    Penalizes incorrect existence classifications while preventing overconfident saturated weights via label smoothing parameter $\alpha$.

17. **Strength Regression Loss (MSE on Positive Edges):**
    $$\mathcal{L}_{\text{strength}} = \frac{1}{|E^+|} \sum_{(i,j) \in E^+} \left( \hat{s}_{ij} - s_{ij} \right)^2$$
    Computes mean squared error between predicted and target continuous association scores exclusively over true observed edges.

18. **Attention Entropy Regularization:**
    $$\mathcal{L}_{\text{entropy}} = -\frac{1}{H} \sum_{h=1}^H \sum_{j} A_{hj} \ln\left(A_{hj} + \varepsilon\right)$$
    Penalizes degenerate/flat attention weights to encourage sharp, informative neighbor selection.

19. **Total Joint Multi-Task Loss:**
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{exist}} + \lambda_{\text{str}} \mathcal{L}_{\text{strength}} + \lambda_{\text{attn}} \mathcal{L}_{\text{entropy}}$$
    Jointly balances classification accuracy, strength quantification, and attention interpretability.

---

### 6. Interpretability, Causal Validation & Discovery

20. **Integrated Gradients Token Attribution:**
    $$\operatorname{Attr}(x_i) = (x_i - x_i^0) \times \frac{1}{M} \sum_{m=1}^M \left. \frac{\partial f}{\partial x_i} \right|_{x_i^0 + \frac{m}{M}(x_i - x_i^0)}$$
    Computes path-integrated gradients from an uninformative zero baseline to quantify the exact causal importance of neighbor tokens.

21. **In-Silico Target Knockout Effect:**
    $$\Delta s_{\text{target}} = \hat{s}_{\text{unperturbed}} - \hat{s}_{\text{knockout}}$$
    Quantifies the reduction in predicted association strength when the target gene or disease is computationally deleted.

22. **Relative Epistasis Score (Synthetic Lethality):**
    $$\epsilon_{\text{rel}} = \frac{\Delta s_{AB} - (\Delta s_A + \Delta s_B)}{\max(|\Delta s_A|, |\Delta s_B|) + \varepsilon}$$
    Measures non-linear interaction between two knocked-out genes to detect epistatic masking versus synthetic lethality.

23. **Hypergeometric Over-Representation Analysis (ORA):**
    $$P(X \ge k) = \sum_{x=k}^{\min(n, K)} \frac{\binom{K}{x} \binom{N - K}{n - x}}{\binom{N}{n}}$$
    Calculates the exact statistical probability of observing $k$ or more driver genes overlapping a known pathway of size $K$ by chance.

24. **Benjamini-Hochberg False Discovery Rate (FDR):**
    $$q_{(i)} = \min_{j \ge i} \left( \frac{M}{j} p_{(j)} \right)$$
    Controls the expected proportion of false positives across multiple hypothesis tests over biological pathways.

25. **Biological Plausibility Index (BPI):**
    $$\operatorname{BPI} = \hat{p} \times \min\left(10, -\log_{10}(q_{\text{val}} + 10^{-10})\right) \times \left(1 + \frac{\Delta s_{\text{target}}}{\hat{s} + \varepsilon}\right)$$
    Integrates predicted probability, statistical pathway significance, and causal knockout necessity into a composite credibility score.

26. **Novel Discovery Priority Score:**
    $$\text{Score}_{\text{discovery}} = \hat{p} \times \hat{s}$$
    Ranks unobserved gene–disease pairs by joint probability of true association and magnitude of biological effect.

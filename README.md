# TransGDA: Interpretable Transformer-based Gene–Disease Association & Strength Prediction

**TransGDA** is a model designed to predict gene–disease associations and quantify continuous association strength by integrating gene–gene similarity networks, disease–disease similarity networks, and bipartite interaction networks.

TransGDA is built with:
- **Dual Graph-Transformer Encoders**: neighborhood self-attention over functional gene modules and disease phenotypes.
- **Bilateral Cross-Attention**: captures disease and gene specific interactions
- **Multi-Task Objective**: Jointly optimizes Association Existence (classification) and Association Strength (regression).
- **Interpretability Engine**: Extracts and visualizes cross attention weights to expose biological driver genes and disease facets.


## Reproducing Results

### Setup

#### Creating a virtual environment 
```bash
python -m venv .venv
```
#### activating the virtual environment 
```bash
source .venv/bin/activate
```
#### installing dependencies
```bash
pip install -r requirements.txt
```


### 1. Run Unit Tests
```bash
./.venv/bin/python -m unittest discover tests
```

### 2. Run Benchmarks across CV Splits
Compare TransGDA against biological baselines (Degree Prior, Neighborhood CF, Matrix Factorization, Static MLP, and Ablations) across **Edge (Transductive)**, **Gene-Held-Out (Inductive)**, and **Disease-Held-Out (Inductive)** splits:
```bash
./.venv/bin/python scripts/run_benchmark.py --split edge --epochs 12
./.venv/bin/python scripts/run_benchmark.py --split gene --epochs 12
./.venv/bin/python scripts/run_benchmark.py --split disease --epochs 12
```

### 3. Run Data Leakage Audit
Quantify the performance inflation caused when using precomputed global similarity matrices vs. strictly leakage-free training matrices:
```bash
./.venv/bin/python scripts/run_leakage_audit.py --epochs 10
```

### 4. Generate Attention Interpretability Visualizations
Extract attention maps and plot biological driver gene/disease explanations for clinical case studies:
```bash
./.venv/bin/python scripts/run_interpret.py --epochs 10
```

### 5. Screen for Novel Gene–Disease Hypotheses
Discover high-confidence unobserved gene–disease relationships:
```bash
./.venv/bin/python scripts/run_discoveries.py --epochs 12 --top_n 25
```

### 6. In-Silico Virtual Knockouts & Statistical Pathway Validation
Execute counterfactual gene/disease knockouts, detect second-order synthetic lethality, and run hypergeometric Over-Representation Analysis with Benjamini-Hochberg FDR correction:
```bash
./.venv/bin/python scripts/run_causal_validation.py --epochs 10 --top_n_discoveries 15
```

### 7. Export Model Artifacts
Export trained model weights, configurations, canonical entity mappings, and graph neighborhoods for production serving:
```bash
./.venv/bin/python export.py --output_dir results/saved_model --epochs 10
```

## Running the model 
Run Prediction on Any Gene and Disease Pair

Pass --gene with any gene symbol and --disease with any disease name:

```bash
# ex 1: Known risk gene for Type 2 Diabetes
python predict.py --gene TCF7L2 --disease "type 2 diabetes mellitus"
# ex 2: Neurological gene for Parkinson's Disease (with plot generation), output will be stored as image in root
python predict.py --gene UNC13C --disease "parkinson's disease" --plot
```

Searching for Valid Gene or Disease Names
```bash
# Search for genes matching a pattern (e.g. BRCA, TP53, IL)
python predict.py --search_gene BRCA
# Search for diseases matching a keyword (e.g., leukemia, diabetes, cancer)
python predict.py --search_disease leukemia
```


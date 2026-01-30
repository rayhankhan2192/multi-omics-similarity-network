# MOGONET: Multi-Omics Graph Convolutional Networks

**Objective:** Develop a robust Multi-Omics Graph Convolutional Network (MOGONET) for biomedical classification tasks. The model integrates multi-omics data using Graph Convolutional Networks (GCN) for view-specific feature learning and a View Correlation Discovery Network (VCDN) for effective multi-omics data fusion.

**Background:** Integrating diverse biological data types (e.g., mRNA, methylation, miRNA) is critical for accurate disease classification. MOGONET leverages the power of GCNs to capture relationships within each data "view" and uses a specialized tensor-product-based fusion layer (VCDN) to understand the correlations between different views, leading to better predictive performance than single-view analysis.

---

# 📁 Project Structure

```text
.
├── main_mogonet.py        # Main entry point; defines hyperparameters and starts training
├── models.py              # Architecture definitions (GCN, Classifier, VCDN)
├── train_test.py          # Core logic for training loops, testing, and data preparation
├── utils.py               # Utility functions for adjacency matrices and similarity metrics
├── model_evaluation.py    # Module for generating and saving performance plots
├── ROSMAP/                # Data directory (example dataset: Alzheimer's)
├── BRCA/                  # Data directory (example dataset: Breast Cancer)
├── plots/                 # Output directory for generated training/testing curves
└── README.md              # Project documentation
```
# 🛠️ Getting Started
## 1. Set Up Environment

python -m venv venv

### Windows
```bash
.\venv\Scripts\activate
```
### macOS/Linux
```bash
source venv/bin/activate
```
## 2. Install Dependencies
```bash
pip install requirements.txt
```
## 3. Prepare Data

Ensure your data folders (e.g., ROSMAP) contain the following .csv files:

1. labels_tr.csv / labels_te.csv: Training and testing labels.

2. 1_tr.csv, 2_tr.csv, etc.: Feature matrices for each omics view.

## 4. Run the Model
```bash
python main_mogonet.py
```

## 🧪 How the Framework Works
### Step 1: Data Preparation & Graph Construction
Normalization: Data is normalized to ensure stable training.

Similarity Metrics: Supports cosine, RBF, and hybrid similarity to construct patient-to-patient graphs.

Adaptive Adjacency: Generates adjacency matrices based on k-nearest neighbors using calculated parameters.

### Step 2: Multi-View Feature Learning
GCN Encoders: Each omics view is processed by a dedicated 3-layer GCN (GCN_E) to extract high-level structural features.

View Classifiers: Individual classifiers (Classifier_1) generate initial predictions for each view.

### Step 3: VCDN Fusion
Cross-View Correlation: The VCDN module takes predictions from all views and computes a cross-view tensor product.

Final Classification: A multi-layer perceptron (MLP) processes the fused tensor to produce the final classification output.

### Step 4: Evaluation
Pretraining: GCNs are pretrained first to stabilize view-specific features before training the full model.

Performance Metrics: Calculates Accuracy, F1-score, and AUC (for binary tasks).

Visualization: model_evaluation.py automatically generates loss and accuracy curves in the /plots folder.

## 🧩 Design Choices
Metric: hybrid_similarity_torch — Combines Cosine and RBF metrics to capture both linear and non-linear relationships.

Initialization: xavier_init — Used for all linear and GCN layers to ensure proper weight scaling.

Hardware: Automatic CUDA detection for GPU acceleration.

Optimization: Uses the Adam optimizer with separate learning rates for encoders and classifiers.

## 🧱 Tech Stack
Component,Choice(s)
Language,Python
Deep Learning,PyTorch
Data Handling,"NumPy, Pandas"
Metrics,Scikit-learn
Visualization,Matplotlib
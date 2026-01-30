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

# 🛠️ Getting Started
## 1. Set Up Environment

python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
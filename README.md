# Machine Learning Algorithms — From Scratch Implementation

> A comprehensive collection of core machine learning algorithms implemented **from scratch** in Python, benchmarked across multiple real-world datasets. Each model is evaluated on accuracy, F1-score, precision, recall, confusion matrix, ROC/AUC, and computation time.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Algorithms Implemented](#algorithms-implemented)
- [Datasets Used](#datasets-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)

---

## Overview

This project implements five families of machine learning algorithms entirely from scratch — without relying on high-level sklearn model APIs — to deeply understand the mechanics behind each method. Every algorithm is then tested on multiple datasets and compared using a consistent set of evaluation metrics.

| Algorithm Family | Datasets Tested |
|---|---|
| K-Nearest Neighbors (KNN) | Digital Digits, Heart Disease, Rice Grains |
| Decision Tree | Digital Digits, Heart Disease, Rice Grains |
| Multinomial Naive Bayes | Custom Q1–Q4, Q6 problems |
| Random Forest | Credit Approval, Oxford's Disease, Digital Digits, Heart Disease, Rice Grains, Raisins, Titanic, WDBC |
| Neural Network | Loan, Raisins, Titanic, WDBC, Digital Digits, Oxford's Disease, Rice Grains, Credit Approval |

---

## Repository Structure

```
Machine-Leaning-Algo-impl/
│
├── Final_Project/
│   ├── Neual_Network_digital_digits.py
│   ├── Neural_Network_Oxford's_Disease.py
│   ├── Neural_Network_for_rice_grains.py
│   ├── Neural_network_credit_approval.py
│   ├── Random_forest_credit_approval.py
│   ├── Random_forest_for_Oxford's_Disease.py
│   ├── Random_forest_for_digital_digits.py
│   ├── Random_forest_for_heart_disease.py
│   └── Random_forest_for_rice_grains.py
│
├── KNN and Decision Tree/
│   ├── DecisionTree.py
│   ├── Gini_impurity.py
│   ├── Gini_impurity_withMaxDepth.py
│   ├── Knn Algorithm.py
│   └── readFile.txt
│
├── Multinomial Naive Bayes/
│   ├── Q1.py
│   ├── Q2.py
│   ├── Q3.py
│   ├── Q4.py
│   ├── Q6.py
│   └── Utils.py
│
├── Naive Bayes/
│   ├── started_code/
│   ├── calculate_accuracy.py
│   ├── computation.py
│   ├── main.py
│   ├── model_sampler.py
│   ├── naive_bayes.py
│   ├── test.py
│   └── train_class.py
│
├── Neural Network/
│   ├── Basic_Neural_network.py
│   ├── Neural_network_Q7.py
│   ├── Neural_network_for_loan.py
│   ├── Neural_network_for_raisins.py
│   ├── Neural_network_for_titanic.py
│   ├── Neural_network_for_wdbc.py
│   ├── latest_neural_network.py
│   └── test.py
│
├── Random_Forest/
│   ├── Random_Forest_old.py
│   ├── Random_forest_for_raisin_datas.py
│   ├── Random_forest_for_titanic_data.py
│   └── Random_forest_for_wdbc_datas.py
│
└── README.md
```

---

## Algorithms Implemented

### K-Nearest Neighbors (KNN)
- Distance-based classification using Euclidean distance
- Custom k-neighbor search without sklearn
- Tested with varying values of k

### Decision Tree
- Recursive binary splitting built from scratch
- **Gini impurity** as the splitting criterion
- **Max depth** control to prevent overfitting (`Gini_impurity_withMaxDepth.py`)
- Supports both pure and depth-limited trees

### Multinomial Naive Bayes
- Full probabilistic implementation with Laplace smoothing
- Utility functions separated into `Utils.py`
- Multiple question-driven experiments (Q1–Q4, Q6)

### Naive Bayes (Gaussian)
- Full training pipeline: `train_class.py` → `naive_bayes.py` → `test.py`
- Accuracy calculation via `calculate_accuracy.py`
- Model sampling via `model_sampler.py`
- Computation time tracking via `computation.py`

### Random Forest
- Ensemble of decision trees with bootstrap sampling (bagging)
- Random feature subsampling at each split
- Applied across eight datasets: Credit Approval, Oxford's Disease, Digital Digits, Heart Disease, Rice Grains, Raisins, Titanic, WDBC

### Neural Network
- Fully connected feedforward network built from scratch
- Forward pass, backpropagation, and weight update implemented manually
- Activation functions: Sigmoid / ReLU
- Applied across: Loan, Raisins, Titanic, WDBC, Digital Digits, Oxford's Disease, Rice Grains, Credit Approval

---

## Datasets Used

| Dataset | Task | Features |
|---------|------|----------|
| Digital Digits | Multi-class classification | Pixel intensity values |
| Oxford's Disease | Binary / multi-class classification | Clinical features |
| Rice Grains | Binary classification | Morphological features |
| Heart Disease | Binary classification | Clinical / demographic features |
| Credit Approval | Binary classification | Financial features |
| Raisins | Binary classification | Shape & texture features |
| Titanic | Binary classification (survival) | Passenger demographic features |
| WDBC (Breast Cancer) | Binary classification | Cell nucleus measurements |
| Loan | Binary classification | Financial / demographic features |

---

## Evaluation Metrics

Every model is evaluated using the following metrics, computed on the test set:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions / total predictions |
| **Precision** | True positives / (true positives + false positives) |
| **Recall** | True positives / (true positives + false negatives) |
| **F1-Score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | Full breakdown of predicted vs. actual classes |
| **ROC / AUC** | Receiver Operating Characteristic curve and area under it |
| **Computation Time** | Wall-clock time for training and inference |

---

## Setup & Installation

**Requirements:** Python 3.8+

```bash
# Clone the repository
git clone https://github.com/swanky-rt/Machine-Leaning-Algo-impl.git
cd Machine-Leaning-Algo-impl
```

**Install dependencies:**
```bash
pip install numpy pandas matplotlib scikit-learn
```

> Note: `scikit-learn` is used **only** for data loading, train/test splitting, and metric computation — not for the algorithm implementations themselves.

---

## How to Run

Each script is self-contained. Navigate to the relevant folder and run directly.

**KNN or Decision Tree:**
```bash
cd "KNN and Decision Tree"
python "Knn Algorithm.py"
python DecisionTree.py
python Gini_impurity_withMaxDepth.py
```

**Naive Bayes:**
```bash
cd "Naive Bayes"
python main.py
```

**Multinomial Naive Bayes:**
```bash
cd "Multinomial Naive Bayes"
python Q1.py    # or Q2.py, Q3.py, Q4.py, Q6.py
```

**Neural Network (e.g., Titanic dataset):**
```bash
cd "Neural Network"
python Neural_network_for_titanic.py
```

**Random Forest (e.g., WDBC dataset):**
```bash
cd Random_Forest
python Random_forest_for_wdbc_datas.py
```

**Final Project experiments:**
```bash
cd Final_Project
python Random_forest_for_rice_grains.py
python Neual_Network_digital_digits.py
```

---

## Results Summary

Each script prints a full evaluation report to the console. Example output format:

```
Dataset       : Titanic
Algorithm     : Neural Network
Accuracy      : 82.4%
Precision     : 0.81
Recall        : 0.79
F1-Score      : 0.80
AUC           : 0.87
Training Time : 1.23s
```

Confusion matrices and ROC curves are plotted using `matplotlib` for visual analysis of each model's performance.

---

## Key Design Decisions

- **No black-box models** — every algorithm (forward pass, splitting criterion, distance function, probability estimation) is hand-coded to reinforce understanding
- **Consistent evaluation pipeline** — the same metrics are applied across all algorithms and datasets for fair comparison
- **Modular structure** — utility functions (data loading, accuracy, computation time) are separated from model logic for reusability
- **Dataset variety** — binary and multi-class problems, tabular and image-derived data, balanced and imbalanced class distributions

---

## Author

**swanky-rt** — [github.com/swanky-rt](https://github.com/swanky-rt)

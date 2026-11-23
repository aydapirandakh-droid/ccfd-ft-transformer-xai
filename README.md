# Credit Card Fraud Detection Using FT-Transformer and Explainable AI

## Project Overview

This project implements a Feature Tokenizer Transformer (FT-Transformer) for highly imbalanced credit card fraud detection.

Beyond predictive performance, the project investigates model interpretability through inter-feature attention analysis and quantitative evaluation of explanation faithfulness and stability.

The main objective is to explore how Transformer-based architectures can learn feature interactions and how their attention patterns behave under feature removal and input perturbations.

## Dataset

The project uses the Credit Card Fraud Detection dataset from Kaggle (ULB Machine Learning Group).

- Total transactions: 284,807
- Normal transactions: 284,315
- Fraudulent transactions: 492
- Fraud prevalence: approximately 0.173%
- Input features: 30
- Target variable: `Class`

The dataset contains anonymized PCA-transformed features (V1–V28), transaction time, and transaction amount.

## Methodology

### 1. Data Preprocessing

- Separates input features and target labels.
- Applies stratified train-validation-test splitting (70/15/15).
- Uses StandardScaler fitted exclusively on the training set.
- Preserves the original class distribution across the partitions.

### 2. FT-Transformer Architecture

The model uses a feature tokenizer to transform individual numerical features into learnable embeddings.

Architecture configuration:

| Component | Configuration |
|---|---|
| Feature Embedding Dimension | 32 |
| Attention Heads | 4 |
| Transformer Encoder Layers | 3 |
| Feedforward Dimension | 128 |
| Dropout | 0.2 |
| Pooling | Global Average Pooling |
| Classification Head | 32 → 16 → 1 |

### 3. Model Training

- Framework: PyTorch
- Loss Function: Weighted Binary Cross-Entropy with Logits
- Optimizer: Adam
- Learning Rate: 0.001
- Weight Decay: 0.00001
- Batch Size: 1024
- Training Epochs: 10
- Hardware: CUDA-enabled GPU

The best checkpoint is selected based on validation PR-AUC.

## Model Performance

### Test Set Results

| Metric | Score |
|---|---:|
| PR-AUC | 0.7483 |
| ROC-AUC | 0.9726 |

### Decision Threshold Optimization

The decision threshold is selected using the validation set by maximizing the F1-score.

| Metric | Default Threshold (0.5) | Optimized Threshold |
|---|---:|---:|
| Threshold | 0.5000 | 0.9958 |
| Precision | 6.96% | 84.85% |
| Recall | 86.49% | 75.68% |
| F1-score | 12.88% | 80.00% |
| False Positives | 856 | 10 |
| False Negatives | 10 | 18 |

The optimized threshold substantially reduces false positives while maintaining a strong fraud detection rate.

## Explainable AI (XAI)

### 1. Inter-Feature Attention Analysis

Attention weights from the final Transformer encoder layer are extracted to investigate relationships between input features.

The analysis compares mean attention distributions for fraudulent and normal transactions.

Top attended features in fraudulent transactions:

- V14
- V4
- V26
- V5
- V10

Attention weights are used as an exploratory feature-interaction signal and are not treated as standalone explanations.

### 2. Quantitative Faithfulness Evaluation

A progressive feature deletion experiment evaluates how model predictions change when highly attended features are replaced with normal-transaction median values.

The attention-guided deletion results are compared against random feature deletion.

| Features Removed | Attention-Guided Probability Drop | Random Probability Drop |
|---|---:|---:|
| 1 | 0.0081 | 0.0023 |
| 2 | 0.0140 | 0.0034 |
| 3 | 0.0312 | 0.0069 |
| 5 | 0.0677 | 0.0111 |
| 8 | 0.0874 | 0.0292 |
| 10 | 0.0902 | 0.0463 |
| 15 | 0.1202 | 0.1050 |

The results show larger prediction changes for attention-guided deletion than for random deletion in the evaluated settings.

### 3. Explanation Stability Evaluation

Explanation stability is measured by adding Gaussian noise (σ = 0.05) to fraudulent test instances and comparing feature-importance rankings.

| Metric | Result |
|---|---:|
| Spearman Rank Correlation | 0.9930 |
| Jaccard Similarity @ Top-5 | 0.9595 |
| Jaccard Similarity @ Top-10 | 0.9631 |

These results indicate high ranking consistency under the specified perturbation experiment.

## Technologies

- Python
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Google Colab

## Repository Structure

```text
ccfd-ft-transformer-xai/
│
├── CCFD.FT_Transformer.XAI.ipynb
└── README.md
```

## How to Run

1. Open the notebook in Google Colab.
2. Install the required Python dependencies.
3. Download the Credit Card Fraud Detection dataset.
4. Run the notebook cells sequentially.
5. Review model performance, attention visualizations, and quantitative XAI results.

A CUDA-enabled GPU is recommended for training.

## Limitations

- The dataset is highly imbalanced and contains anonymized features.
- The train-validation-test partition is stratified and randomized, not chronological.
- Attention weights are exploratory interpretability signals and do not independently establish causal feature importance.
- Faithfulness and stability results are specific to the implemented perturbation and deletion experiments.

This project is intended for educational and research purposes.

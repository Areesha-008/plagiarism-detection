# Semantic-Based Plagiarism Detection Using Deep NLP Models

> A comparative study of TF-IDF, Siamese Bi-LSTM, and Sentence-BERT for detecting paraphrased, obfuscated, and translated plagiarism.

**Authors:** Musa Rashid (23F-0039) · Areesha Saqib (23F-0038)  
**Institution:** National University of Computer and Emerging Sciences, Faisalabad, Pakistan

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Models](#models)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Discussion](#discussion)
- [References](#references)

---

## Overview

Traditional plagiarism detectors rely on surface-level string matching (n-gram overlap, TF-IDF cosine similarity), which works well for verbatim copying but fails entirely when a plagiarist paraphrases, restructures, or translates source material. Modern writing assistants and machine-translation services have made these obfuscation techniques trivially accessible.

This project investigates whether progressively more semantic text representations produce correspondingly larger gains on the plagiarism detection task. We implement and compare three models across the representational spectrum:

| # | Model | Representation |
|---|-------|----------------|
| 1 | TF-IDF + Cosine Similarity | Lexical (sparse bag-of-words) |
| 2 | Siamese Bi-LSTM + GloVe | Distributional (static embeddings) |
| 3 | Fine-tuned Sentence-BERT (MiniLM-L6-v2) | Contextual (transformer embeddings) |

All three models are trained on the same combined dataset and evaluated under an identical pipeline, enabling a direct apples-to-apples comparison.

---

## Key Results

### Overall Performance (Test Set)

| Metric | TF-IDF | Bi-LSTM | SBERT |
|--------|--------|---------|-------|
| **PAN F1** | 0.966 | 0.996 | 0.996 |
| **PAN ROC AUC** | 0.982 | 1.000 | 1.000 |
| **Quora F1** | 0.605 | 0.812 | 0.807 |
| **Quora ROC AUC** | 0.682 | 0.929 | 0.924 |

### Recall by Obfuscation Level (PAN Test Split)

| Obfuscation Type | TF-IDF | Bi-LSTM | SBERT |
|------------------|--------|---------|-------|
| None (verbatim) | 1.000 | 1.000 | 1.000 |
| Low | 1.000 | 1.000 | 1.000 |
| High | 0.994 | 0.992 | **0.999** |
| Artificial | 0.761 | **0.985** | 0.974 |
| **Translation** | **0.476** | **0.990** | **0.976** |

> **Most striking finding:** TF-IDF detects fewer than half of all translated plagiarism cases (recall = 0.476), while both deep models recover nearly all of them (≥ 0.97). This 50+ point gap confirms that lexical methods cannot succeed on cross-lingual plagiarism in principle.

---

## Project Structure

```
.
├── data/
│   ├── pan_pc11/               # PAN Plagiarism Corpus 2011 (parsed pairs)
│   └── quora_pairs/            # Quora Question Pairs
├── models/
│   ├── model1_tfidf.py         # TF-IDF + Cosine Similarity baseline
│   ├── model2_bilstm.py        # Siamese Bi-LSTM with GloVe embeddings
│   └── model3_sbert.py         # Fine-tuned Sentence-BERT (MiniLM-L6-v2)
├── embeddings/
│   └── glove.6B.300d.txt       # GloVe pre-trained embeddings (not tracked)
├── checkpoints/
│   ├── bilstm_best.pt          # Best Bi-LSTM checkpoint (epoch 8)
│   └── sbert_finetuned/        # Fine-tuned SBERT weights
├── figures/
│   ├── roc_combined.png
│   ├── confusion_matrices.png
│   ├── similarity_distributions.png
│   └── model2_training_curves.png
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model1_tfidf.ipynb
│   ├── 03_model2_bilstm.ipynb
│   └── 04_model3_sbert.ipynb
├── requirements.txt
└── README.md
```

---

## Models

### Model 1 — TF-IDF + Cosine Similarity

A classical lexical baseline. A single `TfidfVectorizer` is fitted on all training data with:
- Unigram + bigram features
- Sublinear TF scaling
- English stop-word removal
- Vocabulary capped at 100,000 features

Each passage pair is scored by cosine similarity between their L2-normalised TF-IDF vectors. A decision threshold is tuned independently on the PAN and Quora validation sets by maximising F1.

**Pros:** Fast, interpretable, CPU-only, no training required.  
**Cons:** Collapses on paraphrase and translation — misses ~50% of translated cases.

---

### Model 2 — Siamese Bi-LSTM

A Siamese neural network where both input passages share a common encoder:

- **Embedding layer:** Frozen 300-dimensional GloVe vectors (6B-token corpus)
- **Encoder:** 2-layer bidirectional LSTM with 256 hidden units per direction
- **Pooling:** Mean pooling over non-padding positions → 512-dim sentence vector
- **Classifier:** Dense layers (512 → 128 → 1) with ReLU activation and 0.3 dropout
- **Feature vector:** `[u, v, |u − v|, u ⊙ v]` following Mueller & Thyagarajan (2016)
- **Loss:** `BCEWithLogitsLoss` with `pos_weight=0.25` (to handle 4:1 class imbalance)
- **Optimiser:** Adam (lr = 1e-3) with gradient clipping at 1.0

Training runs for 10 epochs; the best checkpoint (epoch 8, val F1 = 0.876) is retained.

---

### Model 3 — Fine-tuned Sentence-BERT

Built on top of `sentence-transformers/all-MiniLM-L6-v2`, a 22M-parameter distilled BERT encoder:

- **Architecture:** 6-layer transformer, hidden size 384, producing 384-dim sentence embeddings
- **Tokenisation:** WordPiece; PAN passages truncated to 256 tokens
- **Fine-tuning loss:** `CosineSimilarityLoss` — aligns embeddings of plagiarism pairs and separates non-plagiarism pairs
- **Training:** 4 epochs, batch size 128, AdamW optimiser, 10% warm-up schedule
- **Inference:** Each document is encoded once into a fixed vector; comparisons are done by cosine similarity — enabling efficient large-scale indexing

---

## Datasets

### PAN Plagiarism Corpus 2011 (PAN-PC-11)

- **Pairs extracted:** 76,661 (suspicious, source) passage pairs
- **Metadata:** Plagiarism type and obfuscation level (`none`, `low`, `high`, `artificial`, `translation`)
- **Split:** 70 / 15 / 15 (train / validation / test) at **document level** with stratified shuffle splitting to prevent leakage
- **Negatives:** Hard negatives constructed by pairing non-plagiarism passages with source documents from different cases

### Quora Question Pairs

- **Pairs:** ~400,000 short text pairs labelled as duplicate / non-duplicate by human annotators
- **Split:** Kaggle's existing train/test files retained; 10% of training rows held out for validation

The two datasets cover complementary regimes — long machine-generated obfuscation (PAN) and short human-written paraphrases (Quora) — making the combined benchmark more robust.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/plagiarism-detection.git
cd plagiarism-detection

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0
sentence-transformers
scikit-learn>=1.0
numpy
pandas
tqdm
```

### GloVe Embeddings (for Model 2)

Download the 6B token, 300-dimensional GloVe vectors and place them in `embeddings/`:

```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d embeddings/
```

---

## Usage

### Train & Evaluate All Models

```bash
# Model 1 — TF-IDF baseline
python models/model1_tfidf.py

# Model 2 — Siamese Bi-LSTM
python models/model2_bilstm.py

# Model 3 — Fine-tuned Sentence-BERT
python models/model3_sbert.py
```

### Inference on a Single Pair

```python
from models.model3_sbert import load_model, predict

model = load_model("checkpoints/sbert_finetuned/")
score = predict(model, "The cat sat on the mat.", "A feline rested upon the rug.")
print(f"Plagiarism probability: {score:.3f}")
```

---

## Evaluation

All models are evaluated on held-out test splits with the following metrics:
- **Accuracy, Precision, Recall, F1-score, ROC AUC**
- **Per-obfuscation recall breakdown** (PAN only) — isolates model performance by difficulty level

Decision thresholds are selected on the validation split by maximising F1 over a grid from 0.05 to 0.95 (step = 0.01). Thresholds are tuned separately for PAN and Quora, since the two datasets exhibit different similarity distributions.

All experiments were conducted on **Google Colab** with a single **NVIDIA A100 GPU**. Random seeds were fixed at 42 across NumPy, PyTorch, and dataset shuffling.

---

## Discussion

**Why does TF-IDF fail on Quora?** Quora questions average around 10 words, leaving little room for rare vocabulary. Many semantically identical pairs share only stop words and one or two content terms. The model resorts to a very low threshold (0.17), yielding high recall (0.94) but very poor precision (0.44). This is a representation problem — not a threshold or feature-weighting problem.

**Why does the Bi-LSTM handle translation so well?** Pretrained GloVe embeddings place semantically related tokens near each other in vector space, including translations of the same word in many cases. The network learns that plagiarism pairs share GloVe-space geometry even when their surface forms share no vocabulary.

**Why doesn't SBERT beat Bi-LSTM by more?** Three likely factors:
1. PAN-PC-11 is a constrained dataset — once the basic paraphrase signal is captured, headroom is small.
2. The SBERT run used 256-token truncation and effective early stopping at epoch 2 due to session interruptions.
3. MiniLM-L6 is a small distilled model — upgrading to `all-mpnet-base-v2` would be expected to yield further gains.

**Deployment consideration:** SBERT's single-pass encoding is its most important practical advantage. Each document is encoded once into a fixed vector and can then be compared against millions of others by cosine similarity (inner product search). The Bi-LSTM requires a forward pass for every new pair. For a production plagiarism detector with a large source-document index, this scaling characteristic is likely more important than the small F1 differences observed on the benchmark.

---

## Future Work

- Train SBERT for more epochs with a heavier base model (e.g., `all-mpnet-base-v2`) to test whether contextual encoders can open a larger gap over the Bi-LSTM on PAN.
- Evaluate on the **intrinsic detection** portion of PAN — where no source document is available — using either stylometric features or a self-similarity formulation.
- Explore **multilingual SBERT variants** for principled cross-lingual plagiarism detection beyond what GloVe alignment provides.

---

## References

1. Mikolov et al., "Distributed representations of words and phrases," NeurIPS 2013.
2. Pennington et al., "GloVe: Global vectors for word representation," EMNLP 2014.
3. Devlin et al., "BERT: Pre-training of deep bidirectional transformers," NAACL 2019.
4. Reimers & Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," EMNLP-IJCNLP 2019.
5. Potthast et al., "Overview of the 3rd international competition on plagiarism detection," CLEF 2011.
6. Chen et al., "Quora question pairs," Quora Inc., 2017.
7. Mueller & Thyagarajan, "Siamese recurrent architectures for learning sentence similarity," AAAI 2016.
8. Wang et al., "MiniLM: Deep self-attention distillation for task-agnostic compression," NeurIPS 2020.

---

## Acknowledgements

<<<<<<< HEAD
We thank the National University of Computer and Emerging Sciences for providing computational resources, and the maintainers of the PAN-PC-11 corpus and the Quora Question Pairs dataset. We also acknowledge the open-source contributors behind PyTorch, scikit-learn, the Hugging Face `sentence-transformers` library, and the GloVe project.
=======
We thank the National University of Computer and Emerging Sciences for providing computational resources, and the maintainers of the PAN-PC-11 corpus and the Quora Question Pairs dataset. We also acknowledge the open-source contributors behind PyTorch, scikit-learn, the Hugging Face `sentence-transformers` library, and the GloVe project.
>>>>>>> 38bd3da65bc6219ef3afb647c7a37246c579239a

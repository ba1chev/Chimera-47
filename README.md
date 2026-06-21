# Chimera-47

Eight-class Windows malware family classifier built on top of behavioral API call traces. Three classical machine-learning models are trained on the same data pipeline and compared head-to-head, so the project doubles as a small empirical study of which inductive bias actually helps on short, vocabulary-bounded API sequences.

## Dataset — MAL-API-2019

[MAL-API-2019](https://github.com/ocatak/malware_api_class) is a public corpus of 7,107 Windows malware samples whose dynamic execution was traced inside Cuckoo Sandbox. Each sample is reduced to a sequence of Windows API calls (≈278 unique tokens across the corpus) and tagged with one of eight families: `Trojan`, `Backdoor`, `Downloader`, `Worms`, `Spyware`, `Adware`, `Dropper`, `Virus`.

The point of the dataset is to study malware *behaviorally*, not by static bytes — two binaries from the same family will use overlapping API call patterns even when their on-disk hashes differ. Class distribution is imbalanced (Trojan dominates, Adware is rare), which is why every model in this project is scored with **macro-F1** rather than accuracy.

## The three compared models

All three consume the exact same stratified 80/20 split (seed `47`) and are tuned on the same stratified 5-fold CV grid, so the comparison is apples-to-apples.

| Model | What it assumes about the data | Final macro-F1 |
|---|---|---|
| **One-vs-One + 28 linear SVMs** | Classes are linearly separable in TF-IDF space; each pair of families gets its own decision boundary. C(8,2)=28 binary SVMs vote, with margin used for tiebreaks. | **0.5879** (C=10) |
| **Multinomial Logistic Regression** | One joint softmax over all 8 classes — a single linear model with 8 weight rows, optimized natively (no OvO scaffolding). | **0.5720** (C=10) |
| **First-order Markov Chain Classifier** | API calls form a sequence with local dependencies; one transition matrix per family, classify by per-class log-likelihood. Laplace-smoothed, with an UNK index for unseen tokens. | **0.3669** (α=0.001) |

The Markov classifier is kept in the comparison even though it loses — that's the scientific finding. First-order transitions are too local to capture family-level structure once TF-IDF has already done most of the heavy lifting for the linear models.

## How the pipeline works

All three models share one identical protocol — same split, same CV folds, same metric — so the head-to-head numbers are apples-to-apples.

```
  raw .txt + .csv
        │
        ▼
  Dataset (7,107 traces, int labels)
        │
        ▼
  Stratified 80/20 split  (seed 47)
   ├─► X_train/y_train (5,685)
   └─► X_test/y_test   (1,422)  ← locked until step 7
        │
        ▼
  TF-IDF fit on train, transform test    [SVM + MNLR]
  raw token sequences                     [Markov]
        │
        ▼
  5-fold stratified CV on train
  grid: C ∈ {.01,.1,.5,1,2,5,10}  /  α ∈ {.001,.01,.1,1,10}
        │
        ▼
  Refit best hyperparam on full train
        │
        ▼
  Predict on test → macro-F1 + confusion matrix
        │
        ▼
  Side-by-side comparison
```

- **Provision** → `.data/` cache (download + unzip on first run)
- **Load** → integer-encode labels via `np.unique`
- **Split** → stratified, locked test set
- **Vectorise** → TF-IDF (5000 features) for linear models; raw text for Markov
- **CV grid** → 5 folds × 7 C values = 35 OvO fits per sweep
- **Refit + evaluate** → honest test-set macro-F1

## Directory tree

```
Chimera-47/
├── notebooks/
│   ├── 00_theoretical.ipynb        # Math derivations: SVM dual, OvO voting, MNLR softmax, Markov likelihood
│   └── 01_experiment.ipynb         # End-to-end pipeline + 3-model comparison on MAL-API-2019
├── source/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── api_fetcher/            # Download + unzip MAL-API-2019 from the upstream repo
│   │   │   ├── __init__.py
│   │   │   ├── api_fetcher.py
│   │   │   ├── http_api_fetcher.py
│   │   │   ├── archive_extractor.py
│   │   │   ├── zip_archive_extractor.py
│   │   │   └── mal_api_2019_provisioner.py
│   │   ├── loader/                 # Parse the raw corpus into a Dataset (X, y)
│   │   │   ├── __init__.py
│   │   │   ├── dataset.py
│   │   │   ├── dataset_loader.py
│   │   │   └── mal_api_2019_loader.py
│   │   ├── normalization/          # Feature scaling: standardization and TF-IDF
│   │   │   ├── __init__.py
│   │   │   ├── normalizer.py
│   │   │   ├── standard_normalizer.py
│   │   │   └── tfidf_normalizer.py
│   │   ├── encoders/               # Label encoding: binary and one-hot
│   │   │   ├── __init__.py
│   │   │   ├── encoder.py
│   │   │   ├── binary_label_encoder.py
│   │   │   └── one_hot_encoder.py
│   │   ├── chunking/               # Train/test split + stratified K-fold CV
│   │   │   ├── __init__.py
│   │   │   ├── data_splitter.py
│   │   │   ├── data_split.py
│   │   │   ├── stratified_splitter.py
│   │   │   ├── k_fold_splitter.py
│   │   │   ├── stratified_k_fold_splitter.py
│   │   │   └── fold.py
│   │   ├── data_pipeline.py        # Glue: fetch → load → encode → normalize → split
│   │   └── data_pipeline_result.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── learning_model.py                  # Abstract base class
│   │   ├── supervised_learning_model.py
│   │   ├── support_vector_machine_model/      # Linear SVM (hard + soft margin)
│   │   │   ├── __init__.py
│   │   │   ├── svm_model.py
│   │   │   ├── hard_margin_svm.py
│   │   │   └── soft_margin_svm.py
│   │   ├── one_vs_one_classifier/             # 28-SVM ensemble for 8-class problem
│   │   │   ├── __init__.py
│   │   │   └── one_vs_one_classifier.py
│   │   ├── logistic_regression/               # Multinomial Logistic Regression
│   │   │   ├── __init__.py
│   │   │   └── multinomial_logistic_regression.py
│   │   └── markov_chain/                      # Per-class first-order Markov chain
│   │       ├── __init__.py
│   │       └── markov_chain_classifier.py
│   └── evaluations/                # Metrics: accuracy, macro P/R/F1, confusion matrix
│       ├── __init__.py
│       ├── metric.py
│       ├── per_class_metric.py
│       ├── accuracy_metric.py
│       ├── macro_precision_metric.py
│       ├── macro_recall_metric.py
│       ├── macro_f1_metric.py
│       └── confusion_matrix.py
├── tests/                          # 115 unit tests (pytest)
├── pytest.ini
├── requirements.txt
└── README.md
```

The `.data/` directory holds the downloaded corpus and is gitignored — it is materialized on the first run of the experiment notebook.

## Running it

```bash
pip install -r requirements.txt
pytest                              # run the 115 unit tests
jupyter notebook notebooks/01_experiment.ipynb
```

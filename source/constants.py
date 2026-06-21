# MAL-API-2019 dataset filenames inside .data/
SAMPLES_FILENAME: str = "all_analysis_data.txt"
LABELS_FILENAME: str = "labels.csv"
ARCHIVE_FILENAME: str = "mal-api-2019.zip"

# BinaryLabelEncoder — signed labels for binary SVM training
NEGATIVE_LABEL: float = -1.0
POSITIVE_LABEL: float = 1.0

# HardMarginSVM — effectively-infinite regularisation strength
HARD_MARGIN_REGULARIZATION_STRENGTH: float = 1e6

# MarkovChainClassifier — reserved index for out-of-vocabulary tokens
UNKNOWN_TOKEN_INDEX: int = 0

# Reproducibility — every randomised step in the pipeline takes this seed
DEFAULT_RANDOM_STATE: int = 47

# StratifiedSplitter — train/test partition
DEFAULT_TEST_SIZE: float = 0.2

# StratifiedKFoldSplitter — cross-validation
DEFAULT_COUNT_OF_FOLDS: int = 5

# TfidfNormalizer — vocabulary cap and n-gram window
DEFAULT_MAX_FEATURES: int = 5000
DEFAULT_NGRAM_RANGE: tuple = (1, 1)
DEFAULT_SUBLINEAR_TF: bool = True

# HttpApiFetcher — streaming download tunables
DEFAULT_CHUNK_SIZE: int = 1 << 16
DEFAULT_TIMEOUT_SECONDS: int = 60

# Linear-model optimisation knobs — shared by SVMs and MNLR
DEFAULT_REGULARIZATION_STRENGTH: float = 1.0
DEFAULT_MAX_ITERATIONS: int = 2000
DEFAULT_TOLERANCE: float = 1e-6
DEFAULT_SVM_LEARNING_RATE: float = 0.01

# HardMarginSVM — uses a smaller learning rate and more iterations than soft margin
DEFAULT_HARD_MARGIN_LEARNING_RATE: float = 0.0001
DEFAULT_HARD_MARGIN_MAX_ITERATIONS: int = 5000

# MarkovChainClassifier — Laplace smoothing strength
DEFAULT_SMOOTHING_ALPHA: float = 1.0

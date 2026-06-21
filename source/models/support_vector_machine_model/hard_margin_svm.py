from numpy.typing import ArrayLike

from source.data.encoders.binary_label_encoder import BinaryLabelEncoder
from source.models.support_vector_machine_model.soft_margin_svm import SoftMarginSVM
from source.constants import (
    HARD_MARGIN_REGULARIZATION_STRENGTH, DEFAULT_HARD_MARGIN_LEARNING_RATE,
    DEFAULT_HARD_MARGIN_MAX_ITERATIONS, DEFAULT_TOLERANCE
)


class HardMarginSVM(SoftMarginSVM):
    """Hard-margin SVM realised as soft-margin with an effectively infinite regularization_strength."""

    def __init__(self, classes: ArrayLike, binary_label_encoder: BinaryLabelEncoder,
        learning_rate: float = DEFAULT_HARD_MARGIN_LEARNING_RATE,
        max_iterations: int = DEFAULT_HARD_MARGIN_MAX_ITERATIONS,
        tolerance: float = DEFAULT_TOLERANCE) -> None:
        super().__init__(
            classes=classes,
            binary_label_encoder=binary_label_encoder,
            regularization_strength=HARD_MARGIN_REGULARIZATION_STRENGTH,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance
        )

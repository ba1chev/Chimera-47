from numpy.typing import ArrayLike

from source.data.encoders.binary_label_encoder import BinaryLabelEncoder
from source.models.support_vector_machine_model.soft_margin_svm import SoftMarginSVM


class HardMarginSVM(SoftMarginSVM):
    """Hard-margin SVM realised as soft-margin with an effectively infinite regularization_strength."""

    HARD_MARGIN_REGULARIZATION_STRENGTH: float = 1e6

    def __init__(self, classes: ArrayLike, binary_label_encoder: BinaryLabelEncoder,
        learning_rate: float = 0.0001, max_iterations: int = 5000, tolerance: float = 1e-6) -> None:
        super().__init__(
            classes=classes,
            binary_label_encoder=binary_label_encoder,
            regularization_strength=self.HARD_MARGIN_REGULARIZATION_STRENGTH,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance
        )

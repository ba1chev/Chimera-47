from source.evaluations.per_class_metric import PerClassMetric


class MacroPrecisionMetric(PerClassMetric):
    """Macro-averaged precision: per-class TP / (TP + FP), averaged uniformly across classes."""

    def _score_from_counts(self, true_positive: int, false_positive: int, false_negative: int) -> float:
        denominator = true_positive + false_positive
        if denominator == 0:
            return 0.0
        return true_positive / denominator

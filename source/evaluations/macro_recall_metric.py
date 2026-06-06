from source.evaluations.per_class_metric import PerClassMetric


class MacroRecallMetric(PerClassMetric):
    """Macro-averaged recall: per-class TP / (TP + FN), averaged uniformly across classes."""

    def _score_from_counts(self, true_positive: int, false_positive: int, false_negative: int) -> float:
        denominator = true_positive + false_negative
        if denominator == 0:
            return 0.0
        return true_positive / denominator

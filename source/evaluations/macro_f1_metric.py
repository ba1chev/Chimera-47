from source.evaluations.per_class_metric import PerClassMetric


class MacroF1Metric(PerClassMetric):
    """Macro-averaged F1: harmonic mean of per-class precision and recall, averaged uniformly across classes."""

    def _score_from_counts(self, true_positive: int, false_positive: int, false_negative: int) -> float:
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        # If the class is never predicted or never present, precision/recall are undefined — treat as 0.
        if precision_denominator == 0 or recall_denominator == 0:
            return 0.0
        precision = true_positive / precision_denominator
        recall = true_positive / recall_denominator
        if precision + recall == 0.0:
            return 0.0
        # Harmonic mean punishes lopsided precision/recall more than the arithmetic mean would.
        return 2.0 * precision * recall / (precision + recall)

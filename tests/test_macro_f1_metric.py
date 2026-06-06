import pytest
import numpy as np

from source.evaluations.macro_f1_metric import MacroF1Metric


class TestMacroF1Metric:
    def test_perfect_predictions_yield_one(self):
        metric = MacroF1Metric()
        result = metric.evaluate(np.array(["a", "b", "c"]), np.array(["a", "b", "c"]))
        assert result == pytest.approx(1.0)

    def test_completely_wrong_predictions_yield_zero(self):
        metric = MacroF1Metric()
        result = metric.evaluate(np.array(["a", "b"]), np.array(["b", "a"]))
        assert result == pytest.approx(0.0)

    def test_is_harmonic_mean_of_macro_precision_and_recall_per_class(self):
        metric = MacroF1Metric()
        y_true = np.array(["a", "a", "b"])
        y_predicted = np.array(["a", "b", "b"])
        assert metric.evaluate(y_true, y_predicted) == pytest.approx(2.0 / 3.0)

    def test_raises_on_shape_mismatch(self):
        metric = MacroF1Metric()
        with pytest.raises(ValueError, match="Shape mismatch"):
            metric.evaluate(np.array([1, 2, 3]), np.array([1, 2]))

    def test_raises_on_empty_input(self):
        metric = MacroF1Metric()
        with pytest.raises(ValueError, match="empty"):
            metric.evaluate(np.array([]), np.array([]))

    def test_class_with_zero_precision_and_zero_recall_contributes_zero(self):
        metric = MacroF1Metric()
        # class C predicted but never actually present and class B never predicted
        y_true = np.array(["a", "b"])
        y_predicted = np.array(["a", "c"])
        # A: TP=1, FP=0, FN=0 → P=R=F1=1
        # B: TP=0, FP=0, FN=1 → P=0, R=0, F1=0
        # C: TP=0, FP=1, FN=0 → P=0, R=0, F1=0
        # macro = 1/3
        assert metric.evaluate(y_true, y_predicted) == pytest.approx(1.0 / 3.0)

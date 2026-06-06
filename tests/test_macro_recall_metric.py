import pytest
import numpy as np

from source.evaluations.macro_recall_metric import MacroRecallMetric


class TestMacroRecallMetric:
    def test_perfect_predictions_yield_one(self):
        metric = MacroRecallMetric()
        result = metric.evaluate(np.array(["a", "b", "c"]), np.array(["a", "b", "c"]))
        assert result == pytest.approx(1.0)

    def test_completely_wrong_predictions_yield_zero(self):
        metric = MacroRecallMetric()
        result = metric.evaluate(np.array(["a", "b"]), np.array(["b", "a"]))
        assert result == pytest.approx(0.0)

    def test_macro_average_is_unweighted_mean_of_per_class_recalls(self):
        metric = MacroRecallMetric()
        y_true = np.array(["a", "a", "b", "b"])
        y_predicted = np.array(["a", "a", "a", "b"])
        assert metric.evaluate(y_true, y_predicted) == pytest.approx(0.75)

    def test_class_present_only_in_predictions_contributes_zero(self):
        metric = MacroRecallMetric()
        y_true = np.array(["a", "b"])
        y_predicted = np.array(["a", "c"])
        assert metric.evaluate(y_true, y_predicted) == pytest.approx(1.0 / 3.0)

    def test_raises_on_shape_mismatch(self):
        metric = MacroRecallMetric()
        with pytest.raises(ValueError, match="Shape mismatch"):
            metric.evaluate(np.array([1, 2, 3]), np.array([1, 2]))

    def test_raises_on_empty_input(self):
        metric = MacroRecallMetric()
        with pytest.raises(ValueError, match="empty"):
            metric.evaluate(np.array([]), np.array([]))

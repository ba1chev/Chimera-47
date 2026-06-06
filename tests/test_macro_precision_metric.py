import pytest
import numpy as np

from source.evaluations.macro_precision_metric import MacroPrecisionMetric


class TestMacroPrecisionMetric:
    def test_perfect_predictions_yield_one(self):
        metric = MacroPrecisionMetric()
        result = metric.evaluate(np.array(["a", "b", "c"]), np.array(["a", "b", "c"]))
        assert result == pytest.approx(1.0)

    def test_completely_wrong_predictions_yield_zero(self):
        metric = MacroPrecisionMetric()
        result = metric.evaluate(np.array(["a", "b"]), np.array(["b", "a"]))
        assert result == pytest.approx(0.0)

    def test_macro_average_is_unweighted_mean_of_per_class_precisions(self):
        metric = MacroPrecisionMetric()
        y_true = np.array(["a", "a", "b", "b"])
        y_predicted = np.array(["a", "a", "a", "b"])
        assert metric.evaluate(y_true, y_predicted) == pytest.approx(5.0 / 6.0)

    def test_class_with_no_predictions_contributes_zero(self):
        metric = MacroPrecisionMetric()
        y_true = np.array(["a", "b", "c"])
        y_predicted = np.array(["a", "b", "a"])
        assert metric.evaluate(y_true, y_predicted) == pytest.approx(0.5)

    def test_raises_on_shape_mismatch(self):
        metric = MacroPrecisionMetric()
        with pytest.raises(ValueError, match="Shape mismatch"):
            metric.evaluate(np.array([1, 2, 3]), np.array([1, 2]))

    def test_raises_on_empty_input(self):
        metric = MacroPrecisionMetric()
        with pytest.raises(ValueError, match="empty"):
            metric.evaluate(np.array([]), np.array([]))

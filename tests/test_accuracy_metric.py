import pytest
import numpy as np

from source.evaluations.accuracy_metric import AccuracyMetric


class TestAccuracyMetric:
    def test_returns_one_for_perfect_predictions(self):
        metric = AccuracyMetric()
        result = metric.evaluate(np.array(["a", "b", "c"]), np.array(["a", "b", "c"]))
        assert result == 1.0

    def test_returns_zero_for_completely_wrong_predictions(self):
        metric = AccuracyMetric()
        result = metric.evaluate(np.array(["a", "b", "c"]), np.array(["b", "c", "a"]))
        assert result == 0.0

    def test_returns_fraction_of_matches(self):
        metric = AccuracyMetric()
        result = metric.evaluate(
            np.array(["a", "b", "c", "d"]), np.array(["a", "b", "c", "x"])
        )
        assert result == pytest.approx(0.75)

    def test_works_with_integer_labels(self):
        metric = AccuracyMetric()
        result = metric.evaluate(np.array([0, 1, 2, 1]), np.array([0, 1, 0, 1]))
        assert result == pytest.approx(0.75)

    def test_raises_on_shape_mismatch(self):
        metric = AccuracyMetric()
        with pytest.raises(ValueError, match="Shape mismatch"):
            metric.evaluate(np.array([1, 2, 3]), np.array([1, 2]))

    def test_raises_on_empty_input(self):
        metric = AccuracyMetric()
        with pytest.raises(ValueError, match="empty"):
            metric.evaluate(np.array([]), np.array([]))

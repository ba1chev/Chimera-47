import pytest
import numpy as np

from source.functions.function_predicates.sigmoid_function_predicate import sigmoid_function_predicate


class TestSigmoidFunctionPredicate:
    def test_returns_zero_point_five_when_logit_is_zero(self):
        parameters = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
        result = sigmoid_function_predicate(parameters)
        assert result == pytest.approx(0.5)

    def test_returns_value_above_half_for_positive_logit(self):
        parameters = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = sigmoid_function_predicate(parameters)
        assert result > 0.5

    def test_returns_value_below_half_for_negative_logit(self):
        parameters = np.array([-3.0, 1.0, 1.0, 1.0, 1.0])
        result = sigmoid_function_predicate(parameters)
        assert result < 0.5

    def test_returns_near_one_for_very_large_logit(self):
        parameters = np.array([100.0, 1.0, 1.0, 1.0, 1.0])
        result = sigmoid_function_predicate(parameters)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_returns_near_zero_for_very_negative_logit(self):
        parameters = np.array([-100.0, 1.0, 1.0, 1.0, 1.0])
        result = sigmoid_function_predicate(parameters)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_raises_when_length_is_even(self):
        parameters = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="length"):
            sigmoid_function_predicate(parameters)

    def test_raises_when_length_is_less_than_three(self):
        parameters = np.array([1.0])
        with pytest.raises(ValueError, match="length"):
            sigmoid_function_predicate(parameters)

    def test_returns_float_type(self):
        parameters = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
        result = sigmoid_function_predicate(parameters)
        assert isinstance(result, float)

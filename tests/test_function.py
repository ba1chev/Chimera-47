import pytest
import numpy as np

from source.functions.function import Function


def constant_predicate(parameters):
    return 42.0


def sum_predicate(parameters):
    return float(parameters.sum())


class TestFunction:
    def test_arity_returns_count_of_variables(self):
        function = Function(constant_predicate, count_of_variables=5)
        assert function.arity == 5

    def test_call_returns_predicate_result(self):
        function = Function(sum_predicate, count_of_variables=3)
        result = function(np.array([1.0, 2.0, 3.0]))
        assert result == pytest.approx(6.0)

    def test_call_raises_on_wrong_arity(self):
        function = Function(constant_predicate, count_of_variables=3)
        with pytest.raises(ValueError, match="3 parameters"):
            function(np.array([1.0, 2.0]))

    def test_constructor_raises_on_none_predicate(self):
        with pytest.raises(ValueError, match="function_predicate"):
            Function(None, count_of_variables=3)

    def test_constructor_raises_on_negative_arity(self):
        with pytest.raises(ValueError, match="non-negative"):
            Function(constant_predicate, count_of_variables=-1)

    def test_zero_arity_is_allowed(self):
        function = Function(lambda parameters: 99.0, count_of_variables=0)
        assert function(np.array([])) == pytest.approx(99.0)

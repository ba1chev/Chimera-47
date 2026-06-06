import pytest
import numpy as np

from source.functions.vector_function import VectorFunction


def identity_predicate(parameters):
    return parameters


def double_predicate(parameters):
    return parameters * 2.0


class TestVectorFunction:
    def test_count_of_input_features_property(self):
        function = VectorFunction(identity_predicate, count_of_input_features=4)
        assert function.count_of_input_features == 4

    def test_call_returns_predicate_result(self):
        function = VectorFunction(double_predicate, count_of_input_features=3)
        result = function(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result, [2.0, 4.0, 6.0])

    def test_accepts_one_dimensional_input(self):
        function = VectorFunction(identity_predicate, count_of_input_features=3)
        result = function(np.array([1.0, 2.0, 3.0]))
        assert result.shape == (3,)

    def test_accepts_two_dimensional_input_with_matching_trailing_axis(self):
        function = VectorFunction(identity_predicate, count_of_input_features=3)
        result = function(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        assert result.shape == (2, 3)

    def test_raises_on_wrong_trailing_axis(self):
        function = VectorFunction(identity_predicate, count_of_input_features=3)
        with pytest.raises(ValueError, match="trailing axis"):
            function(np.array([1.0, 2.0]))

    def test_constructor_raises_on_none_predicate(self):
        with pytest.raises(ValueError, match="function_predicate"):
            VectorFunction(None, count_of_input_features=3)

    def test_constructor_raises_on_zero_features(self):
        with pytest.raises(ValueError, match="positive"):
            VectorFunction(identity_predicate, count_of_input_features=0)

    def test_constructor_raises_on_negative_features(self):
        with pytest.raises(ValueError, match="positive"):
            VectorFunction(identity_predicate, count_of_input_features=-1)

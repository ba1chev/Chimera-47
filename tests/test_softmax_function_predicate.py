import pytest
import numpy as np

from source.functions.vector_function_predicates.softmax_function_predicate import softmax_function_predicate


class TestSoftmaxFunctionPredicate:
    def test_output_sums_to_one_for_one_dimensional_input(self):
        logits = np.array([1.0, 2.0, 3.0])
        result = softmax_function_predicate(logits)
        assert result.sum() == pytest.approx(1.0)

    def test_output_sums_to_one_per_row_for_two_dimensional_input(self):
        logits = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = softmax_function_predicate(logits)
        np.testing.assert_allclose(result.sum(axis=-1), [1.0, 1.0])

    def test_uniform_input_produces_uniform_output(self):
        logits = np.array([5.0, 5.0, 5.0, 5.0])
        result = softmax_function_predicate(logits)
        np.testing.assert_allclose(result, [0.25, 0.25, 0.25, 0.25])

    def test_largest_logit_yields_largest_probability(self):
        logits = np.array([1.0, 5.0, 2.0])
        result = softmax_function_predicate(logits)
        assert int(np.argmax(result)) == 1

    def test_translation_invariance(self):
        logits = np.array([1.0, 2.0, 3.0])
        shifted = logits + 1000.0
        np.testing.assert_allclose(
            softmax_function_predicate(logits),
            softmax_function_predicate(shifted),
            atol=1e-12,
        )

    def test_does_not_overflow_with_large_logits(self):
        logits = np.array([1000.0, 1001.0, 1002.0])
        result = softmax_function_predicate(logits)
        assert np.all(np.isfinite(result))
        assert result.sum() == pytest.approx(1.0)

    def test_all_outputs_are_non_negative(self):
        logits = np.array([-3.0, 0.0, 2.5, -100.0])
        result = softmax_function_predicate(logits)
        assert np.all(result >= 0.0)

    def test_preserves_input_shape(self):
        logits = np.zeros((4, 3, 5))
        result = softmax_function_predicate(logits)
        assert result.shape == (4, 3, 5)

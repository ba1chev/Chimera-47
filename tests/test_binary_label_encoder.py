import pytest
import numpy as np

from source.data.encoders.binary_label_encoder import BinaryLabelEncoder


class TestBinaryLabelEncoder:
    def test_constructor_raises_on_single_class(self):
        with pytest.raises(ValueError, match="exactly 2 classes"):
            BinaryLabelEncoder(["only_one"])

    def test_constructor_raises_on_three_classes(self):
        with pytest.raises(ValueError, match="exactly 2 classes"):
            BinaryLabelEncoder(["a", "b", "c"])

    def test_constructor_raises_on_duplicate_classes(self):
        with pytest.raises(ValueError, match="distinct"):
            BinaryLabelEncoder(["same", "same"])

    def test_encode_maps_first_class_to_minus_one_and_second_to_plus_one(self):
        encoder = BinaryLabelEncoder(["neg", "pos"])
        encoded = encoder.encode(np.array(["neg", "pos", "neg", "pos"]))
        np.testing.assert_array_equal(encoded, [-1.0, 1.0, -1.0, 1.0])

    def test_encode_raises_on_unseen_label(self):
        encoder = BinaryLabelEncoder(["neg", "pos"])
        with pytest.raises(ValueError, match="not in declared classes"):
            encoder.encode(np.array(["neg", "outsider"]))

    def test_decode_inverts_encode_for_signed_values(self):
        encoder = BinaryLabelEncoder(["neg", "pos"])
        decoded = encoder.decode(np.array([-1.0, 1.0, -1.0, 1.0]))
        np.testing.assert_array_equal(decoded, ["neg", "pos", "neg", "pos"])

    def test_decode_treats_zero_as_negative_class(self):
        encoder = BinaryLabelEncoder(["neg", "pos"])
        decoded = encoder.decode(np.array([0.0]))
        assert decoded[0] == "neg"

    def test_decode_uses_sign_for_arbitrary_magnitudes(self):
        encoder = BinaryLabelEncoder(["neg", "pos"])
        decoded = encoder.decode(np.array([-5.5, 3.2, -0.01, 0.001]))
        np.testing.assert_array_equal(decoded, ["neg", "pos", "neg", "pos"])

    def test_decode_raises_on_two_dimensional_input(self):
        encoder = BinaryLabelEncoder(["neg", "pos"])
        with pytest.raises(ValueError, match="1D"):
            encoder.decode(np.array([[1.0], [-1.0]]))

    def test_classes_property_returns_declared_classes(self):
        encoder = BinaryLabelEncoder(["alpha", "beta"])
        np.testing.assert_array_equal(encoder.classes, ["alpha", "beta"])

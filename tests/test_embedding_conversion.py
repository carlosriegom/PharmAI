import unittest
import numpy as np

class TestEmbeddingConversion(unittest.TestCase):
    def test_conversion_to_float32(self):
        array = np.array([1.0, 2.0, 3.0])
        converted = array.astype(np.float32)
        self.assertEqual(converted.dtype, np.float32)

    def test_empty_array(self):
        array = np.array([])
        converted = array.astype(np.float32)
        self.assertEqual(converted.dtype, np.float32)

    def test_negative_values(self):
        array = np.array([-1.0, -2.0, -3.0])
        converted = array.astype(np.float32)
        self.assertEqual(converted.dtype, np.float32)

    def test_large_values(self):
        array = np.array([1e10, 1e20])
        converted = array.astype(np.float32)
        self.assertEqual(converted.dtype, np.float32)

if __name__ == '__main__':
    unittest.main()
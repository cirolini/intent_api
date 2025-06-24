import unittest

from app.utils import preprocess


class TestUtils(unittest.TestCase):
    def test_preprocess(self):
        self.assertEqual(preprocess("Hello, WORLD!"), "hello world")
        self.assertEqual(preprocess("  Test...  "), "test")
        self.assertEqual(preprocess("Número 123!"), "número 123")
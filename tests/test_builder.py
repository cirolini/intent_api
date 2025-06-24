import unittest

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from app.builder import build_hnsw


@unittest.skipUnless(HAS_FAISS, "faiss library is required for builder tests")
class TestBuilder(unittest.TestCase):
    def test_build_hnsw_default(self):
        dim = 64
        index = build_hnsw(dim)
        self.assertIsInstance(index, faiss.Index)
        self.assertEqual(index.d, dim)
        self.assertEqual(index.hnsw.efConstruction, 400)
        self.assertEqual(index.hnsw.efSearch, 128)

    def test_build_hnsw_custom(self):
        dim = 32
        m = 16
        ef_construction = 100
        index = build_hnsw(dim, m=m, ef_construction=ef_construction)
        self.assertEqual(index.d, dim)
        self.assertEqual(index.hnsw.efConstruction, ef_construction)
        self.assertEqual(index.hnsw.efSearch, 128)
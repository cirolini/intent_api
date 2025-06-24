import unittest

from app.model import _predict_intent


class TestModelPredictIntent(unittest.TestCase):
    def test_majority_vote(self):
        intents = ["a", "b", "a", "c"]
        sims = [0.9, 0.8, 0.7, 0.6]
        self.assertEqual(_predict_intent(intents, sims, thr=0.5), "a")

    def test_tie_by_average(self):
        intents = ["a", "b", "a", "b"]
        sims = [0.9, 0.8, 0.7, 0.6]
        self.assertEqual(_predict_intent(intents, sims, thr=0.5), "a")

    def test_threshold_oos(self):
        intents = ["a", "b", "c"]
        sims = [0.4, 0.3, 0.2]
        self.assertEqual(_predict_intent(intents, sims, thr=0.5), "oos")
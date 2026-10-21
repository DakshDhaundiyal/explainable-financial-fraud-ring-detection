"""End-to-End API and TGAT Inference Integration Tests."""
import unittest
from fastapi.testclient import TestClient

from backend.main import app


class TestTGATBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["feature_dim"], 26)
        self.assertAlmostEqual(data["frozen_threshold"], 0.737155, places=4)

    def test_metrics_endpoint(self):
        res = self.client.get("/api/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("tgat_frozen_test", data)
        self.assertIn("baselines", data)
        self.assertIn("logistic_regression", data["baselines"])
        self.assertIn("static_gcn", data["baselines"])
        self.assertIn("xgboost", data["baselines"])
        self.assertIn("temporal_xgboost", data["baselines"])
        self.assertAlmostEqual(data["tgat_frozen_test"]["roc_auc"], 0.9022, places=3)
        self.assertAlmostEqual(data["tgat_frozen_test"]["pr_auc"], 0.0184, places=3)

    def test_threshold_endpoint(self):
        res = self.client.get("/api/threshold")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("frozen", data)
        self.assertAlmostEqual(data["frozen"]["threshold"], 0.737155, places=4)
        self.assertTrue(len(data["sweep_sample"]) > 0)

    def test_cases_endpoint(self):
        res = self.client.get("/api/cases")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("cases", data)
        self.assertGreaterEqual(len(data["cases"]), 3)
        case1 = data["cases"][0]
        self.assertIn("graph", case1)
        self.assertIn("nodes", case1["graph"])
        self.assertIn("edges", case1["graph"])

    def test_ablations_endpoint(self):
        res = self.client.get("/api/ablations")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["count"], 15)
        self.assertEqual(len(data["experiments"]), 15)

    def test_predict_fraud_case(self):
        payload = {
            "from_account": "81156E7A0",
            "to_account": "81156E840",
            "amount": 189.19,
            "cross_bank": True,
            "same_currency": True,
            "custom_features": {
                "Amount Received": 189.19,
                "Amount Paid": 189.19,
                "sender_in_count": 22.0,
                "sender_out_count": 45.0,
                "sender_total_count": 67.0,
                "sender_time_since_last": 15.0,
            }
        }
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("probability", data)
        self.assertIn("decision", data)
        self.assertIn("sender_attention", data)
        self.assertIn("explainability", data)
        self.assertIsInstance(data["probability"], float)

    def test_predict_clean_case(self):
        payload = {
            "from_account": "8018B08B0",
            "to_account": "8005A0470",
            "amount": 240.0,
            "cross_bank": False,
            "same_currency": True,
        }
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["decision"], "LEGITIMATE_CLEAR")
        self.assertLess(data["probability"], 0.737155)

    def test_frontend_static_serving(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("TGAT", res.text)
        self.assertIn("FRAUD RING DEFENDER", res.text)


if __name__ == "__main__":
    unittest.main()

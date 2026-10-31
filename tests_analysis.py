"""Integration tests for batch upload, validation, TGAT scoring, and investigation API endpoints."""
import unittest
from fastapi.testclient import TestClient

from backend.main import app


class TestAnalysisWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_sample_endpoint(self):
        res = self.client.get("/api/analysis/sample?count=20")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("csv_content", data)
        self.assertEqual(data["row_count"], 20)
        self.assertTrue(len(data["preview"]) > 0)

    def test_validation_endpoint_valid_and_invalid(self):
        # Invalid CSV: missing required columns
        bad_csv = "colA,colB\n1,2"
        res = self.client.post("/api/analysis/validate", json={"filename": "bad.csv", "csv_content": bad_csv})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["valid"])
        self.assertTrue(len(data["errors"]) > 0)

        # Valid CSV from sample
        sample_res = self.client.get("/api/analysis/sample?count=10")
        sample_csv = sample_res.json()["csv_content"]
        res_valid = self.client.post("/api/analysis/validate", json={"filename": "test.csv", "csv_content": sample_csv})
        self.assertEqual(res_valid.status_code, 200)
        val_data = res_valid.json()
        self.assertTrue(val_data["valid"])
        self.assertEqual(val_data["valid_rows"], 10)

    def test_run_analysis_and_query_results(self):
        # Get sample CSV (15 transactions)
        sample_res = self.client.get("/api/analysis/sample?count=15")
        sample_csv = sample_res.json()["csv_content"]

        # Run Analysis
        run_res = self.client.post("/api/analysis/run", json={"filename": "demo_batch.csv", "csv_content": sample_csv})
        self.assertEqual(run_res.status_code, 200)
        run_data = run_res.json()
        self.assertEqual(run_data["status"], "success")
        session_id = run_data["session_id"]
        self.assertEqual(run_data["total_transactions"], 15)
        self.assertAlmostEqual(run_data["threshold"], 0.737155, places=4)

        # Check Overview
        ov_res = self.client.get(f"/api/analysis/overview?session_id={session_id}")
        self.assertEqual(ov_res.status_code, 200)
        ov_data = ov_res.json()
        self.assertTrue(ov_data["has_data"])
        self.assertEqual(ov_data["total_transactions"], 15)
        self.assertEqual(len(ov_data["score_distribution"]), 10)

        # Check Results Table (Paginated)
        res_table = self.client.get(f"/api/analysis/{session_id}/results?page=1&limit=5&sort_by=risk_score&sort_dir=desc")
        self.assertEqual(res_table.status_code, 200)
        table_data = res_table.json()
        self.assertEqual(table_data["total"], 15)
        self.assertEqual(len(table_data["items"]), 5)
        self.assertEqual(table_data["pages"], 3)
        first_tx = table_data["items"][0]
        self.assertIn("id", first_tx)
        self.assertIn("risk_score", first_tx)
        self.assertIn("classification", first_tx)
        self.assertIn("explainability", first_tx)

        # Check Graph API
        graph_res = self.client.get(f"/api/analysis/{session_id}/graph")
        self.assertEqual(graph_res.status_code, 200)
        graph_data = graph_res.json()
        self.assertTrue(len(graph_data["nodes"]) > 0)
        self.assertTrue(len(graph_data["edges"]) > 0)

        # Check Transaction Detail
        tx_id = first_tx["id"]
        detail_res = self.client.get(f"/api/analysis/{session_id}/transaction/{tx_id}")
        self.assertEqual(detail_res.status_code, 200)
        tx_detail = detail_res.json()
        self.assertEqual(tx_detail["id"], tx_id)
        self.assertIn("sender_attention", tx_detail)
        self.assertIn("receiver_attention", tx_detail)

        # Check Account Detail
        acc_id = first_tx["from_account"]
        acc_res = self.client.get(f"/api/analysis/{session_id}/account/{acc_id}")
        self.assertEqual(acc_res.status_code, 200)
        acc_detail = acc_res.json()
        self.assertEqual(acc_detail["account_id"], acc_id)
        self.assertTrue(acc_detail["total_transactions"] >= 1)
        self.assertIn("ai_narrative", acc_detail)
        self.assertTrue(len(acc_detail["ai_narrative"]) > 10)

        # Check Account Ask AI Endpoint
        ask_res = self.client.post(
            f"/api/analysis/{session_id}/account/{acc_id}/ask",
            json={"question": "Who are the connected accounts?"}
        )
        self.assertEqual(ask_res.status_code, 200)
        ask_data = ask_res.json()
        self.assertIn("answer", ask_data)
        self.assertTrue(len(ask_data["answer"]) > 5)

        # Check CSV Export
        exp_res = self.client.get(f"/api/analysis/{session_id}/export")
        self.assertEqual(exp_res.status_code, 200)
        self.assertEqual(exp_res.headers["content-type"], "text/csv; charset=utf-8")
        self.assertIn("From_Account", exp_res.text)
        self.assertIn("Risk_Score", exp_res.text)
        self.assertIn("Classification", exp_res.text)


if __name__ == "__main__":
    unittest.main()

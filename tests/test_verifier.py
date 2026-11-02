"""Native receipt API controls, using stdlib unittest only."""
import json
import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify.verifier import GENESIS, digest, verify
from tools.sync_results import main as sync_results


class NativeVerifierTests(unittest.TestCase):
    def test_malformed_receipts_cannot_publish_measured_rows(self):
        base = {"plane": "retrieval", "status": "MEASURED",
                "machine": {"cpu": "unit-fixture", "ram_gb": 1, "gpu": "none"},
                "measured_at": "2000-01-01T00:00:00Z", "method": "fixture-control",
                "metrics": {"score": 0.5}, "prev_hash": GENESIS}

        def legacy_encoded(changes):
            receipt = {**base, **changes}
            receipt["hash"] = hashlib.sha256(json.dumps(
                receipt, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            return json.dumps(receipt)

        valid = legacy_encoded({})
        cases = ["null", "[]", '"receipt"', "42",
                 legacy_encoded({"metrics": "made up"}),
                 legacy_encoded({"metrics": [0.5]}),
                 legacy_encoded({"plane": []}),
                 legacy_encoded({"status": {}}),
                 legacy_encoded({"metrics": {"score": float("nan")}}),
                 legacy_encoded({"metrics": {"nested": [float("inf")]}}),
                 legacy_encoded({"metrics": {"score": -float("inf")}}),
                 valid.replace('"score": 0.5', '"score": 0.1, "score": 0.5'),
                 valid.replace('"status": "MEASURED"',
                               '"status": "BLOCKED", "status": "MEASURED"')]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            receipts = root / "receipts"
            receipts.mkdir()
            path = receipts / "receipt.json"
            output = root / "site" / "results.json"
            output.parent.mkdir()
            for text in cases:
                with self.subTest(text=text):
                    path.write_text(text, encoding="utf-8")
                    output.write_text("existing publication", encoding="utf-8")
                    errors, measured = verify([str(path)])
                    self.assertTrue(errors)
                    self.assertEqual(measured, [])
                    self.assertEqual(sync_results(str(receipts), str(output)), 1)
                    self.assertEqual(output.read_text(), "existing publication")

    def test_valid_nested_metrics_remain_publishable(self):
        receipt = {"plane": "retrieval", "status": "MEASURED",
                   "machine": {"cpu": "unit-fixture", "ram_gb": 1, "gpu": "none"},
                   "measured_at": "2000-01-01T00:00:00Z", "method": "fixture-control",
                   "metrics": {"score": 0.5, "by_query": [0, 0.5, 1]},
                   "prev_hash": GENESIS}
        receipt["hash"] = digest(receipt)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            output = root / "site" / "results.json"
            self.assertEqual(sync_results(str(root), str(output)), 0)
            self.assertEqual(json.loads(output.read_text())["results"][0]["metrics"],
                             receipt["metrics"])

    def test_recorded_native_repair_report_recomputes(self):
        path = Path(__file__).resolve().parents[1] / "evidence/eclipse-native-repaired-20260905.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        receipt = report.pop("receipt")
        encoded = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.assertEqual(hashlib.sha256(encoded.encode()).hexdigest(), receipt)
        self.assertEqual(report["sensitivity"], "10/10")
        self.assertEqual(report["blind_spots"], [])
        self.assertTrue(report["baseline"]["valid"])
        self.assertEqual(report["baseline"]["detail"]["measured_count"], 3)

    def test_empty_collection_and_iterator_fail_closed(self):
        for paths in ([], (), iter(())):
            errors, measured = verify(paths)
            self.assertTrue(errors)
            self.assertEqual(measured, [])

    def test_missing_collection_fails_closed(self):
        errors, measured = verify(None)
        self.assertTrue(errors)
        self.assertEqual(measured, [])

    def test_valid_nonempty_control_and_changed_payload(self):
        receipt = {"plane": "retrieval", "status": "MEASURED",
                   "machine": {"cpu": "unit-fixture", "ram_gb": 1, "gpu": "none"},
                   "measured_at": "2000-01-01T00:00:00Z", "method": "fixture-control",
                   "metrics": {"score": 0.5}, "prev_hash": GENESIS}
        receipt["hash"] = digest(receipt)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            errors, measured = verify([str(path)])
            self.assertEqual(errors, [])
            self.assertEqual(measured, [str(path)])
            receipt["metrics"]["score"] = 0.9
            path.write_text(json.dumps(receipt), encoding="utf-8")
            errors, _ = verify([str(path)])
            self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()

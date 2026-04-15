import os
import tempfile
import unittest
from datetime import date, timedelta

from portfolio_engine import PortfolioEngine


class TestPortfolioEngine(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = os.path.join(self.tmpdir.name, "portfolio_test.db")
        self.engine = PortfolioEngine(db_path=self.db_path)
        self.portfolio = self.engine.create_portfolio("Oncology Portfolio", owner="alice")
        self.project = self.engine.create_project(
            portfolio_id=self.portfolio["id"],
            name="KRAS Program",
            indication="NSCLC",
            modality="small molecule",
            owner="bob",
        )
        self.candidate = self.engine.add_target_candidate(
            project_id=self.project["id"],
            target_id="KRAS",
            alias="KRAS-G12C",
            priority=90,
        )

    def tearDown(self):
        del self.engine
        try:
            self.tmpdir.cleanup()
        except PermissionError:
            pass

    def _scores(self, base: float) -> dict:
        return {
            "expression": {"available": True, "score": base, "source_quality": 0.8},
            "pathway": {"available": True, "score": base, "source_quality": 0.8},
            "ppi": {"available": True, "score": base, "source_quality": 0.8},
            "genetic": {"available": True, "score": base, "source_quality": 0.8},
            "ligandability": {"available": True, "score": base, "source_quality": 0.8},
            "trials": {"available": True, "score": base, "source_quality": 0.8},
        }

    def test_db_init_and_crud(self):
        portfolios = self.engine.list_portfolios()
        self.assertEqual(len(portfolios), 1)
        projects = self.engine.list_projects(self.portfolio["id"])
        self.assertEqual(len(projects), 1)
        candidates = self.engine.list_target_candidates(self.project["id"])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["target_id"], "KRAS")

    def test_snapshot_versioning_and_time_series(self):
        self.engine.save_evidence_snapshot(
            target_candidate_id=self.candidate["id"],
            source_version="v1",
            component_scores=self._scores(52.0),
            confidence=0.62,
            completeness=0.70,
            key_findings={"highlights": ["baseline"]},
            trial_status="Phase 1",
        )
        self.engine.save_evidence_snapshot(
            target_candidate_id=self.candidate["id"],
            source_version="v2",
            component_scores=self._scores(66.0),
            confidence=0.75,
            completeness=0.88,
            key_findings={"highlights": ["expanded evidence"]},
            trial_status="Phase 2",
        )
        ts = self.engine.get_target_time_series(self.candidate["id"])
        self.assertEqual(len(ts["snapshots"]), 2)
        self.assertEqual(len(ts["drift_events"]), 1)
        self.assertIn(ts["drift_events"][0]["classification"], {"Positive", "Neutral", "Negative"})

    def test_evidence_drift_classification(self):
        self.engine.save_evidence_snapshot(
            target_candidate_id=self.candidate["id"],
            source_version="v1",
            component_scores=self._scores(40.0),
            confidence=0.45,
            completeness=0.60,
            key_findings={"highlights": ["limited"]},
            trial_status="Phase 1",
        )
        out = self.engine.save_evidence_snapshot(
            target_candidate_id=self.candidate["id"],
            source_version="v2",
            component_scores=self._scores(70.0),
            confidence=0.82,
            completeness=0.91,
            key_findings={"highlights": ["new positive dataset", "replicated signal", "trial update"]},
            trial_status="Phase 2",
        )
        drift = out["drift"]
        self.assertEqual(drift["classification"], "Positive")
        self.assertTrue(any("Trial phase advanced" in msg or "increased" in msg for msg in drift["change_log"]))

    def test_milestone_status_calculations(self):
        self.engine.create_milestone(
            project_id=self.project["id"],
            title="Mechanism confirmation",
            milestone_type="mechanism",
            due_date=(date.today() - timedelta(days=2)).isoformat(),
            owner="sam",
            status="in progress",
            criteria={"acceptance": ["orthogonal assay"]},
        )
        blocked = self.engine.create_milestone(
            project_id=self.project["id"],
            title="Lead optimization sprint",
            milestone_type="lead optimization",
            due_date=(date.today() + timedelta(days=10)).isoformat(),
            owner="sam",
            status="blocked",
            criteria={"acceptance": ["potency <= 100nM"]},
        )
        self.engine.update_milestone_status(blocked["id"], "complete", actor="sam")
        dashboard = self.engine.get_project_dashboard_data(self.project["id"])
        metrics = dashboard["milestone_metrics"]
        self.assertGreaterEqual(metrics["overdue_count"], 1)
        self.assertGreaterEqual(metrics["completion_pct"], 50.0)

    def test_go_no_go_decision_logic_edge_cases(self):
        # Weak evidence + blockers should map to NO-GO
        self.engine.create_milestone(
            project_id=self.project["id"],
            title="Translational package",
            milestone_type="translational",
            due_date=(date.today() + timedelta(days=8)).isoformat(),
            owner="lead",
            status="blocked",
            criteria={"acceptance": ["in vivo signal"]},
        )
        self.engine.save_evidence_snapshot(
            target_candidate_id=self.candidate["id"],
            source_version="v1",
            component_scores=self._scores(20.0),
            confidence=0.25,
            completeness=0.4,
            key_findings={"highlights": ["conflicting data"]},
            trial_status="Preclinical",
        )
        decision = self.engine.generate_decision_snapshot(self.project["id"])
        self.assertIn(decision["recommendation"], {"NO-GO", "HOLD", "GO"})
        self.assertIn("checks", decision)
        self.assertIn("hard_blockers", decision)

    def test_export_schema_validity(self):
        self.engine.save_evidence_snapshot(
            target_candidate_id=self.candidate["id"],
            source_version="v1",
            component_scores=self._scores(62.0),
            confidence=0.71,
            completeness=0.84,
            key_findings={"highlights": ["signal convergent"]},
            trial_status="Phase 2",
        )
        self.engine.create_milestone(
            project_id=self.project["id"],
            title="Clinical readiness gate",
            milestone_type="clinical readiness",
            due_date=(date.today() + timedelta(days=18)).isoformat(),
            owner="owner",
            status="complete",
            criteria={"acceptance": ["safety profile reviewed"]},
        )
        self.engine.generate_decision_snapshot(self.project["id"])

        packet_json = self.engine.export_project_packet(self.project["id"], format="json")
        packet_csv = self.engine.export_project_packet(self.project["id"], format="csv")
        packet_md = self.engine.export_project_packet(self.project["id"], format="md")
        self.assertTrue(packet_json["schema_valid"])
        self.assertTrue(packet_csv["schema_valid"])
        self.assertTrue(packet_md["schema_valid"])
        self.assertIn("candidate_ranking", packet_csv["content"])
        self.assertIn("Decision Snapshot", packet_md["content"])


if __name__ == "__main__":
    unittest.main()

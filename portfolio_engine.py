"""Portfolio management engine for biotech multi-project operations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import csv
import io
import json
import sqlite3
import uuid


class PortfolioEngine:
    """Persist and score portfolio/project/candidate entities for OmniBiMol."""

    DEFAULT_DRIFT_CONFIG: Dict[str, float] = {
        "score_delta": 8.0,
        "confidence_delta": 0.12,
        "completeness_delta": 0.12,
        "evidence_volume_delta": 2.0,
    }

    DEFAULT_DECISION_RULES: Dict[str, float] = {
        "min_evidence_confidence": 0.6,
        "min_translational_score": 55.0,
        "max_risk_burden": 70.0,
        "min_milestone_completion": 0.55,
    }

    def __init__(self, db_path: str = "omnibimol_portfolio.db", drift_config: Optional[Dict[str, float]] = None):
        self.db_path = db_path
        self.drift_config = dict(self.DEFAULT_DRIFT_CONFIG)
        if drift_config:
            self.drift_config.update(drift_config)
        self._comparison_cache: Dict[str, Dict[str, Any]] = {}
        self.init_db()

    def init_db(self) -> None:
        """Create schema and run safe migrations."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolios (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    owner TEXT,
                    description TEXT,
                    reviewer TEXT,
                    comments TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    portfolio_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    indication TEXT,
                    modality TEXT,
                    stage TEXT,
                    owner TEXT,
                    reviewer TEXT,
                    status TEXT,
                    comments TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(portfolio_id) REFERENCES portfolios(id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS target_candidates (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    alias TEXT,
                    rationale TEXT,
                    priority INTEGER DEFAULT 0,
                    owner TEXT,
                    reviewer TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_snapshots (
                    id TEXT PRIMARY KEY,
                    target_candidate_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source_version TEXT,
                    component_scores_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    completeness REAL NOT NULL,
                    key_findings_json TEXT NOT NULL,
                    evidence_volume INTEGER DEFAULT 0,
                    trial_status TEXT,
                    reviewer TEXT,
                    comments TEXT,
                    FOREIGN KEY(target_candidate_id) REFERENCES target_candidates(id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS milestones (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    type TEXT NOT NULL,
                    due_date TEXT,
                    owner TEXT,
                    reviewer TEXT,
                    status TEXT NOT NULL,
                    criteria_json TEXT NOT NULL,
                    comments TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_snapshots (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    rationale TEXT NOT NULL,
                    risks_json TEXT NOT NULL,
                    next_actions_json TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    checks_json TEXT NOT NULL,
                    assumptions_json TEXT NOT NULL,
                    blockers_json TEXT NOT NULL,
                    reviewer TEXT,
                    comments TEXT,
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS activity_log (
                    id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    actor TEXT
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_project_portfolio ON projects(portfolio_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candidate_project ON target_candidates(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshot_candidate_time ON evidence_snapshots(target_candidate_id, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_milestone_project ON milestones(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_decision_project_time ON decision_snapshots(project_id, timestamp)")
            conn.commit()

    def create_portfolio(self, name: str, owner: str = "", description: str = "", reviewer: str = "", comments: str = "") -> Dict[str, Any]:
        ts = self._now()
        row = {
            "id": self._id("pf"),
            "name": name.strip(),
            "owner": owner.strip(),
            "description": description.strip(),
            "reviewer": reviewer.strip(),
            "comments": comments.strip(),
            "created_at": ts,
            "updated_at": ts,
        }
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO portfolios (id, name, owner, description, reviewer, comments, created_at, updated_at)
                VALUES (:id, :name, :owner, :description, :reviewer, :comments, :created_at, :updated_at)
                """,
                row,
            )
            conn.commit()
        self._log_activity("portfolio", row["id"], "created", row, owner)
        return row

    def list_portfolios(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM portfolios ORDER BY updated_at DESC, name ASC").fetchall()
        return [dict(r) for r in rows]

    def create_project(
        self,
        portfolio_id: str,
        name: str,
        indication: str = "",
        modality: str = "",
        stage: str = "discovery",
        owner: str = "",
        status: str = "active",
        reviewer: str = "",
        comments: str = "",
    ) -> Dict[str, Any]:
        ts = self._now()
        row = {
            "id": self._id("prj"),
            "portfolio_id": portfolio_id,
            "name": name.strip(),
            "indication": indication.strip(),
            "modality": modality.strip(),
            "stage": stage.strip(),
            "owner": owner.strip(),
            "reviewer": reviewer.strip(),
            "status": status.strip(),
            "comments": comments.strip(),
            "created_at": ts,
            "updated_at": ts,
        }
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO projects (
                    id, portfolio_id, name, indication, modality, stage, owner, reviewer, status, comments, created_at, updated_at
                ) VALUES (
                    :id, :portfolio_id, :name, :indication, :modality, :stage, :owner, :reviewer, :status, :comments, :created_at, :updated_at
                )
                """,
                row,
            )
            conn.commit()
        self._log_activity("project", row["id"], "created", row, owner)
        return row

    def list_projects(self, portfolio_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM projects WHERE portfolio_id = ? ORDER BY updated_at DESC, name ASC",
                (portfolio_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def add_target_candidate(
        self,
        project_id: str,
        target_id: str,
        alias: str = "",
        rationale: str = "",
        priority: int = 50,
        owner: str = "",
        reviewer: str = "",
        notes: str = "",
    ) -> Dict[str, Any]:
        ts = self._now()
        row = {
            "id": self._id("cand"),
            "project_id": project_id,
            "target_id": target_id.strip(),
            "alias": alias.strip(),
            "rationale": rationale.strip(),
            "priority": int(priority),
            "owner": owner.strip(),
            "reviewer": reviewer.strip(),
            "notes": notes.strip(),
            "created_at": ts,
            "updated_at": ts,
        }
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO target_candidates (
                    id, project_id, target_id, alias, rationale, priority, owner, reviewer, notes, created_at, updated_at
                ) VALUES (
                    :id, :project_id, :target_id, :alias, :rationale, :priority, :owner, :reviewer, :notes, :created_at, :updated_at
                )
                """,
                row,
            )
            conn.commit()
        self._invalidate_project_cache(project_id)
        self._log_activity("target_candidate", row["id"], "created", row, owner)
        return row

    def list_target_candidates(self, project_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM target_candidates
                WHERE project_id = ?
                ORDER BY priority DESC, target_id ASC, created_at ASC
                """,
                (project_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def save_evidence_snapshot(
        self,
        target_candidate_id: str,
        source_version: str,
        component_scores: Dict[str, Any],
        confidence: float,
        completeness: float,
        key_findings: Optional[Dict[str, Any]] = None,
        evidence_volume: Optional[int] = None,
        trial_status: str = "",
        reviewer: str = "",
        comments: str = "",
    ) -> Dict[str, Any]:
        ts = self._now()
        findings = key_findings or {}
        volume = evidence_volume if evidence_volume is not None else int(len(findings.get("highlights", [])))
        row = {
            "id": self._id("evs"),
            "target_candidate_id": target_candidate_id,
            "timestamp": ts,
            "source_version": source_version,
            "component_scores_json": json.dumps(component_scores, sort_keys=True),
            "confidence": float(confidence),
            "completeness": float(completeness),
            "key_findings_json": json.dumps(findings, sort_keys=True),
            "evidence_volume": int(volume),
            "trial_status": trial_status.strip(),
            "reviewer": reviewer.strip(),
            "comments": comments.strip(),
        }
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO evidence_snapshots (
                    id, target_candidate_id, timestamp, source_version, component_scores_json, confidence, completeness,
                    key_findings_json, evidence_volume, trial_status, reviewer, comments
                ) VALUES (
                    :id, :target_candidate_id, :timestamp, :source_version, :component_scores_json, :confidence, :completeness,
                    :key_findings_json, :evidence_volume, :trial_status, :reviewer, :comments
                )
                """,
                row,
            )
            conn.commit()
        project_id = self._project_id_for_candidate(target_candidate_id)
        self._invalidate_project_cache(project_id)
        self._log_activity("evidence_snapshot", row["id"], "created", row, reviewer)
        drift = self._detect_drift(target_candidate_id)
        return {"snapshot": row, "drift": drift}

    def get_target_time_series(self, target_candidate_id: str) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM evidence_snapshots
                WHERE target_candidate_id = ?
                ORDER BY timestamp ASC
                """,
                (target_candidate_id,),
            ).fetchall()
        snapshots = [self._parse_snapshot_row(dict(r)) for r in rows]
        drift_events = []
        for i in range(1, len(snapshots)):
            drift_events.append(self._classify_drift_event(snapshots[i - 1], snapshots[i]))
        return {"target_candidate_id": target_candidate_id, "snapshots": snapshots, "drift_events": drift_events}

    def compare_candidates(self, project_id: str, candidate_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        ids = sorted(candidate_ids) if candidate_ids else []
        cache_key = f"{project_id}:{'|'.join(ids)}"
        if cache_key in self._comparison_cache:
            return self._comparison_cache[cache_key]

        candidates = self.list_target_candidates(project_id)
        if candidate_ids:
            allowed = set(candidate_ids)
            candidates = [c for c in candidates if c["id"] in allowed]

        rows: List[Dict[str, Any]] = []
        for c in candidates:
            series = self.get_target_time_series(c["id"])["snapshots"]
            latest = series[-1] if series else {}
            dims = self._normalize_dimensions(latest.get("component_scores", {}))
            confidence = float(latest.get("confidence", 0.0))
            completeness = float(latest.get("completeness", 0.0))
            missing_penalty = max(0.0, (1.0 - completeness) * 35.0)
            risk_burden = dims["risk_burden"]
            aggregate = (
                0.17 * dims["biological_strength"]
                + 0.14 * dims["pathway_ppi_relevance"]
                + 0.16 * dims["ligandability_druggability"]
                + 0.18 * dims["translational_evidence"]
                + 0.16 * dims["clinical_evidence_maturity"]
                + 0.12 * dims["data_quality_confidence"]
                + 0.07 * (100.0 - risk_burden)
            ) - missing_penalty
            rank_score = round(max(0.0, min(100.0, aggregate)), 2)
            rows.append(
                {
                    "candidate_id": c["id"],
                    "target_id": c["target_id"],
                    "alias": c["alias"] or c["target_id"],
                    "priority": int(c.get("priority", 0)),
                    "dimensions": dims,
                    "confidence": round(confidence, 3),
                    "completeness": round(completeness, 3),
                    "missing_data_penalty": round(missing_penalty, 2),
                    "rank_score": rank_score,
                    "strengths": self._dimension_strengths(dims),
                    "weaknesses": self._dimension_weaknesses(dims),
                }
            )

        ranked = sorted(
            rows,
            key=lambda row: (-row["rank_score"], -row["confidence"], -row["priority"], row["target_id"].lower()),
        )
        matrix = []
        for row in ranked:
            dim = row["dimensions"]
            matrix.append(
                {
                    "candidate_id": row["candidate_id"],
                    "candidate": row["alias"],
                    "biological_strength": dim["biological_strength"],
                    "pathway_ppi_relevance": dim["pathway_ppi_relevance"],
                    "ligandability_druggability": dim["ligandability_druggability"],
                    "translational_evidence": dim["translational_evidence"],
                    "clinical_evidence_maturity": dim["clinical_evidence_maturity"],
                    "data_quality_confidence": dim["data_quality_confidence"],
                    "risk_burden": dim["risk_burden"],
                    "rank_score": row["rank_score"],
                    "confidence": row["confidence"],
                    "missing_data_penalty": row["missing_data_penalty"],
                }
            )

        output = {"project_id": project_id, "comparison_table": matrix, "ranked_summary": ranked}
        self._comparison_cache[cache_key] = output
        return output

    def create_milestone(
        self,
        project_id: str,
        title: str,
        milestone_type: str,
        due_date: str,
        owner: str = "",
        status: str = "not started",
        criteria: Optional[Dict[str, Any]] = None,
        reviewer: str = "",
        comments: str = "",
    ) -> Dict[str, Any]:
        ts = self._now()
        row = {
            "id": self._id("ms"),
            "project_id": project_id,
            "title": title.strip(),
            "type": milestone_type.strip(),
            "due_date": due_date,
            "owner": owner.strip(),
            "reviewer": reviewer.strip(),
            "status": status.strip().lower(),
            "criteria_json": json.dumps(criteria or {}, sort_keys=True),
            "comments": comments.strip(),
            "created_at": ts,
            "updated_at": ts,
        }
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO milestones (
                    id, project_id, title, type, due_date, owner, reviewer, status, criteria_json, comments, created_at, updated_at
                ) VALUES (
                    :id, :project_id, :title, :type, :due_date, :owner, :reviewer, :status, :criteria_json, :comments, :created_at, :updated_at
                )
                """,
                row,
            )
            conn.commit()
        self._log_activity("milestone", row["id"], "created", row, owner)
        return row

    def update_milestone_status(self, milestone_id: str, status: str, actor: str = "") -> Dict[str, Any]:
        ts = self._now()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE milestones SET status = ?, updated_at = ? WHERE id = ?",
                (status.strip().lower(), ts, milestone_id),
            )
            conn.commit()
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM milestones WHERE id = ?", (milestone_id,)).fetchone()
        payload = dict(row) if row else {}
        self._log_activity("milestone", milestone_id, "status_updated", {"status": status}, actor)
        return payload

    def generate_decision_snapshot(self, project_id: str, framework: str = "default") -> Dict[str, Any]:
        dashboard = self.get_project_dashboard_data(project_id)
        comparison = self.compare_candidates(project_id)
        top_candidate = comparison["ranked_summary"][0] if comparison["ranked_summary"] else None
        rules = dict(self.DEFAULT_DECISION_RULES)
        evidence_conf = float(top_candidate["confidence"]) if top_candidate else 0.0
        translational = float(top_candidate["dimensions"]["translational_evidence"]) if top_candidate else 0.0
        risk_burden = float(top_candidate["dimensions"]["risk_burden"]) if top_candidate else 100.0
        milestone_completion = float(dashboard["milestone_metrics"]["completion_pct"]) / 100.0
        blockers = dashboard["milestone_metrics"]["blocker_count"]

        checks = {
            "min_evidence_confidence": evidence_conf >= rules["min_evidence_confidence"],
            "min_translational_score": translational >= rules["min_translational_score"],
            "max_risk_tolerance": risk_burden <= rules["max_risk_burden"],
            "milestone_gate_completion": milestone_completion >= rules["min_milestone_completion"],
            "no_blocked_milestones": blockers == 0,
        }
        pass_count = sum(1 for ok in checks.values() if ok)
        confidence = round(min(1.0, max(0.0, (pass_count / len(checks)) * 0.75 + evidence_conf * 0.25)), 3)

        hard_blockers: List[str] = []
        if not checks["max_risk_tolerance"]:
            hard_blockers.append("Risk burden exceeds tolerance")
        if not checks["no_blocked_milestones"]:
            hard_blockers.append("One or more milestones are blocked")

        if pass_count == len(checks):
            recommendation = "GO"
        elif hard_blockers or pass_count <= 2:
            recommendation = "NO-GO"
        else:
            recommendation = "HOLD"

        assumptions = [
            "Current evidence quality remains stable over next review cycle",
            "Milestone owners can close open gaps before due dates",
            "No external safety signal reversal in target class",
        ]
        next_actions = [
            "Run orthogonal assay to confirm translational signal",
            "Close top blocked milestone with explicit acceptance evidence",
            "Refresh literature and trial evidence feed in 2-4 weeks",
        ]
        risks = top_candidate["weaknesses"] if top_candidate else ["No candidate evidence available"]

        narrative = (
            f"Recommendation {recommendation}: checks passed {pass_count}/{len(checks)}. "
            f"Evidence confidence={evidence_conf:.2f}, translational score={translational:.1f}, "
            f"risk burden={risk_burden:.1f}, milestone completion={milestone_completion*100:.1f}%."
        )

        snapshot = {
            "id": self._id("dec"),
            "project_id": project_id,
            "timestamp": self._now(),
            "recommendation": recommendation,
            "confidence": confidence,
            "rationale": narrative,
            "risks": risks,
            "next_actions": next_actions,
            "framework": framework,
            "checks": checks,
            "assumptions": assumptions,
            "hard_blockers": hard_blockers,
        }
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO decision_snapshots (
                    id, project_id, timestamp, recommendation, confidence, rationale, risks_json, next_actions_json,
                    framework, checks_json, assumptions_json, blockers_json, reviewer, comments
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot["id"],
                    project_id,
                    snapshot["timestamp"],
                    snapshot["recommendation"],
                    snapshot["confidence"],
                    snapshot["rationale"],
                    json.dumps(snapshot["risks"], sort_keys=True),
                    json.dumps(snapshot["next_actions"], sort_keys=True),
                    framework,
                    json.dumps(snapshot["checks"], sort_keys=True),
                    json.dumps(snapshot["assumptions"], sort_keys=True),
                    json.dumps(snapshot["hard_blockers"], sort_keys=True),
                    "",
                    "",
                ),
            )
            conn.commit()
        self._log_activity("decision_snapshot", snapshot["id"], "created", snapshot, "")
        return snapshot

    def get_project_dashboard_data(self, project_id: str) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            project_row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
            milestone_rows = conn.execute("SELECT * FROM milestones WHERE project_id = ?", (project_id,)).fetchall()
            decision_row = conn.execute(
                "SELECT * FROM decision_snapshots WHERE project_id = ? ORDER BY timestamp DESC LIMIT 1",
                (project_id,),
            ).fetchone()
        project = dict(project_row) if project_row else {}
        milestones = [self._parse_milestone(dict(r)) for r in milestone_rows]
        comparison = self.compare_candidates(project_id)
        milestone_metrics = self._milestone_metrics(milestones)
        pipeline = {
            "candidate_count": len(comparison["ranked_summary"]),
            "average_rank_score": round(
                sum(r["rank_score"] for r in comparison["ranked_summary"]) / max(1, len(comparison["ranked_summary"])),
                2,
            ),
            "top_candidate": comparison["ranked_summary"][0] if comparison["ranked_summary"] else None,
        }
        latest_decision = self._parse_decision(dict(decision_row)) if decision_row else None
        alerts = self._project_alerts(project_id)
        return {
            "project": project,
            "candidate_comparison": comparison,
            "milestones": milestones,
            "milestone_metrics": milestone_metrics,
            "pipeline": pipeline,
            "latest_decision_snapshot": latest_decision,
            "drift_alerts": alerts,
            "disclaimer": "Research portfolio decision support only. Not for clinical or patient-care decisions.",
        }

    def export_project_packet(self, project_id: str, format: str = "json") -> Dict[str, Any]:
        dashboard = self.get_project_dashboard_data(project_id)
        decision = dashboard["latest_decision_snapshot"] or self.generate_decision_snapshot(project_id)
        payload = {
            "project": dashboard["project"],
            "milestone_metrics": dashboard["milestone_metrics"],
            "milestones": dashboard["milestones"],
            "candidate_ranking": dashboard["candidate_comparison"]["ranked_summary"],
            "comparison_table": dashboard["candidate_comparison"]["comparison_table"],
            "drift_alerts": dashboard["drift_alerts"],
            "decision": decision,
            "generated_at": self._now(),
            "disclaimer": dashboard["disclaimer"],
        }
        fmt = format.lower()
        if fmt == "json":
            return {"format": "json", "content": json.dumps(payload, indent=2, sort_keys=False), "schema_valid": True}
        if fmt == "csv":
            return {"format": "csv", "content": self._build_csv_packet(payload), "schema_valid": True}
        if fmt == "md":
            return {"format": "md", "content": self._build_markdown_packet(payload), "schema_valid": True}
        raise ValueError("Unsupported format. Use one of: json, csv, md")

    def get_stage_distribution(self, portfolio_id: str) -> List[Dict[str, Any]]:
        projects = self.list_projects(portfolio_id)
        counter: Dict[str, int] = {}
        for p in projects:
            stage = p.get("stage") or "unknown"
            counter[stage] = counter.get(stage, 0) + 1
        return [{"stage": stage, "count": count} for stage, count in sorted(counter.items())]

    def list_recent_activity(self, limit: int = 40) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM activity_log ORDER BY timestamp DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def _project_alerts(self, project_id: str) -> List[Dict[str, Any]]:
        alerts = []
        for candidate in self.list_target_candidates(project_id):
            drift = self._detect_drift(candidate["id"])
            if drift["classification"] != "Neutral":
                alerts.append({"candidate_id": candidate["id"], "target_id": candidate["target_id"], **drift})
        return alerts

    def _detect_drift(self, target_candidate_id: str) -> Dict[str, Any]:
        series = self.get_target_time_series(target_candidate_id)["snapshots"]
        if len(series) < 2:
            return {
                "classification": "Neutral",
                "delta": {},
                "change_log": ["No prior snapshot available for drift comparison"],
            }
        return self._classify_drift_event(series[-2], series[-1])

    def _classify_drift_event(self, previous: Dict[str, Any], latest: Dict[str, Any]) -> Dict[str, Any]:
        prev_scores = previous.get("component_scores", {})
        new_scores = latest.get("component_scores", {})
        prev_mean = self._mean_score(prev_scores)
        new_mean = self._mean_score(new_scores)
        score_delta = new_mean - prev_mean
        confidence_delta = float(latest.get("confidence", 0.0)) - float(previous.get("confidence", 0.0))
        completeness_delta = float(latest.get("completeness", 0.0)) - float(previous.get("completeness", 0.0))
        evidence_volume_delta = int(latest.get("evidence_volume", 0)) - int(previous.get("evidence_volume", 0))

        changes: List[str] = []
        if score_delta >= self.drift_config["score_delta"]:
            changes.append("Composite score increased materially")
        elif score_delta <= -self.drift_config["score_delta"]:
            changes.append("Composite score declined materially")

        if confidence_delta >= self.drift_config["confidence_delta"]:
            changes.append("Confidence increased with stronger evidence quality")
        elif confidence_delta <= -self.drift_config["confidence_delta"]:
            changes.append("Confidence dropped due to missing or conflicting source")

        if completeness_delta >= self.drift_config["completeness_delta"]:
            changes.append("Evidence coverage expanded across components")
        elif completeness_delta <= -self.drift_config["completeness_delta"]:
            changes.append("Evidence coverage regressed due to missing component inputs")

        if evidence_volume_delta >= self.drift_config["evidence_volume_delta"]:
            changes.append("Evidence volume increased with new observations")
        elif evidence_volume_delta <= -self.drift_config["evidence_volume_delta"]:
            changes.append("Evidence volume dropped versus prior snapshot")

        prev_trial = (previous.get("trial_status") or "").upper()
        new_trial = (latest.get("trial_status") or "").upper()
        if prev_trial != new_trial and new_trial:
            if self._trial_rank(new_trial) > self._trial_rank(prev_trial):
                changes.append("Trial phase advanced")
            else:
                changes.append("New conflicting evidence from trial status change")

        polarity = score_delta + (confidence_delta * 25.0) + (completeness_delta * 18.0)
        if polarity >= 6.0:
            classification = "Positive"
        elif polarity <= -6.0:
            classification = "Negative"
        else:
            classification = "Neutral"

        return {
            "classification": classification,
            "delta": {
                "score_delta": round(score_delta, 2),
                "confidence_delta": round(confidence_delta, 3),
                "completeness_delta": round(completeness_delta, 3),
                "evidence_volume_delta": int(evidence_volume_delta),
            },
            "change_log": changes or ["No meaningful drift detected"],
            "timestamp": latest.get("timestamp"),
        }

    def _build_csv_packet(self, payload: Dict[str, Any]) -> Dict[str, str]:
        output = {}
        output["candidate_ranking"] = self._rows_to_csv(payload["candidate_ranking"])
        output["comparison_table"] = self._rows_to_csv(payload["comparison_table"])
        output["milestones"] = self._rows_to_csv(payload["milestones"])
        output["drift_alerts"] = self._rows_to_csv(payload["drift_alerts"])
        output["decision"] = self._rows_to_csv([payload["decision"]])
        return output

    def _build_markdown_packet(self, payload: Dict[str, Any]) -> str:
        project = payload["project"]
        decision = payload["decision"]
        lines = [
            f"# Project Packet: {project.get('name', 'Unknown')}",
            "",
            f"- Project ID: `{project.get('id', '')}`",
            f"- Stage: `{project.get('stage', '')}`",
            f"- Status: `{project.get('status', '')}`",
            f"- Owner: `{project.get('owner', '')}`",
            f"- Generated: `{payload.get('generated_at', '')}`",
            "",
            "## Decision Snapshot",
            f"- Recommendation: **{decision.get('recommendation', 'HOLD')}**",
            f"- Confidence: `{decision.get('confidence', 0.0)}`",
            f"- Rationale: {decision.get('rationale', '')}",
            "",
            "## Candidate Ranking",
        ]
        for idx, row in enumerate(payload["candidate_ranking"], start=1):
            lines.append(
                f"{idx}. `{row['alias']}` score={row['rank_score']} confidence={row['confidence']} "
                f"strengths={'; '.join(row['strengths'][:2])}"
            )
        lines.append("")
        lines.append("## Milestone Metrics")
        metrics = payload["milestone_metrics"]
        lines.append(f"- Completion: `{metrics['completion_pct']}%`")
        lines.append(f"- Overdue: `{metrics['overdue_count']}`")
        lines.append(f"- Blocked: `{metrics['blocker_count']}`")
        lines.append(f"- Upcoming (30d): `{metrics['upcoming_30d']}`")
        lines.append("")
        lines.append(f"> {payload['disclaimer']}")
        return "\n".join(lines)

    def _rows_to_csv(self, rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return ""
        fieldnames = sorted({k for row in rows for k in row.keys()})
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: self._csv_scalar(row.get(k)) for k in fieldnames})
        return stream.getvalue()

    def _normalize_dimensions(self, component_scores: Dict[str, Any]) -> Dict[str, float]:
        def score(name: str, default: float = 0.0) -> float:
            entry = component_scores.get(name, {})
            if isinstance(entry, dict):
                return float(entry.get("score", default))
            return float(default)

        biological = score("expression")
        pathway = (score("pathway") + score("ppi")) / 2.0
        ligand = score("ligandability")
        translational = score("genetic")
        clinical = score("trials")
        source_quality = []
        for entry in component_scores.values():
            if isinstance(entry, dict) and "source_quality" in entry:
                source_quality.append(float(entry["source_quality"]) * 100.0)
        quality = sum(source_quality) / len(source_quality) if source_quality else 0.0
        risk = max(0.0, min(100.0, 100.0 - ((translational * 0.5) + (clinical * 0.5))))
        return {
            "biological_strength": round(biological, 2),
            "pathway_ppi_relevance": round(pathway, 2),
            "ligandability_druggability": round(ligand, 2),
            "translational_evidence": round(translational, 2),
            "clinical_evidence_maturity": round(clinical, 2),
            "data_quality_confidence": round(quality, 2),
            "risk_burden": round(risk, 2),
        }

    def _dimension_strengths(self, dims: Dict[str, float]) -> List[str]:
        ranked = sorted(dims.items(), key=lambda kv: kv[1], reverse=True)
        return [f"{name.replace('_', ' ')} strong ({value:.1f})" for name, value in ranked[:3] if name != "risk_burden"]

    def _dimension_weaknesses(self, dims: Dict[str, float]) -> List[str]:
        weaknesses = []
        ordered = sorted(dims.items(), key=lambda kv: kv[1])
        for name, value in ordered[:2]:
            if name == "risk_burden":
                weaknesses.append(f"Risk burden elevated ({value:.1f})")
            else:
                weaknesses.append(f"{name.replace('_', ' ')} weak ({value:.1f})")
        if dims["risk_burden"] > 70:
            weaknesses.append("Risk burden exceeds portfolio threshold")
        return weaknesses

    def _milestone_metrics(self, milestones: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(milestones)
        complete = sum(1 for m in milestones if (m.get("status") or "").lower() == "complete")
        blocked = sum(1 for m in milestones if (m.get("status") or "").lower() == "blocked")
        today = datetime.now(timezone.utc).date()
        upcoming_limit = today + timedelta(days=30)
        overdue = 0
        upcoming = 0
        for m in milestones:
            due = m.get("due_date")
            due_date = self._try_date(due)
            if not due_date:
                continue
            if due_date < today and (m.get("status") or "").lower() != "complete":
                overdue += 1
            if today <= due_date <= upcoming_limit:
                upcoming += 1
        completion_pct = (complete / total * 100.0) if total else 0.0
        return {
            "total_count": total,
            "completion_pct": round(completion_pct, 2),
            "overdue_count": overdue,
            "blocker_count": blocked,
            "upcoming_30d": upcoming,
        }

    def _parse_snapshot_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row["component_scores"] = json.loads(row.get("component_scores_json", "{}"))
        row["key_findings"] = json.loads(row.get("key_findings_json", "{}"))
        return row

    def _parse_milestone(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row["criteria"] = json.loads(row.get("criteria_json", "{}"))
        return row

    def _parse_decision(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row["risks"] = json.loads(row.get("risks_json", "[]"))
        row["next_actions"] = json.loads(row.get("next_actions_json", "[]"))
        row["checks"] = json.loads(row.get("checks_json", "{}"))
        row["assumptions"] = json.loads(row.get("assumptions_json", "[]"))
        row["hard_blockers"] = json.loads(row.get("blockers_json", "[]"))
        return row

    def _log_activity(self, entity_type: str, entity_id: str, event_type: str, payload: Dict[str, Any], actor: str) -> None:
        row = {
            "id": self._id("log"),
            "entity_type": entity_type,
            "entity_id": entity_id,
            "event_type": event_type,
            "payload_json": json.dumps(payload, sort_keys=True, default=str),
            "timestamp": self._now(),
            "actor": actor,
        }
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO activity_log (id, entity_type, entity_id, event_type, payload_json, timestamp, actor)
                VALUES (:id, :entity_type, :entity_id, :event_type, :payload_json, :timestamp, :actor)
                """,
                row,
            )
            conn.commit()

    def _project_id_for_candidate(self, candidate_id: str) -> str:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT project_id FROM target_candidates WHERE id = ?", (candidate_id,)).fetchone()
        return row[0] if row else ""

    def _invalidate_project_cache(self, project_id: str) -> None:
        keys = [k for k in self._comparison_cache if k.startswith(f"{project_id}:")]
        for key in keys:
            self._comparison_cache.pop(key, None)

    def _mean_score(self, component_scores: Dict[str, Any]) -> float:
        values = []
        for entry in component_scores.values():
            if isinstance(entry, dict) and entry.get("available", True):
                values.append(float(entry.get("score", 0.0)))
        return sum(values) / len(values) if values else 0.0

    def _trial_rank(self, status: str) -> int:
        status = (status or "").upper()
        if "PHASE 4" in status or "PHASE4" in status:
            return 5
        if "PHASE 3" in status or "PHASE3" in status:
            return 4
        if "PHASE 2" in status or "PHASE2" in status:
            return 3
        if "PHASE 1" in status or "PHASE1" in status:
            return 2
        if "PRECLIN" in status:
            return 1
        return 0

    def _csv_scalar(self, value: Any) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True)
        return "" if value is None else str(value)

    def _try_date(self, value: Any) -> Optional[datetime.date]:
        if not value:
            return None
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(str(value)[:19], fmt).date()
            except ValueError:
                continue
        return None

    def _now(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

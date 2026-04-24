"""Target prioritization engine for transparent target ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComponentConfig:
    """Configuration metadata for one target component."""

    key: str
    display_name: str


class TargetPrioritizationEngine:
    """Compute explainable target prioritization scores in range 0-100."""

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "expression": 0.20,
        "pathway": 0.15,
        "ppi": 0.15,
        "genetic": 0.20,
        "ligandability": 0.20,
        "trials": 0.10,
    }

    COMPONENTS: Tuple[ComponentConfig, ...] = (
        ComponentConfig("expression", "Expression relevance"),
        ComponentConfig("pathway", "Pathway centrality"),
        ComponentConfig("ppi", "PPI topology"),
        ComponentConfig("genetic", "Genetic risk evidence"),
        ComponentConfig("ligandability", "Ligandability"),
        ComponentConfig("trials", "Clinical trial evidence"),
    )

    def compute_component_scores(self, input_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Compute all component scores with details and risk flags."""

        component_scores = {
            "expression": self._score_expression(input_data.get("expression_data")),
            "pathway": self._score_pathway(input_data.get("pathway_data")),
            "ppi": self._score_ppi(input_data.get("ppi_data")),
            "genetic": self._score_genetic(input_data.get("genetic_data")),
            "ligandability": self._score_ligandability(input_data.get("ligandability_data")),
            "trials": self._score_trials(input_data.get("trial_data")),
        }
        return component_scores

    def compute_composite_score(
        self,
        component_scores: Dict[str, Dict[str, Any]],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Compute weighted composite score with missing-data renormalization."""

        resolved_weights = self._normalize_weight_input(weights)
        present_keys = [
            key
            for key in resolved_weights
            if component_scores.get(key, {}).get("available") is True
        ]
        missing_keys = [key for key in resolved_weights if key not in present_keys]

        if not present_keys:
            return {
                "composite_score": 0.0,
                "confidence_score": 0.0,
                "data_completeness": 0.0,
                "missing_components": missing_keys,
                "renormalized_weights": {},
                "component_contributions": {},
            }

        present_weight_sum = sum(resolved_weights[k] for k in present_keys)
        renormalized_weights = {
            k: resolved_weights[k] / present_weight_sum for k in present_keys
        }

        contributions: Dict[str, Dict[str, float]] = {}
        composite_score = 0.0
        source_quality_weighted = 0.0
        for key, renorm_weight in renormalized_weights.items():
            score = float(component_scores[key]["score"])
            weighted_contribution = score * renorm_weight
            contributions[key] = {
                "score": round(score, 2),
                "weight": round(renorm_weight, 4),
                "weighted_contribution": round(weighted_contribution, 4),
            }
            composite_score += weighted_contribution
            source_quality_weighted += renorm_weight * float(
                component_scores[key].get("source_quality", 0.5)
            )

        completeness = (len(present_keys) / len(self.DEFAULT_WEIGHTS)) * 100.0
        confidence_score = (0.65 * completeness) + (35.0 * source_quality_weighted)

        if confidence_score < 45.0:
            LOGGER.warning("Low-confidence prioritization score detected: %.2f", confidence_score)

        return {
            "composite_score": round(self._clamp(composite_score), 2),
            "confidence_score": round(self._clamp(confidence_score), 2),
            "data_completeness": round(self._clamp(completeness), 2),
            "missing_components": missing_keys,
            "renormalized_weights": {k: round(v, 4) for k, v in renormalized_weights.items()},
            "component_contributions": contributions,
        }

    def rank_targets(
        self,
        target_payloads: List[Dict[str, Any]],
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Rank targets by composite score (stable deterministic tie-breaking)."""

        results: List[Dict[str, Any]] = []
        for idx, payload in enumerate(target_payloads):
            component_scores = self.compute_component_scores(payload)
            composite = self.compute_composite_score(component_scores, weights=weights)
            merged = {
                "target_id": payload.get("target_id") or payload.get("uniprot_id") or payload.get("gene_name") or f"target_{idx}",
                "input_data": payload,
                "component_scores": component_scores,
                **composite,
            }
            merged["explainability"] = self.explain_score(merged)
            results.append(merged)

        return sorted(
            results,
            key=lambda row: (-row["composite_score"], -row["confidence_score"], str(row["target_id"]).lower()),
        )

    def explain_score(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explainability output for one scored target."""

        component_scores = result.get("component_scores", {})
        contributions = result.get("component_contributions", {})
        breakdown: List[Dict[str, Any]] = []

        for comp in self.COMPONENTS:
            contribution = contributions.get(comp.key, {})
            comp_raw = component_scores.get(comp.key, {})
            breakdown.append(
                {
                    "component": comp.key,
                    "label": comp.display_name,
                    "available": bool(comp_raw.get("available", False)),
                    "score": round(float(comp_raw.get("score", 0.0)), 2),
                    "weight": round(float(contribution.get("weight", 0.0)), 4),
                    "weighted_contribution": round(float(contribution.get("weighted_contribution", 0.0)), 2),
                    "notes": comp_raw.get("notes", []),
                }
            )

        sorted_positive = sorted(
            [b for b in breakdown if b["available"]],
            key=lambda row: row["weighted_contribution"],
            reverse=True,
        )
        top_drivers = [
            f"{b['label']} ({b['weighted_contribution']:.1f})" for b in sorted_positive[:3]
        ]

        risk_flags: List[str] = []
        for comp in self.COMPONENTS:
            risk_flags.extend(component_scores.get(comp.key, {}).get("risk_flags", []))
        risk_flags = risk_flags[:3]

        improvements = self._improvement_suggestions(component_scores, result.get("missing_components", []))
        rationale = self._build_rationale(result, top_drivers, risk_flags, improvements)

        return {
            "breakdown": breakdown,
            "top_positive_drivers": top_drivers,
            "top_risk_flags": risk_flags,
            "improvement_suggestions": improvements,
            "rationale": rationale,
        }

    def sensitivity_analysis(
        self,
        target_payload: Dict[str, Any],
        weight_scenarios: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """Evaluate score sensitivity under named weight scenarios."""

        component_scores = self.compute_component_scores(target_payload)
        baseline = self.compute_composite_score(component_scores, weights=None)
        scenario_scores: Dict[str, Any] = {
            "baseline": baseline["composite_score"],
        }

        for scenario_name, scenario_weights in weight_scenarios.items():
            scenario_output = self.compute_composite_score(component_scores, scenario_weights)
            scenario_scores[scenario_name] = scenario_output["composite_score"]

        deltas = {
            name: round(score - baseline["composite_score"], 2)
            for name, score in scenario_scores.items()
            if name != "baseline"
        }

        return {
            "target_id": target_payload.get("target_id") or target_payload.get("uniprot_id") or target_payload.get("gene_name"),
            "baseline_score": baseline["composite_score"],
            "scenario_scores": scenario_scores,
            "scenario_deltas": deltas,
            "max_delta": max([abs(v) for v in deltas.values()], default=0.0),
        }

    def _score_expression(self, expression_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not expression_data:
            return self._missing_component("expression", ["No expression data available"])

        tissue_rows = expression_data.get("tissues", [])
        disease_tissues = {str(t).lower() for t in expression_data.get("disease_tissues", [])}
        levels = [self._as_float(row.get("level_numeric")) for row in tissue_rows if isinstance(row, dict)]
        levels = [lv for lv in levels if lv is not None]
        if not levels:
            return self._missing_component("expression", ["Expression rows missing numeric levels"])

        avg_level_norm = (sum(levels) / len(levels)) / 3.0
        signal_strength = max(levels) / 3.0
        consistency = 1.0 - (min(1.0, self._std(levels) / 1.5))

        enrichment_hits = 0
        enrichment_total = max(1, len(disease_tissues))
        for row in tissue_rows:
            tissue_name = str(row.get("tissue", "")).lower()
            if tissue_name in disease_tissues and self._as_float(row.get("level_numeric"), 0.0) >= 2.0:
                enrichment_hits += 1
        enrichment = enrichment_hits / enrichment_total if disease_tissues else 0.5

        score = 100.0 * (0.40 * signal_strength + 0.30 * enrichment + 0.30 * consistency)
        return {
            "available": True,
            "score": round(self._clamp(score), 2),
            "source_quality": 0.85,
            "notes": [
                f"Avg expression level={avg_level_norm*100:.1f}%",
                f"Disease tissue enrichment={enrichment*100:.1f}%",
                f"Cross-tissue consistency={consistency*100:.1f}%",
            ],
            "risk_flags": [] if enrichment >= 0.4 else ["Weak disease-tissue enrichment"],
        }

    def _score_pathway(self, pathway_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not pathway_data or not pathway_data.get("available"):
            return self._missing_component("pathway", ["No pathway mapping available"])

        pathways = pathway_data.get("pathways", [])
        count = len(pathways)
        if count == 0:
            return self._missing_component("pathway", ["Pathway list empty"])

        membership = min(1.0, count / 12.0)
        avg_importance = self._avg(
            [self._as_float(p.get("importance_score"), 0.55) for p in pathways if isinstance(p, dict)],
            default=0.55,
        )
        centrality = self._as_float(pathway_data.get("graph_centrality"), default=min(1.0, 0.3 + (count * 0.05)))

        score = 100.0 * (0.45 * membership + 0.30 * self._clamp01(avg_importance) + 0.25 * self._clamp01(centrality))
        return {
            "available": True,
            "score": round(self._clamp(score), 2),
            "source_quality": 0.75,
            "notes": [
                f"Pathway membership count={count}",
                f"Pathway importance proxy={self._clamp01(avg_importance)*100:.1f}%",
                f"Centrality proxy={self._clamp01(centrality)*100:.1f}%",
            ],
            "risk_flags": [] if count >= 3 else ["Low pathway coverage"],
        }

    def _score_ppi(self, ppi_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not ppi_data or not ppi_data.get("available"):
            return self._missing_component("ppi", ["No PPI data available"])

        interactions = ppi_data.get("interactions", [])
        if not interactions:
            return self._missing_component("ppi", ["No STRING interactions found"])

        degree = min(1.0, len(interactions) / 20.0)
        confidence_vals = [self._as_float(i.get("combined_score"), 0.0) / 1000.0 for i in interactions if isinstance(i, dict)]
        confidence = self._avg(confidence_vals, default=0.4)
        betweenness_proxy = min(1.0, (degree * confidence) + 0.1)
        clustering_proxy = max(0.0, min(1.0, 0.2 + 0.8 * confidence))

        score = 100.0 * (0.35 * degree + 0.25 * betweenness_proxy + 0.20 * clustering_proxy + 0.20 * confidence)
        return {
            "available": True,
            "score": round(self._clamp(score), 2),
            "source_quality": 0.8,
            "notes": [
                f"Degree proxy={degree*100:.1f}%",
                f"Betweenness proxy={betweenness_proxy*100:.1f}%",
                f"Clustering proxy={clustering_proxy*100:.1f}%",
                f"Interaction confidence={confidence*100:.1f}%",
            ],
            "risk_flags": [] if confidence >= 0.5 else ["Low-confidence interaction network"],
        }

    def _score_genetic(self, genetic_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not genetic_data:
            return self._missing_component("genetic", ["No genetic evidence available"])

        fusion = genetic_data.get("multiomics_fusion", {})
        fusion_prob = self._as_float(fusion.get("predicted_response_probability"))
        fusion_uncertainty = self._as_float(fusion.get("uncertainty"))
        fusion_prob_quality = (
            self._clamp01(1.0 - fusion_uncertainty)
            if fusion_uncertainty is not None
            else None
        )

        mutation = genetic_data.get("mutation_analysis", {})
        biomarkers = genetic_data.get("biomarker_detection", {})
        associations = genetic_data.get("disease_associations", {})

        total_variants = self._as_float(mutation.get("total_variants"), 0.0)
        high_risk_variants = self._as_float(mutation.get("high_risk_variants"), 0.0)
        therapeutic_targets = self._as_float(biomarkers.get("therapeutic_targets"), 0.0)
        high_confidence_assoc = self._as_float(associations.get("high_confidence"), 0.0)
        moderate_assoc = self._as_float(associations.get("moderate_confidence"), 0.0)

        variant_burden = self._clamp01((0.7 * high_risk_variants + 0.3 * total_variants) / 8.0)
        biomarker_support = self._clamp01(therapeutic_targets / 6.0)
        association_confidence = self._clamp01((high_confidence_assoc + 0.5 * moderate_assoc) / 6.0)

        base_score = 100.0 * (0.40 * variant_burden + 0.30 * biomarker_support + 0.30 * association_confidence)
        score = base_score
        notes = [
            f"Variant burden={variant_burden*100:.1f}%",
            f"Biomarker support={biomarker_support*100:.1f}%",
            f"Disease association confidence={association_confidence*100:.1f}%",
        ]
        source_quality = 0.7
        risk_flags = [] if high_risk_variants > 0 else ["No high-risk variant signal"]

        if fusion_prob is not None:
            # Blend pathway-level multi-omics signal while preserving legacy genomics contribution.
            score = (0.7 * base_score) + (0.3 * (self._clamp01(fusion_prob) * 100.0))
            notes.append(
                f"Fusion response probability={self._clamp01(fusion_prob)*100:.1f}%"
            )
        if fusion_prob_quality is not None:
            source_quality = self._clamp01((0.75 * source_quality) + (0.25 * fusion_prob_quality))
            if fusion_prob_quality < 0.45:
                risk_flags.append("High uncertainty in multi-omics fusion evidence")

        return {
            "available": True,
            "score": round(self._clamp(score), 2),
            "source_quality": source_quality,
            "notes": notes,
            "risk_flags": risk_flags,
        }

    def _score_ligandability(self, ligandability_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not ligandability_data:
            return self._missing_component("ligandability", ["No ligandability data available"])

        chembl = ligandability_data.get("chembl", {})
        docking = ligandability_data.get("docking", {})
        prediction = ligandability_data.get("binding_prediction", {})
        fusion = ligandability_data.get("multiomics_fusion", {})

        known_ligands = len(chembl.get("ligands", [])) if chembl.get("available") else 0
        ligand_score = min(1.0, known_ligands / 15.0)

        docking_affinity = self._as_float(docking.get("binding_affinity"))
        docking_quality = 0.0
        if docking_affinity is not None:
            docking_quality = self._clamp01((-docking_affinity - 4.0) / 8.0)

        predicted = prediction.get("ranked_molecules", []) if prediction.get("available") else []
        pred_quality = self._avg([self._as_float(item.get("binding_likelihood"), 0.0) / 100.0 for item in predicted], default=0.0)

        simulated_penalty = 0.15 if docking.get("simulated") else 0.0
        score = 100.0 * (0.45 * ligand_score + 0.35 * docking_quality + 0.20 * pred_quality)
        score = score * (1.0 - simulated_penalty)

        source_quality = 0.8 if known_ligands > 0 else 0.55
        if docking.get("simulated"):
            source_quality -= 0.15
        source_quality = self._clamp01(source_quality)

        flags = []
        if known_ligands == 0:
            flags.append("No known ligands in ChEMBL")
        if docking.get("simulated"):
            flags.append("Docking evidence is simulated-only")

        fusion_prob = self._as_float(fusion.get("predicted_response_probability"))
        fusion_uncertainty = self._as_float(fusion.get("uncertainty"))
        fusion_quality = self._clamp01(1.0 - fusion_uncertainty) if fusion_uncertainty is not None else None
        if fusion_prob is not None:
            score = (0.8 * score) + (0.2 * self._clamp01(fusion_prob) * 100.0)
        if fusion_quality is not None:
            source_quality = self._clamp01((0.8 * source_quality) + (0.2 * fusion_quality))

        notes = [
            f"Known ligand support={ligand_score*100:.1f}%",
            f"Docking quality={docking_quality*100:.1f}%",
            f"Predicted binder quality={pred_quality*100:.1f}%",
            f"Simulated evidence penalty={simulated_penalty*100:.0f}%",
        ]
        if fusion_prob is not None:
            notes.append(f"Fusion response probability={self._clamp01(fusion_prob)*100:.1f}%")
        if fusion_quality is not None and fusion_quality < 0.45:
            flags.append("Fusion response uncertainty is high")

        return {
            "available": bool(known_ligands or docking_affinity is not None or predicted),
            "score": round(self._clamp(score), 2),
            "source_quality": source_quality,
            "notes": notes,
            "risk_flags": flags,
        }

    def _score_trials(self, trial_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not trial_data or not trial_data.get("available"):
            return self._missing_component("trials", ["No trial evidence available"])

        trials = trial_data.get("clinical_trials", [])
        if not trials:
            return self._missing_component("trials", ["No clinical trials found"])

        trial_count_score = min(1.0, len(trials) / 20.0)
        phase_values = []
        status_values = []
        for trial in trials:
            phase_values.append(self._phase_to_score(str(trial.get("phase", "N/A"))))
            status_values.append(self._status_to_score(str(trial.get("status", "Unknown"))))

        phase_score = self._avg(phase_values, default=0.3)
        status_score = self._avg(status_values, default=0.3)
        score = 100.0 * (0.30 * trial_count_score + 0.40 * phase_score + 0.30 * status_score)

        flags = []
        if status_score < 0.45:
            flags.append("Trial statuses are weak/uncertain")
        if phase_score < 0.4:
            flags.append("Evidence concentrated in early or unknown phases")

        return {
            "available": True,
            "score": round(self._clamp(score), 2),
            "source_quality": 0.9,
            "notes": [
                f"Trial count strength={trial_count_score*100:.1f}%",
                f"Phase maturity={phase_score*100:.1f}%",
                f"Status quality={status_score*100:.1f}%",
            ],
            "risk_flags": flags[:3],
        }

    def _normalize_weight_input(self, weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        base = dict(self.DEFAULT_WEIGHTS)
        if not weights:
            return base
        for key in base:
            if key in weights and self._as_float(weights[key]) is not None:
                base[key] = max(0.0, float(weights[key]))
        total = sum(base.values())
        if total <= 0:
            return dict(self.DEFAULT_WEIGHTS)
        return {k: v / total for k, v in base.items()}

    def _phase_to_score(self, phase_value: str) -> float:
        phase = phase_value.upper().replace(" ", "")
        if "PHASE4" in phase:
            return 1.0
        if "PHASE3" in phase:
            return 0.85
        if "PHASE2" in phase and "PHASE3" in phase:
            return 0.8
        if "PHASE2" in phase:
            return 0.65
        if "PHASE1" in phase and "PHASE2" in phase:
            return 0.5
        if "PHASE1" in phase:
            return 0.35
        return 0.2

    def _status_to_score(self, status_value: str) -> float:
        status = status_value.upper().strip()
        if status in {"COMPLETED", "APPROVED"}:
            return 1.0
        if status in {"ACTIVE_NOT_RECRUITING", "RECRUITING"}:
            return 0.75
        if status in {"ENROLLING_BY_INVITATION"}:
            return 0.6
        if "TERMINATED" in status or "WITHDRAWN" in status:
            return 0.15
        if "SUSPENDED" in status:
            return 0.25
        return 0.35

    def _improvement_suggestions(
        self,
        component_scores: Dict[str, Dict[str, Any]],
        missing_components: List[str],
    ) -> List[str]:
        tips: List[str] = []
        if "expression" in missing_components:
            tips.append("Add disease-relevant tissue expression measurements to reduce uncertainty.")
        if "pathway" in missing_components:
            tips.append("Integrate curated pathway mappings and pathway-level importance metrics.")
        if "ppi" in missing_components:
            tips.append("Expand STRING interactions with higher-confidence experimental edges.")
        if "genetic" in missing_components:
            tips.append("Incorporate variant and biomarker evidence linked to disease risk.")
        if "ligandability" in missing_components:
            tips.append("Add confirmed ligand bioactivity or experimental docking/assay evidence.")
        if "trials" in missing_components:
            tips.append("Link target to trial records with phase and status metadata.")
        if not tips:
            low_components = sorted(
                [
                    (k, float(v.get("score", 0.0)))
                    for k, v in component_scores.items()
                    if v.get("available")
                ],
                key=lambda item: item[1],
            )
            for key, _ in low_components[:2]:
                if key == "ligandability":
                    tips.append("Increase ligand quality evidence and validate simulated docking experimentally.")
                elif key == "trials":
                    tips.append("Improve trial evidence by connecting the target to later-phase studies.")
                elif key == "genetic":
                    tips.append("Strengthen human-genetic support with high-confidence variant associations.")
        return tips[:3]

    def _build_rationale(
        self,
        result: Dict[str, Any],
        top_drivers: List[str],
        risk_flags: List[str],
        improvements: List[str],
    ) -> str:
        lines = [
            f"Composite actionability score is {result.get('composite_score', 0):.1f}/100.",
            f"Confidence is {result.get('confidence_score', 0):.1f}/100 with {result.get('data_completeness', 0):.1f}% data completeness.",
            "Strongest positive drivers: " + (", ".join(top_drivers) if top_drivers else "none detected."),
            "Main risks: " + (", ".join(risk_flags) if risk_flags else "no major risk flags detected."),
            "Missing components: " + (", ".join(result.get("missing_components", [])) if result.get("missing_components") else "none."),
            "Most impactful improvements: " + (", ".join(improvements) if improvements else "none identified."),
        ]
        return "\n".join(lines)

    def _missing_component(self, key: str, notes: List[str]) -> Dict[str, Any]:
        return {
            "available": False,
            "score": 0.0,
            "source_quality": 0.0,
            "notes": notes,
            "risk_flags": [f"Missing {key} evidence"],
        }

    @staticmethod
    def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _avg(values: List[float], default: float = 0.0) -> float:
        if not values:
            return default
        return sum(values) / len(values)

    @staticmethod
    def _std(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5

    @staticmethod
    def _clamp(value: float, min_value: float = 0.0, max_value: float = 100.0) -> float:
        return max(min_value, min(max_value, value))

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

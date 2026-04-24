"""Wet-lab handoff engine for converting computational findings into bench-ready plans."""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Tuple


class WetLabHandoffEngine:
    """Generate assay, CRISPR, and primer recommendations with uncertainty-aware outputs."""

    DISCLAIMER = (
        "Research-use only. Not for clinical diagnosis or treatment. "
        "All recommendations are uncertainty-labeled and require qualified human review."
    )

    def build_experiment_context(
        self,
        target_data: Dict[str, Any],
        sequence_data: Dict[str, Any],
        pathway_data: Dict[str, Any],
        variant_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build normalized context payload for wet-lab planning."""
        gene_name = str(
            target_data.get("gene_name")
            or target_data.get("target_id")
            or sequence_data.get("gene_name")
            or "UNKNOWN"
        ).strip()
        sequence = self._normalize_sequence(
            sequence_data.get("sequence")
            or sequence_data.get("fasta_sequence")
            or target_data.get("sequence")
            or ""
        )
        pathways = pathway_data.get("pathways", []) if isinstance(pathway_data, dict) else []
        variants = []
        if variant_data and isinstance(variant_data, dict):
            variants = variant_data.get("annotated") or variant_data.get("variants") or []

        missing_inputs: List[str] = []
        if not gene_name or gene_name == "UNKNOWN":
            missing_inputs.append("gene_name")
        if not sequence:
            missing_inputs.append("sequence")
        if not pathways:
            missing_inputs.append("pathway_data")

        return {
            "target": {
                "gene_name": gene_name,
                "uniprot_id": target_data.get("uniprot_id", ""),
                "protein_name": target_data.get("protein_name", ""),
                "target_profile": target_data,
            },
            "sequence": {
                "sequence": sequence,
                "length": len(sequence),
                "gc_content_pct": round(self._gc_content(sequence), 2) if sequence else 0.0,
            },
            "pathway": {
                "pathway_count": len(pathways),
                "top_pathways": [p.get("name", "Unknown") for p in pathways[:5] if isinstance(p, dict)],
                "raw": pathway_data,
            },
            "variant": {
                "variant_count": len(variants),
                "high_impact_variants": len(
                    [v for v in variants if str(v.get("predicted_effect_class", "")).lower() == "high"]
                ),
                "raw": variant_data or {},
            },
            "missing_inputs": missing_inputs,
            "disclaimer": self.DISCLAIMER,
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }

    def suggest_assays(self, context: Dict[str, Any], objective: str, lab_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank practical assay recommendations tailored to objective and available instruments."""
        objective_l = str(objective or "").strip().lower()
        instruments = {str(i).lower() for i in lab_profile.get("available_instruments", [])}
        throughput = str(lab_profile.get("throughput_preference", "medium")).lower()
        budget_tier = str(lab_profile.get("budget_tier", "medium")).lower()
        gene_name = context.get("target", {}).get("gene_name", "target")
        pathway_count = int(context.get("pathway", {}).get("pathway_count", 0))
        variant_count = int(context.get("variant", {}).get("variant_count", 0))

        templates = self._assay_templates()
        ranked: List[Dict[str, Any]] = []
        for row in templates:
            score = 50.0
            drivers: List[str] = []
            risks: List[str] = []

            if row["category"] == "expression":
                score += 12
                drivers.append("Objective generally requires expression-level confirmation.")
                if "qpcr" in instruments and row["assay_name"] == "qPCR":
                    score += 18
                    drivers.append("qPCR instrument availability enables rapid transcript validation.")
                if "western" in instruments and row["assay_name"] == "Western blot":
                    score += 15
                    drivers.append("Western capability supports orthogonal protein-level validation.")
                if "plate reader" in instruments and row["assay_name"] == "ELISA":
                    score += 10
                    drivers.append("Plate reader availability enables scalable ELISA readouts.")

            if row["category"] == "functional" and objective_l in {"target validation", "knockout", "pathway perturbation"}:
                score += 16
                drivers.append("Functional phenotype readouts fit perturbation-oriented objectives.")

            if row["category"] == "pathway" and ("pathway" in objective_l or pathway_count > 0):
                score += 16
                drivers.append("Detected pathway context supports pathway activity assays.")

            if row["category"] == "engagement" and ("ligand" in objective_l or "response" in objective_l):
                score += 14
                drivers.append("Objective includes ligand-response/engagement verification.")

            if row["category"] == "genotype" and (variant_count > 0 or "knockout" in objective_l):
                score += 18
                drivers.append("Genotype-level confirmation is needed for edit verification.")

            if throughput == "high" and row["high_throughput_friendly"]:
                score += 8
                drivers.append("Assay scales with high-throughput preference.")
            if budget_tier == "low" and row["cost_tier"] == "High":
                score -= 12
                risks.append("Cost pressure may limit practical execution.")
            if budget_tier == "high" and row["cost_tier"] in {"Medium", "High"}:
                score += 4

            if not drivers:
                drivers.append("Broadly useful orthogonal assay for mechanistic triangulation.")
            if row["special_equipment"] and row["special_equipment"] not in instruments:
                score -= 9
                risks.append(f"Requires instrument not confirmed in profile: {row['special_equipment']}.")

            confidence = self._confidence_label_from_score(score)
            ranked.append(
                {
                    "assay_name": row["assay_name"],
                    "purpose": row["purpose"],
                    "why_it_fits_this_target": f"{row['fit_note']} for {gene_name}.",
                    "required_materials": row["required_materials"],
                    "readout_type": row["readout_type"],
                    "positive_negative_controls": row["controls"],
                    "expected_signal_pattern": row["expected_signal_pattern"],
                    "turnaround_estimate": row["turnaround_estimate"],
                    "cost_tier": row["cost_tier"],
                    "risk_flags": risks[:3],
                    "confidence": confidence,
                    "references_evidence_tags": row["evidence_tags"],
                    "top_3_drivers": drivers[:3],
                    "top_3_risks_assumptions": (risks + ["Requires protocol optimization in local model system."])[:3],
                    "missing_inputs_reducing_uncertainty": self._top_missing_inputs(context)[:3],
                    "rank_score": round(self._clamp(score), 2),
                }
            )

        ranked.sort(key=lambda r: (-float(r["rank_score"]), r["assay_name"]))
        return ranked

    def suggest_crispr_targets(self, sequence: str, gene_name: str, genome_build: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate preliminary sgRNA candidates using transparent SpCas9-compatible heuristics."""
        seq = self._normalize_sequence(sequence)
        if len(seq) < 30:
            return []

        candidates: List[Dict[str, Any]] = []
        seq_len = len(seq)
        # Assumption: SpCas9 with NGG PAM.
        for i in range(20, seq_len - 3):
            pam = seq[i : i + 3]
            if len(pam) == 3 and pam[1:] == "GG":
                spacer = seq[i - 20 : i]
                if len(spacer) == 20:
                    candidates.append(
                        self._score_crispr_candidate(
                            spacer=spacer,
                            pam=pam,
                            target_position=i - 20,
                            strand="+",
                            seq=seq,
                            objective="KO",
                        )
                    )

        for i in range(3, seq_len - 20):
            pam_window = seq[i - 3 : i]
            if len(pam_window) == 3 and pam_window[:2] == "CC":
                protospacer = seq[i : i + 20]
                if len(protospacer) == 20:
                    spacer = self._reverse_complement(protospacer)
                    pam = self._reverse_complement(pam_window)
                    candidates.append(
                        self._score_crispr_candidate(
                            spacer=spacer,
                            pam=pam,
                            target_position=i,
                            strand="-",
                            seq=seq,
                            objective="KO",
                        )
                    )

        for row in candidates:
            row["gene_name"] = gene_name or "UNKNOWN"
            row["genome_build"] = genome_build or "unspecified"
            row["notes_for_manual_review"] = (
                row["notes_for_manual_review"]
                + " Preliminary heuristic only; full genome-wide off-target search not performed."
            )

        candidates.sort(key=lambda r: (-float(r["heuristic_score"]), r["target_position"]))
        return candidates[:25]

    def analyze_crispr_off_targets(
        self,
        guide_rna_sequence: str,
        patient_genome: Dict[str, Any],
        *,
        guide_label: str = "SpCas9-NGG",
    ) -> Dict[str, Any]:
        """Two-stage deterministic off-target analysis with explainable scoring and tiering."""
        guide = self._normalize_sequence(str(guide_rna_sequence or "").upper().replace("U", "T"))
        if len(guide) < 18:
            return {
                "guide_rna_sequence": guide,
                "guide_label": guide_label,
                "specificity_score_pct": 0.0,
                "overall_risk_label": "High",
                "ranked_off_targets": [],
                "top_3_drivers": ["Guide sequence is too short for robust off-target analysis."],
                "top_3_risks_assumptions": [
                    "Insufficient guide sequence length prevents candidate enumeration.",
                    "No patient-specific off-target ranking can be produced.",
                ],
                "missing_inputs_reducing_uncertainty": [
                    "Provide a 20nt guide RNA sequence.",
                    "Provide patient genome sequence with annotations.",
                ],
                "notes_for_manual_review": (
                    "Research-use only deterministic heuristic. "
                    "Sequence/model simplifications may under-represent context-dependent cleavage behavior."
                ),
                "summary_text": "Guide sequence too short for off-target analysis.",
                "disclaimer": self.DISCLAIMER,
            }

        genome_seq = self._normalize_sequence(patient_genome.get("sequence", ""))
        annotations = patient_genome.get("annotations", []) if isinstance(patient_genome, dict) else []
        variants = patient_genome.get("variants", []) if isinstance(patient_genome, dict) else []
        uncertainty_notes: List[str] = []
        if not genome_seq:
            uncertainty_notes.append("Patient genome sequence missing; no candidate off-target sites can be enumerated.")
        if not annotations:
            uncertainty_notes.append("Genome annotations missing; impact terms fall back to conservative defaults.")

        candidates = self._enumerate_off_target_candidates(guide=guide, genome_seq=genome_seq)
        ranked_off_targets: List[Dict[str, Any]] = []
        for row in candidates:
            impact = self._compute_off_target_impact(
                site_position=int(row["site_position"]),
                site_end=int(row["site_end"]),
                annotations=annotations,
                variants=variants,
            )
            cleavage_prob = self._estimate_off_target_cleavage_probability(
                mismatches=int(row["mismatches"]),
                bulges=int(row["bulges"]),
                pam=str(row["pam"]),
                seed_mismatches=int(row["seed_mismatches"]),
            )
            risk_score = self._clamp(cleavage_prob * impact["impact_weight"] * 100.0)
            ranked_off_targets.append(
                {
                    **row,
                    "cleavage_probability": round(cleavage_prob, 4),
                    "impact_weight": round(float(impact["impact_weight"]), 3),
                    "impact_flags": impact["impact_flags"],
                    "gene_name": impact["gene_name"],
                    "risk_score": round(risk_score, 2),
                    "tier_label": self._assign_off_target_tier(risk_score=risk_score, impact=impact),
                    "rank_explanation": self._build_off_target_explanation(row=row, impact=impact, cleavage_prob=cleavage_prob),
                }
            )

        ranked_off_targets.sort(key=lambda r: (-float(r["risk_score"]), r["gene_name"], int(r["site_position"])))
        ranked_off_targets = ranked_off_targets[:25]

        on_target_strength = self._estimate_on_target_strength(guide)
        aggregate_off_target_risk = sum(float(r["risk_score"]) / 100.0 for r in ranked_off_targets)
        specificity = (
            (on_target_strength / (on_target_strength + aggregate_off_target_risk)) * 100.0
            if (on_target_strength + aggregate_off_target_risk) > 0
            else 0.0
        )
        specificity = self._clamp(specificity)

        if specificity >= 85.0:
            overall_risk = "Low"
        elif specificity >= 65.0:
            overall_risk = "Medium"
        else:
            overall_risk = "High"

        top_rows = ranked_off_targets[:3]
        risky_sites = ", ".join(f"{r.get('gene_name', 'Intergenic')} ({r.get('tier_label', 'Tier 3')})" for r in top_rows)
        if not risky_sites:
            risky_sites = "No high-priority off-targets detected"
        summary_text = f"This gRNA has {round(specificity):.0f}% specificity. Risk off-targets: {risky_sites}"
        if uncertainty_notes:
            summary_text = f"{summary_text}. Uncertainty notes: {' '.join(uncertainty_notes)}"

        return {
            "guide_rna_sequence": guide,
            "guide_label": guide_label,
            "specificity_score_pct": round(specificity, 2),
            "on_target_strength": round(on_target_strength, 4),
            "aggregate_off_target_risk": round(aggregate_off_target_risk, 4),
            "overall_risk_label": overall_risk,
            "ranked_off_targets": ranked_off_targets,
            "top_3_drivers": [
                f"On-target strength estimate={on_target_strength:.3f}",
                f"Total enumerated PAM-adjacent candidates={len(candidates)}",
                "Impact model upweights coding/cancer/essential/regulatory/variant overlap loci.",
            ],
            "top_3_risks_assumptions": [
                "Deterministic heuristic scorer approximates cleavage probability.",
                "Bulges limited to small single-event edits in this implementation.",
                "Chromatin accessibility and cell-state context are not modeled.",
            ],
            "missing_inputs_reducing_uncertainty": (
                uncertainty_notes
                + [
                    "Add patient-specific chromatin context (ATAC/histone marks) when available.",
                    "Validate top-ranked sites with orthogonal wet-lab assays.",
                ]
            )[:3],
            "notes_for_manual_review": (
                "Research-use only deterministic model. "
                "Architecture is pluggable: candidate generation and scoring are separated for future ML replacement."
            ),
            "summary_text": summary_text,
            "disclaimer": self.DISCLAIMER,
        }

    def suggest_primers(self, sequence: str, regions: List[Dict[str, Any]], primer_constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest practical primer pairs from candidate regions with conservative heuristics."""
        seq = self._normalize_sequence(sequence)
        if not seq:
            return []

        mode = str(primer_constraints.get("intended_use", "qPCR"))
        min_len = int(primer_constraints.get("min_len", 18))
        max_len = int(primer_constraints.get("max_len", 24))
        min_gc = float(primer_constraints.get("min_gc", 40.0))
        max_gc = float(primer_constraints.get("max_gc", 60.0))
        min_tm = float(primer_constraints.get("min_tm", 57.0))
        max_tm = float(primer_constraints.get("max_tm", 64.0))
        amplicon_min = int(primer_constraints.get("amplicon_min", 80 if mode == "qPCR" else 180))
        amplicon_max = int(primer_constraints.get("amplicon_max", 200 if mode == "qPCR" else 650))

        outputs: List[Dict[str, Any]] = []
        for region in regions:
            center = int(region.get("target_position", 0))
            window_start = max(0, center - 250)
            window_end = min(len(seq), center + 250)

            forward_candidates = self._enumerate_primers(
                seq=seq,
                start=window_start,
                end=max(window_start, center),
                min_len=min_len,
                max_len=max_len,
                min_gc=min_gc,
                max_gc=max_gc,
                min_tm=min_tm,
                max_tm=max_tm,
                strand="+",
            )
            reverse_candidates = self._enumerate_primers(
                seq=seq,
                start=min(center, len(seq) - 1),
                end=window_end,
                min_len=min_len,
                max_len=max_len,
                min_gc=min_gc,
                max_gc=max_gc,
                min_tm=min_tm,
                max_tm=max_tm,
                strand="-",
            )

            for fwd in forward_candidates[:25]:
                for rev in reverse_candidates[:25]:
                    amplicon = int(rev["position_5p"]) - int(fwd["position_5p"]) + 1
                    if amplicon < amplicon_min or amplicon > amplicon_max:
                        continue
                    tm_diff = abs(float(fwd["tm_c"]) - float(rev["tm_c"]))
                    if tm_diff > 2.5:
                        continue
                    pair_score = self._clamp(
                        0.45 * float(fwd["quality"])
                        + 0.45 * float(rev["quality"])
                        + 0.10 * max(0.0, 100.0 - (tm_diff * 20.0))
                    )
                    caveats = []
                    if self._contains_repetitive_run(fwd["sequence"], threshold=5) or self._contains_repetitive_run(rev["sequence"], threshold=5):
                        caveats.append("Repetitive run detected; validate specificity in silico/in wet-lab.")
                    caveats.append("Heuristic-only Tm/secondary structure checks; full thermodynamic modeling not applied.")
                    outputs.append(
                        {
                            "forward_sequence": fwd["sequence"],
                            "reverse_sequence": rev["sequence"],
                            "forward_tm_c": round(float(fwd["tm_c"]), 2),
                            "reverse_tm_c": round(float(rev["tm_c"]), 2),
                            "forward_gc_pct": round(float(fwd["gc_pct"]), 2),
                            "reverse_gc_pct": round(float(rev["gc_pct"]), 2),
                            "forward_length": int(fwd["length"]),
                            "reverse_length": int(rev["length"]),
                            "expected_amplicon_size": int(amplicon),
                            "intended_use": mode,
                            "quality_score": round(pair_score, 2),
                            "caveats": caveats[:3],
                            "region_reference": {
                                "target_position": int(region.get("target_position", center)),
                                "objective_fit": region.get("objective_fit", "KO"),
                            },
                        }
                    )

        outputs.sort(key=lambda r: (-float(r["quality_score"]), abs(int(r["expected_amplicon_size"]) - ((amplicon_min + amplicon_max) // 2))))
        return outputs[:30]

    def generate_validation_checklist(self, plan: Dict[str, Any], objective: str) -> List[Dict[str, Any]]:
        """Generate execution checklist with severity and ownership metadata."""
        objective_text = str(objective or "target validation")
        sections = {
            "pre_experiment_qc": [
                ("Verify cell model identity and mycoplasma-negative status.", "High", "Research associate"),
                ("Confirm target sequence/reference build consistency across all designs.", "High", "Bioinformatics scientist"),
                ("Document objective-specific success criteria before experiment start.", "Medium", "Project lead"),
            ],
            "reagent_and_control_checklist": [
                ("Validate lot IDs and storage conditions for critical reagents.", "High", "Lab manager"),
                ("Include positive and negative controls for each assay panel.", "High", "Research associate"),
                ("Prepare transfection/editing efficiency control readouts.", "Medium", "Research associate"),
            ],
            "execution_checkpoints": [
                ("Record transfection/transduction conditions and timing.", "Medium", "Research associate"),
                ("Capture protocol deviations in real-time run log.", "Medium", "Research associate"),
                ("Gate go/no-go decision after first orthogonal readout.", "High", "Project lead"),
            ],
            "data_quality_acceptance_criteria": [
                ("Ensure technical replicates have CV within predeclared threshold.", "High", "Data analyst"),
                ("Require control behavior to match expected signal direction.", "High", "Data analyst"),
                ("Flag any assay with inconsistent control outcomes for repeat.", "High", "Project lead"),
            ],
            "troubleshooting_triggers": [
                ("Trigger troubleshooting if controls fail in >1 assay type.", "High", "Project lead"),
                ("Trigger troubleshooting if CRISPR editing proxy < expected minimum.", "Medium", "Research associate"),
                ("Trigger troubleshooting for unexpected toxicity unrelated to objective.", "Medium", "Research associate"),
            ],
            "replication_and_stats_minimums": [
                ("Use at least 3 biological replicates for primary conclusions.", "High", "Project lead"),
                ("Predefine statistical test and exclusion criteria.", "High", "Data analyst"),
                ("Archive raw data and analysis scripts for reproducibility.", "Medium", "Data analyst"),
            ],
            "documentation_requirements": [
                ("Attach objective statement and design rationale to runbook.", "Medium", "Project lead"),
                ("Link assay/CRISPR/primer versions used in execution.", "High", "Bioinformatics scientist"),
                ("Complete final sign-off with reviewer comments.", "Medium", "Project lead"),
            ],
        }

        checklist: List[Dict[str, Any]] = []
        for section, items in sections.items():
            for item_text, severity, owner in items:
                checklist.append(
                    {
                        "section": section,
                        "item_text": item_text.replace("objective", objective_text),
                        "severity_if_skipped": severity,
                        "owner_role_suggestion": owner,
                        "completion_status": "unchecked",
                    }
                )
        if not plan.get("crispr_candidates"):
            checklist.append(
                {
                    "section": "pre_experiment_qc",
                    "item_text": "No CRISPR candidates generated; perform manual design review before proceeding.",
                    "severity_if_skipped": "High",
                    "owner_role_suggestion": "Bioinformatics scientist",
                    "completion_status": "unchecked",
                }
            )
        return checklist

    def compute_plan_confidence(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Compute confidence, completeness, and readiness status for the generated plan."""
        assays = plan.get("assays", [])
        crispr = plan.get("crispr_candidates", [])
        primers = plan.get("primers", [])
        context = plan.get("context", {})

        completeness_components = [
            1.0 if context.get("target", {}).get("gene_name") not in {"", "UNKNOWN"} else 0.0,
            1.0 if context.get("sequence", {}).get("sequence") else 0.0,
            1.0 if context.get("pathway", {}).get("pathway_count", 0) > 0 else 0.0,
            1.0 if assays else 0.0,
            1.0 if crispr else 0.0,
            1.0 if primers else 0.0,
        ]
        data_completeness = (sum(completeness_components) / len(completeness_components)) * 100.0

        assay_conf = self._avg([self._confidence_to_numeric(a.get("confidence", "Low")) for a in assays], default=0.45)
        crispr_conf = self._avg([float(c.get("heuristic_score", 0.0)) / 100.0 for c in crispr], default=0.4)
        primer_conf = self._avg([float(p.get("quality_score", 0.0)) / 100.0 for p in primers], default=0.4)

        risk_count = sum(
            len(a.get("risk_flags", []))
            for a in assays[:5]
        ) + sum(1 for c in crispr[:10] if c.get("off_target_risk_level") == "High")
        risk_penalty = min(25.0, risk_count * 2.0)

        score = self._clamp((0.45 * data_completeness) + 20.0 * assay_conf + 20.0 * crispr_conf + 15.0 * primer_conf - risk_penalty)
        if score >= 75:
            readiness = "green"
        elif score >= 50:
            readiness = "yellow"
        else:
            readiness = "red"

        top_risks = self._collect_top_risks(plan, limit=3)
        missing = context.get("missing_inputs", []) or ["Add replicate pilot data to tighten uncertainty."]

        return {
            "plan_confidence_score": round(score, 2),
            "data_completeness": round(self._clamp(data_completeness), 2),
            "readiness_label": readiness,
            "top_3_drivers": self._collect_top_drivers(plan, limit=3),
            "top_3_risks_assumptions": top_risks,
            "missing_inputs_reducing_uncertainty": missing[:3],
        }

    def export_wet_lab_package(self, plan: Dict[str, Any], format: str = "json") -> Dict[str, Any]:
        """Export wet-lab handoff package in JSON/CSV/Markdown/TXT-friendly structures."""
        confidence = plan.get("confidence") or self.compute_plan_confidence(plan)
        plan = {**plan, "confidence": confidence, "disclaimer": self.DISCLAIMER}

        assays_csv = self._to_csv(plan.get("assays", []))
        crispr_csv = self._to_csv(plan.get("crispr_candidates", []))
        primers_csv = self._to_csv(plan.get("primers", []))
        checklist_csv = self._to_csv(plan.get("validation_checklist", []))
        crispr_off_targets_csv = self._to_csv(plan.get("crispr_off_target_analysis", {}).get("ranked_off_targets", []))
        markdown_brief = self._build_markdown_brief(plan)
        txt_brief = markdown_brief.replace("## ", "").replace("### ", "")

        bundle = {
            "json": plan,
            "csv": {
                "assays": assays_csv,
                "crispr_candidates": crispr_csv,
                "primers": primers_csv,
                "validation_checklist": checklist_csv,
                "crispr_off_targets": crispr_off_targets_csv,
            },
            "markdown_brief": markdown_brief,
            "txt_brief": txt_brief,
            "disclaimer": self.DISCLAIMER,
            "export_generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        return {"format": format.lower(), "content": bundle}

    def _assay_templates(self) -> List[Dict[str, Any]]:
        return [
            {
                "assay_name": "qPCR",
                "category": "expression",
                "purpose": "Transcript-level expression validation after perturbation.",
                "fit_note": "Fast and sensitive quantification of target mRNA change",
                "required_materials": ["RNA extraction kit", "Reverse transcription kit", "qPCR master mix", "Validated primers"],
                "readout_type": "Ct/Cq-based fold-change",
                "controls": ["Housekeeping gene control", "No-template control", "Non-targeting control"],
                "expected_signal_pattern": "KO/CRISPRi: increased Ct for target; overexpression: decreased Ct.",
                "turnaround_estimate": "1-2 days",
                "cost_tier": "Low",
                "risk_flags": [],
                "evidence_tags": ["Internal:sequence_data", "Method:qPCR"],
                "special_equipment": "qpcr",
                "high_throughput_friendly": True,
            },
            {
                "assay_name": "Western blot",
                "category": "expression",
                "purpose": "Protein-level validation of target modulation.",
                "fit_note": "Orthogonal protein-level confirmation of transcript observations",
                "required_materials": ["Lysis buffer", "SDS-PAGE reagents", "Primary/secondary antibodies", "Chemiluminescence substrate"],
                "readout_type": "Band intensity ratio",
                "controls": ["Loading control (e.g., ACTB)", "Untreated control", "Positive control lysate"],
                "expected_signal_pattern": "Target band decreases with KO/CRISPRi and increases with overexpression.",
                "turnaround_estimate": "2-3 days",
                "cost_tier": "Medium",
                "risk_flags": [],
                "evidence_tags": ["Internal:target_profile", "Method:western_blot"],
                "special_equipment": "western",
                "high_throughput_friendly": False,
            },
            {
                "assay_name": "ELISA",
                "category": "expression",
                "purpose": "Secreted or soluble protein quantification.",
                "fit_note": "Useful when target or downstream biomarker is measurable in supernatant",
                "required_materials": ["ELISA kit", "Plate reader", "Standards", "Wash buffers"],
                "readout_type": "Absorbance-derived concentration",
                "controls": ["Standard curve", "Blank wells", "Positive matrix control"],
                "expected_signal_pattern": "Concentration shifts consistent with perturbation direction.",
                "turnaround_estimate": "1 day",
                "cost_tier": "Medium",
                "risk_flags": [],
                "evidence_tags": ["Internal:pathway_data", "Method:ELISA"],
                "special_equipment": "plate reader",
                "high_throughput_friendly": True,
            },
            {
                "assay_name": "Cell viability/proliferation assay",
                "category": "functional",
                "purpose": "Functional response to gene perturbation or ligand treatment.",
                "fit_note": "Links molecular changes to cell-level phenotype",
                "required_materials": ["Viability reagent (MTT/CellTiter-Glo)", "Cells", "Plate reader"],
                "readout_type": "Luminescence/absorbance viability index",
                "controls": ["Vehicle control", "Known cytotoxic control", "Non-targeting edit control"],
                "expected_signal_pattern": "Objective-dependent viability shift vs matched controls.",
                "turnaround_estimate": "2-4 days",
                "cost_tier": "Low",
                "risk_flags": [],
                "evidence_tags": ["Internal:target_profile", "Method:viability_assay"],
                "special_equipment": "plate reader",
                "high_throughput_friendly": True,
            },
            {
                "assay_name": "Apoptosis / migration panel",
                "category": "functional",
                "purpose": "Downstream functional phenotype profiling.",
                "fit_note": "Captures mechanism-relevant phenotypes beyond expression",
                "required_materials": ["Annexin V kit or migration chamber", "Flow/microscope", "Controls"],
                "readout_type": "Flow cytometry or imaging metric",
                "controls": ["Untreated baseline", "Positive apoptosis/migration control", "Rescue control if available"],
                "expected_signal_pattern": "Perturbation-specific directional shift with control concordance.",
                "turnaround_estimate": "3-5 days",
                "cost_tier": "Medium",
                "risk_flags": [],
                "evidence_tags": ["Internal:pathway_data", "Method:functional_panel"],
                "special_equipment": "flow",
                "high_throughput_friendly": False,
            },
            {
                "assay_name": "Pathway reporter assay",
                "category": "pathway",
                "purpose": "Quantify pathway activity perturbation.",
                "fit_note": "Directly measures pathway-level consequence of target modulation",
                "required_materials": ["Reporter construct", "Transfection reagent", "Plate reader"],
                "readout_type": "Luciferase/fluorescence reporter intensity",
                "controls": ["Reporter-only control", "Pathway agonist/antagonist control", "Non-targeting perturbation control"],
                "expected_signal_pattern": "Reporter signal shifts concordant with expected pathway direction.",
                "turnaround_estimate": "2-4 days",
                "cost_tier": "Medium",
                "risk_flags": [],
                "evidence_tags": ["Internal:pathway_data", "Method:reporter_assay"],
                "special_equipment": "plate reader",
                "high_throughput_friendly": True,
            },
            {
                "assay_name": "Phospho-readout assay",
                "category": "pathway",
                "purpose": "Assess pathway activation state via phospho-markers.",
                "fit_note": "Supports signaling cascade verification with mechanistic specificity",
                "required_materials": ["Phospho-specific antibodies", "Western/flow reagents", "Lysis buffer"],
                "readout_type": "Phospho/total ratio",
                "controls": ["Stimulus control", "Inhibitor control", "Time-zero baseline"],
                "expected_signal_pattern": "Phospho marker modulation aligns with target/pathway hypothesis.",
                "turnaround_estimate": "2-3 days",
                "cost_tier": "Medium",
                "risk_flags": [],
                "evidence_tags": ["Internal:pathway_data", "Method:phospho_readout"],
                "special_equipment": "western",
                "high_throughput_friendly": False,
            },
            {
                "assay_name": "Target engagement proxy assay",
                "category": "engagement",
                "purpose": "Estimate target engagement after ligand or perturbation.",
                "fit_note": "Adds proximal evidence that intervention reaches intended target axis",
                "required_materials": ["Engagement probe or proxy biomarker panel", "Detection platform"],
                "readout_type": "Binding/proxy index",
                "controls": ["No-ligand control", "Competitor control", "Dose-response controls"],
                "expected_signal_pattern": "Dose-aligned engagement proxy increase/decrease.",
                "turnaround_estimate": "3-6 days",
                "cost_tier": "High",
                "risk_flags": [],
                "evidence_tags": ["Internal:ligand_binding_data", "Method:engagement_proxy"],
                "special_equipment": "microscopy",
                "high_throughput_friendly": False,
            },
            {
                "assay_name": "Genotype validation (amplicon sequencing)",
                "category": "genotype",
                "purpose": "Confirm edit outcomes and indel spectrum.",
                "fit_note": "Critical for confirming intended genomic perturbation",
                "required_materials": ["PCR reagents", "Primer pairs", "NGS or Sanger workflow"],
                "readout_type": "Indel/edit frequency",
                "controls": ["Unedited baseline DNA", "No-template control", "Known-edit positive control"],
                "expected_signal_pattern": "On-target edit enrichment at expected locus.",
                "turnaround_estimate": "4-7 days",
                "cost_tier": "High",
                "risk_flags": [],
                "evidence_tags": ["Internal:sequence_data", "Method:genotype_validation"],
                "special_equipment": "ngs",
                "high_throughput_friendly": True,
            },
        ]

    def _score_crispr_candidate(
        self,
        spacer: str,
        pam: str,
        target_position: int,
        strand: str,
        seq: str,
        objective: str,
    ) -> Dict[str, Any]:
        gc = self._gc_content(spacer)
        gc_score = max(0.0, 100.0 - abs(gc - 50.0) * 2.5)
        homopolymer_penalty = 25.0 if self._contains_repetitive_run(spacer, threshold=4) else 0.0

        seed = spacer[8:20]
        seed_hits = self._count_substring(seq, seed)
        uniqueness_score = 100.0 if seed_hits <= 1 else max(25.0, 100.0 - (seed_hits - 1) * 18.0)

        relative_pos = target_position / max(1, len(seq))
        context_score = 100.0 * (1.0 - min(1.0, relative_pos / 0.55)) if objective.upper() == "KO" else 70.0
        heuristic = self._clamp(0.35 * gc_score + 0.35 * uniqueness_score + 0.30 * context_score - homopolymer_penalty)

        if seed_hits <= 1 and heuristic >= 75:
            off_target_risk = "Low"
        elif seed_hits <= 2 and heuristic >= 55:
            off_target_risk = "Medium"
        else:
            off_target_risk = "High"

        return {
            "spacer_sequence": spacer,
            "pam": pam,
            "target_position": int(target_position),
            "strand": strand,
            "objective_fit": "KO/KI/CRISPRi/a preliminary",
            "heuristic_score": round(heuristic, 2),
            "off_target_risk_level": off_target_risk,
            "gc_content_pct": round(gc, 2),
            "notes_for_manual_review": (
                "Uses PAM scan + GC + homopolymer + local seed uniqueness proxy."
            ),
            "top_3_drivers": [
                f"GC suitability at {gc:.1f}%",
                f"Seed uniqueness proxy count={seed_hits}",
                f"Region placement preference score={context_score:.1f}",
            ],
            "top_3_risks_assumptions": [
                "No full-genome off-target alignments run.",
                "Chromatin/accessibility context not modeled.",
                "Edit outcome distribution may vary by cell model.",
            ],
            "missing_inputs_reducing_uncertainty": [
                "Genome-wide off-target search results",
                "Exon/intron annotations for precise region targeting",
                "Cell-model editing efficiency baseline",
            ],
        }

    def _enumerate_off_target_candidates(self, guide: str, genome_seq: str) -> List[Dict[str, Any]]:
        """Stage 1: PAM-aware candidate enumeration with mismatch and small-bulge support."""
        if not genome_seq or len(guide) < 18:
            return []
        g = guide[:20]
        results: List[Dict[str, Any]] = []
        genome_len = len(genome_seq)

        for pam_idx in range(20, genome_len - 2):
            pam = genome_seq[pam_idx : pam_idx + 3]
            if len(pam) == 3 and pam[1:] == "GG":
                protospacer = genome_seq[pam_idx - 20 : pam_idx]
                alignment_label, mismatches, bulges = self._align_guide_with_small_bulges(g, protospacer)
                if not alignment_label:
                    continue
                seed_mismatches = self._count_mismatches(g[8:20], protospacer[8:20])
                results.append(
                    {
                        "site_position": int(pam_idx - 20),
                        "site_end": int(pam_idx + 2),
                        "strand": "+",
                        "pam": pam,
                        "candidate_sequence": protospacer,
                        "mismatches": int(mismatches),
                        "bulges": int(bulges),
                        "seed_mismatches": int(seed_mismatches),
                        "alignment_label": alignment_label,
                    }
                )

        for pam_idx in range(3, genome_len - 20):
            pam_window = genome_seq[pam_idx - 3 : pam_idx]
            if len(pam_window) == 3 and pam_window[:2] == "CC":
                protospacer_fw = genome_seq[pam_idx : pam_idx + 20]
                protospacer = self._reverse_complement(protospacer_fw)
                pam = self._reverse_complement(pam_window)
                alignment_label, mismatches, bulges = self._align_guide_with_small_bulges(g, protospacer)
                if not alignment_label:
                    continue
                seed_mismatches = self._count_mismatches(g[8:20], protospacer[8:20])
                results.append(
                    {
                        "site_position": int(pam_idx - 3),
                        "site_end": int(pam_idx + 19),
                        "strand": "-",
                        "pam": pam,
                        "candidate_sequence": protospacer,
                        "mismatches": int(mismatches),
                        "bulges": int(bulges),
                        "seed_mismatches": int(seed_mismatches),
                        "alignment_label": alignment_label,
                    }
                )

        results.sort(key=lambda r: (int(r["mismatches"]), int(r["bulges"]), int(r["seed_mismatches"]), int(r["site_position"])))
        return results[:300]

    def _align_guide_with_small_bulges(self, guide: str, target: str) -> Tuple[str, int, int]:
        mismatch_direct = self._count_mismatches(guide, target)
        best = ("match", mismatch_direct, 0)
        if mismatch_direct <= 4:
            return best

        for idx in range(len(guide)):
            trimmed_guide = guide[:idx] + guide[idx + 1 :]
            mismatch = self._count_mismatches(trimmed_guide, target[: len(trimmed_guide)])
            if mismatch <= 4 and (1 + mismatch) < (best[1] + best[2]):
                best = ("guide_bulge", mismatch, 1)

        for idx in range(len(target)):
            trimmed_target = target[:idx] + target[idx + 1 :]
            mismatch = self._count_mismatches(guide[: len(trimmed_target)], trimmed_target)
            if mismatch <= 4 and (1 + mismatch) < (best[1] + best[2]):
                best = ("target_bulge", mismatch, 1)

        if best[1] <= 4 and best[2] <= 1:
            return best
        return ("", 99, 99)

    def _estimate_off_target_cleavage_probability(self, mismatches: int, bulges: int, pam: str, seed_mismatches: int) -> float:
        score = 0.88 - (0.12 * mismatches) - (0.16 * bulges) - (0.07 * seed_mismatches)
        if pam == "AGG":
            score += 0.03
        elif pam == "TGG":
            score += 0.01
        return self._clamp(score * 100.0) / 100.0

    def _compute_off_target_impact(
        self,
        site_position: int,
        site_end: int,
        annotations: List[Dict[str, Any]],
        variants: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        impact_weight = 1.0
        impact_flags: List[str] = []
        gene_name = "Intergenic"

        matched = None
        for ann in annotations:
            start = int(ann.get("start", -1))
            end = int(ann.get("end", -1))
            if start <= site_end and site_position <= end:
                matched = ann
                break

        if matched:
            gene_name = str(matched.get("gene_name") or matched.get("gene") or "Intergenic")
            if bool(matched.get("is_coding")):
                impact_weight += 0.30
                impact_flags.append("coding_region")
            if bool(matched.get("is_cancer_gene")):
                impact_weight += 0.40
                impact_flags.append("cancer_gene")
            if bool(matched.get("is_essential_gene")):
                impact_weight += 0.35
                impact_flags.append("essential_gene")
            if bool(matched.get("is_regulatory_hotspot")):
                impact_weight += 0.25
                impact_flags.append("regulatory_hotspot")
        else:
            impact_flags.append("annotation_missing_for_site")

        for variant in variants:
            pos = variant.get("position")
            if pos is None:
                continue
            if site_position <= int(pos) <= site_end:
                if bool(variant.get("pathogenic")) or str(variant.get("impact", "")).lower() in {"high", "pathogenic"}:
                    impact_weight += 0.45
                    impact_flags.append("patient_variant_overlap")
                    break

        return {
            "impact_weight": impact_weight,
            "impact_flags": sorted(set(impact_flags)),
            "gene_name": gene_name,
        }

    def _assign_off_target_tier(self, risk_score: float, impact: Dict[str, Any]) -> str:
        high_impact = bool({"cancer_gene", "essential_gene", "patient_variant_overlap"} & set(impact.get("impact_flags", [])))
        if risk_score >= 70.0 or (risk_score >= 50.0 and high_impact):
            return "Tier 1"
        if risk_score >= 35.0 or high_impact:
            return "Tier 2"
        return "Tier 3"

    def _estimate_on_target_strength(self, guide: str) -> float:
        gc = self._gc_content(guide)
        gc_term = max(0.2, 1.0 - (abs(gc - 50.0) / 60.0))
        homopolymer_penalty = 0.2 if self._contains_repetitive_run(guide, threshold=4) else 0.0
        return max(0.1, gc_term - homopolymer_penalty)

    def _build_off_target_explanation(self, row: Dict[str, Any], impact: Dict[str, Any], cleavage_prob: float) -> str:
        impact_flags = impact.get("impact_flags", [])
        impact_desc = " / ".join(impact_flags) if impact_flags else "baseline genomic context"
        return (
            f"Ranked high due to cleavage={cleavage_prob:.2f}, mismatches={row.get('mismatches')}, "
            f"bulges={row.get('bulges')}, and impact={impact_desc}."
        )

    def _enumerate_primers(
        self,
        seq: str,
        start: int,
        end: int,
        min_len: int,
        max_len: int,
        min_gc: float,
        max_gc: float,
        min_tm: float,
        max_tm: float,
        strand: str,
    ) -> List[Dict[str, Any]]:
        primers: List[Dict[str, Any]] = []
        if end <= start:
            return primers

        for pos in range(start, end):
            for length in range(min_len, max_len + 1):
                if strand == "+":
                    if pos + length > len(seq):
                        continue
                    oligo = seq[pos : pos + length]
                    pos_5p = pos
                    pos_3p = pos + length - 1
                else:
                    if pos - length < 0:
                        continue
                    oligo = self._reverse_complement(seq[pos - length + 1 : pos + 1])
                    pos_5p = pos
                    pos_3p = pos - length + 1

                gc_pct = self._gc_content(oligo)
                if gc_pct < min_gc or gc_pct > max_gc:
                    continue
                tm_c = self._wallace_tm(oligo)
                if tm_c < min_tm or tm_c > max_tm:
                    continue
                if self._contains_repetitive_run(oligo, threshold=4):
                    continue

                quality = self._clamp(
                    100.0
                    - abs(gc_pct - 50.0) * 1.7
                    - abs(tm_c - ((min_tm + max_tm) / 2.0)) * 4.0
                )
                primers.append(
                    {
                        "sequence": oligo,
                        "gc_pct": round(gc_pct, 2),
                        "tm_c": round(tm_c, 2),
                        "length": int(length),
                        "position_5p": int(pos_5p),
                        "position_3p": int(pos_3p),
                        "quality": round(quality, 2),
                    }
                )

        primers.sort(key=lambda r: (-float(r["quality"]), int(r["length"])))
        return primers

    def _top_missing_inputs(self, context: Dict[str, Any]) -> List[str]:
        missing = list(context.get("missing_inputs", []))
        if not missing:
            missing = ["Genome-wide off-target validation", "Pilot replicate variability", "Confirmed protein-level baseline"]
        return missing

    def _collect_top_drivers(self, plan: Dict[str, Any], limit: int = 3) -> List[str]:
        drivers: List[str] = []
        if plan.get("assays"):
            drivers.extend(plan["assays"][0].get("top_3_drivers", []))
        if plan.get("crispr_candidates"):
            drivers.extend(plan["crispr_candidates"][0].get("top_3_drivers", []))
        if plan.get("primers"):
            drivers.append(f"Primer pair quality top score={plan['primers'][0].get('quality_score')}")
        dedup = []
        for d in drivers:
            if d not in dedup:
                dedup.append(d)
        return dedup[:limit]

    def _collect_top_risks(self, plan: Dict[str, Any], limit: int = 3) -> List[str]:
        risks: List[str] = []
        if plan.get("assays"):
            risks.extend(plan["assays"][0].get("top_3_risks_assumptions", []))
        if plan.get("crispr_candidates"):
            risks.extend(plan["crispr_candidates"][0].get("top_3_risks_assumptions", []))
        if plan.get("primers"):
            risks.extend(plan["primers"][0].get("caveats", []))
        dedup = []
        for r in risks:
            if r not in dedup:
                dedup.append(r)
        return dedup[:limit]

    def _build_markdown_brief(self, plan: Dict[str, Any]) -> str:
        objective = plan.get("objective", "unspecified")
        confidence = plan.get("confidence", {})
        assays = plan.get("assays", [])[:5]
        crispr = plan.get("crispr_candidates", [])[:8]
        primers = plan.get("primers", [])[:8]
        checklist = plan.get("validation_checklist", [])
        crispr_off_target = plan.get("crispr_off_target_analysis", {}) or {}

        lines = [
            "# OmniBiMol Wet-Lab Handoff Brief",
            "",
            f"**Objective:** {objective}",
            f"**Readiness:** {confidence.get('readiness_label', 'unknown')} ({confidence.get('plan_confidence_score', 0)}/100)",
            f"**Data completeness:** {confidence.get('data_completeness', 0)}%",
            "",
            "## Safety and Review Requirements",
            f"- {self.DISCLAIMER}",
            "- Experimental recommendations are preliminary and must be reviewed by qualified bench scientists.",
            "",
            "## Ranked Plan",
        ]
        for idx, assay in enumerate(assays, start=1):
            lines.append(
                f"{idx}. **{assay.get('assay_name')}** | score {assay.get('rank_score')} | confidence {assay.get('confidence')}"
            )
            lines.append(f"   - Purpose: {assay.get('purpose')}")
            lines.append(f"   - Controls: {', '.join(assay.get('positive_negative_controls', []))}")

        lines.extend(["", "## CRISPR Candidate Highlights"])
        for row in crispr[:5]:
            lines.append(
                f"- Pos {row.get('target_position')} ({row.get('strand')}): {row.get('spacer_sequence')} | PAM {row.get('pam')} | score {row.get('heuristic_score')} | off-target risk {row.get('off_target_risk_level')}"
            )

        if crispr_off_target:
            lines.extend(["", "## CRISPR Off-Target Risk"])
            lines.append(f"- {crispr_off_target.get('summary_text', 'Off-target summary unavailable.')}")
            for row in crispr_off_target.get("ranked_off_targets", [])[:5]:
                lines.append(
                    "- "
                    f"{row.get('gene_name', 'Intergenic')} | tier {row.get('tier_label')} | "
                    f"risk={row.get('risk_score')} | cleavage={row.get('cleavage_probability')} | "
                    f"mm={row.get('mismatches')} | bulges={row.get('bulges')}"
                )

        lines.extend(["", "## Primer Pair Highlights"])
        for row in primers[:5]:
            lines.append(
                f"- Amplicon {row.get('expected_amplicon_size')} bp | Q={row.get('quality_score')} | F={row.get('forward_sequence')} | R={row.get('reverse_sequence')}"
            )

        lines.extend(["", "## Key Risks", *[f"- {r}" for r in confidence.get("top_3_risks_assumptions", [])]])
        lines.extend(["", "## Stepwise Runbook"])
        lines.extend(
            [
                "1. Confirm pre-experiment QC and reagent/control readiness.",
                "2. Execute primary expression and functional assays with controls.",
                "3. Perform genotype and CRISPR validation checks.",
                "4. Review data acceptance criteria and trigger troubleshooting if needed.",
                "5. Record outcomes and sign off with reviewer.",
            ]
        )
        lines.extend(["", "## Sign-off", "- Executor:", "- Reviewer:", "- Date:", "- Notes:"])
        lines.extend(["", "## Checklist Items"])
        for item in checklist[:12]:
            lines.append(f"- [{item.get('completion_status')}] ({item.get('severity_if_skipped')}) {item.get('item_text')}")
        return "\n".join(lines)

    def _to_csv(self, rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return ""
        flattened_rows = [self._flatten_dict(row) for row in rows]
        fields = sorted({k for row in flattened_rows for k in row.keys()})
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        for row in flattened_rows:
            writer.writerow(row)
        return buffer.getvalue()

    def _flatten_dict(self, row: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        for key, value in row.items():
            field = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            if isinstance(value, dict):
                flat.update(self._flatten_dict(value, field))
            elif isinstance(value, list):
                flat[field] = json.dumps(value)
            else:
                flat[field] = value
        return flat

    def _normalize_sequence(self, sequence: str) -> str:
        return "".join(ch for ch in str(sequence or "").upper() if ch in {"A", "C", "G", "T"})

    def _gc_content(self, sequence: str) -> float:
        seq = self._normalize_sequence(sequence)
        if not seq:
            return 0.0
        gc = sum(1 for ch in seq if ch in {"G", "C"})
        return (gc / len(seq)) * 100.0

    def _wallace_tm(self, sequence: str) -> float:
        seq = self._normalize_sequence(sequence)
        at = sum(1 for ch in seq if ch in {"A", "T"})
        gc = sum(1 for ch in seq if ch in {"G", "C"})
        return float((2 * at) + (4 * gc))

    def _contains_repetitive_run(self, sequence: str, threshold: int = 4) -> bool:
        if not sequence:
            return False
        run = 1
        prev = sequence[0]
        for ch in sequence[1:]:
            if ch == prev:
                run += 1
                if run >= threshold:
                    return True
            else:
                prev = ch
                run = 1
        return False

    def _count_substring(self, sequence: str, motif: str) -> int:
        if not sequence or not motif:
            return 0
        count = 0
        m = len(motif)
        for idx in range(0, max(0, len(sequence) - m + 1)):
            if sequence[idx : idx + m] == motif:
                count += 1
        return count

    def _count_mismatches(self, a: str, b: str) -> int:
        n = min(len(a), len(b))
        mismatch = sum(1 for i in range(n) if a[i] != b[i])
        mismatch += abs(len(a) - len(b))
        return mismatch

    def _reverse_complement(self, sequence: str) -> str:
        table = str.maketrans("ACGT", "TGCA")
        return self._normalize_sequence(sequence).translate(table)[::-1]

    def _confidence_label_from_score(self, score: float) -> str:
        value = float(score)
        if value >= 75.0:
            return "High"
        if value >= 55.0:
            return "Med"
        return "Low"

    def _confidence_to_numeric(self, value: str) -> float:
        mapping = {"high": 0.9, "med": 0.7, "medium": 0.7, "low": 0.45}
        return mapping.get(str(value).lower(), 0.5)

    def _avg(self, values: List[float], default: float = 0.0) -> float:
        if not values:
            return float(default)
        return float(sum(values) / len(values))

    def _clamp(self, value: float, lo: float = 0.0, hi: float = 100.0) -> float:
        return max(lo, min(hi, float(value)))

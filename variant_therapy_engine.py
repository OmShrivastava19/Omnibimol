"""Variant-to-therapy engine for transparent precision-medicine research workflows."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import math
import os
import re
import warnings
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from api_client import get_drug_metadata, get_manual_drug_database
try:
    from variant_prioritizer import VariantPrioritizer
    VARIANT_PRIORITIZER_AVAILABLE = True
except ImportError:
    VARIANT_PRIORITIZER_AVAILABLE = False
    VariantPrioritizer = None


class VariantTherapyEngine:
    """End-to-end deterministic variant-to-therapy scoring pipeline."""

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "target_gene_match": 0.30,
        "pathway_correction_relevance": 0.20,
        "evidence_quality": 0.20,
        "clinical_maturity": 0.15,
        "safety_risk_penalty": 0.15,
    }

    DISCLAIMER = "For research use only. Not for clinical diagnosis/treatment decisions."

    def __init__(
        self,
        api_client: Optional[Any] = None,
        cache_manager: Optional[Any] = None,
        prioritizer: Optional[Any] = None,
        prioritizer_cache_path: Optional[str] = None,
        enable_prioritizer_download: Optional[bool] = None,
    ) -> None:
        self.api_client = api_client
        self.cache = cache_manager
        
        # Initialize variant prioritizer (research-backed scoring)
        self._prioritizer = None
        if prioritizer is not None:
            self._prioritizer = prioritizer
        elif VARIANT_PRIORITIZER_AVAILABLE:
            try:
                cache_path = prioritizer_cache_path or os.getenv(
                    "OMNIBIMOL_VARIANT_PRIORITY_CACHE",
                    "./cache/hf_artifacts",
                )
                if enable_prioritizer_download is None:
                    enable_prioritizer_download = os.getenv(
                        "OMNIBIMOL_VARIANT_PRIORITY_ENABLE_REMOTE_DOWNLOAD",
                        "true",
                    ).strip().lower() in {"1", "true", "yes", "on"}
                self._prioritizer = VariantPrioritizer(
                    cache_path=cache_path,
                    enable_remote_download=enable_prioritizer_download,
                )
            except Exception as e:
                warnings.warn(f"Could not initialize VariantPrioritizer: {e}")

    def parse_vcf(self, vcf_text: str) -> Dict[str, Any]:
        """Parse VCF text into structured records with warnings and ingestion stats."""
        variants: List[Dict[str, Any]] = []
        warnings: List[str] = []
        errors: List[str] = []
        sample_names: List[str] = []
        parsed_count = 0
        filtered_count = 0
        retained_count = 0

        if not vcf_text or not vcf_text.strip():
            return {
                "variants": [],
                "warnings": ["VCF input is empty."],
                "errors": ["No VCF records found."],
                "stats": {"parsed": 0, "filtered": 0, "retained": 0},
                "sample_names": [],
            }

        for line_number, raw_line in enumerate(vcf_text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                columns = line.split("\t")
                if len(columns) > 9:
                    sample_names = columns[9:]
                continue
            if line.startswith("#"):
                continue

            fields = line.split("\t")
            parsed_count += 1
            if len(fields) < 8:
                errors.append(f"Line {line_number}: malformed VCF record (<8 fields).")
                continue

            chrom, pos_raw, var_id, ref, alt_raw, qual_raw, filt, info_raw = fields[:8]
            fmt = fields[8] if len(fields) > 8 else ""
            sample_values = fields[9:] if len(fields) > 9 else []

            try:
                pos = int(pos_raw)
            except ValueError:
                errors.append(f"Line {line_number}: invalid POS '{pos_raw}'.")
                continue

            if not ref or ref == ".":
                errors.append(f"Line {line_number}: missing REF allele.")
                continue
            if not alt_raw or alt_raw == ".":
                errors.append(f"Line {line_number}: missing ALT allele.")
                continue

            info_dict = self._parse_info_field(info_raw)
            qual_val = self._safe_float(qual_raw)
            low_qual = qual_val is not None and qual_val < 20.0
            failed_filter = filt not in {"PASS", ".", ""}

            if low_qual:
                warnings.append(f"Line {line_number}: low QUAL={qual_val}.")
            if failed_filter:
                warnings.append(f"Line {line_number}: FILTER={filt}.")

            format_keys = fmt.split(":") if fmt else []
            sample_dict = self._parse_sample_fields(format_keys, sample_values)
            genotype = sample_dict.get("GT")
            if not genotype:
                warnings.append(f"Line {line_number}: genotype missing.")

            filtered_count += 1 if (low_qual or failed_filter) else 0
            retained_count += 1

            for alt in alt_raw.split(","):
                variants.append(
                    {
                        "chrom": chrom,
                        "pos": pos,
                        "id": var_id if var_id != "." else "",
                        "ref": ref,
                        "alt": alt,
                        "qual": qual_val,
                        "filter": filt,
                        "info_raw": info_raw,
                        "info": info_dict,
                        "format": format_keys,
                        "samples": sample_dict,
                        "line_number": line_number,
                    }
                )

        return {
            "variants": variants,
            "warnings": warnings,
            "errors": errors,
            "stats": {"parsed": parsed_count, "filtered": filtered_count, "retained": retained_count},
            "sample_names": sample_names,
        }

    def normalize_variants(self, variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize parsed variants to canonical dictionary keys."""
        normalized: List[Dict[str, Any]] = []
        for row in variants:
            ref = str(row.get("ref", "")).upper()
            alt = str(row.get("alt", "")).upper()
            gene, transcript, protein, consequence = self._extract_annotation_fields(row.get("info", {}))

            normalized.append(
                {
                    "variant_key": f"{row.get('chrom')}:{row.get('pos')}:{ref}>{alt}",
                    "chrom": str(row.get("chrom", "")),
                    "pos": int(row.get("pos", 0)),
                    "ref": ref,
                    "alt": alt,
                    "qual": row.get("qual"),
                    "filter": row.get("filter", ""),
                    "genotype": row.get("samples", {}).get("GT", ""),
                    "allele_depth": row.get("samples", {}).get("AD", ""),
                    "gene": gene,
                    "transcript": transcript,
                    "protein": protein,
                    "consequence": consequence,
                    "info": row.get("info", {}),
                    "line_number": row.get("line_number"),
                }
            )
        return normalized

    def annotate_variant_effects(self, variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Annotate variant consequence, impact, confidence, and evidence tags."""
        annotated: List[Dict[str, Any]] = []
        for variant in variants:
            consequence = str(variant.get("consequence", "")).lower()
            gene = variant.get("gene") or self._infer_gene_from_id(variant.get("info", {}))
            variant_type = self._infer_variant_type(variant, consequence)
            impact_class, impact_score = self._estimate_impact(variant_type, consequence)
            confidence = self._estimate_confidence(variant)
            evidence_level = self._to_evidence_level(confidence)

            evidence_tags = [f"type:{variant_type}", f"impact:{impact_class}"]
            assumptions: List[str] = []
            if not consequence:
                evidence_tags.append("annotation:fallback_heuristics")
                assumptions.append("No ANN/CSQ consequence found; impact inferred from REF/ALT/position heuristics.")
            if not gene:
                assumptions.append("Gene symbol unavailable; variant contributes only to global uncertainty.")
            if variant.get("filter") not in {"PASS", ".", ""}:
                assumptions.append(f"Variant FILTER is {variant.get('filter')} and may be technical artifact.")

            annotated.append(
                {
                    **variant,
                    "gene": gene or "UNKNOWN",
                    "predicted_effect_class": impact_class,
                    "variant_type": variant_type,
                    "impact_score": round(self._clamp01(impact_score), 4),
                    "confidence": confidence,
                    "evidence_level": evidence_level,
                    "evidence_tags": evidence_tags,
                    "assumptions": assumptions,
                }
            )
        return annotated

    def aggregate_gene_impact(self, annotated_variants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate variant-level effects into normalized per-gene impact scores."""
        per_gene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for variant in annotated_variants:
            gene = str(variant.get("gene", "UNKNOWN")).strip() or "UNKNOWN"
            per_gene[gene].append(variant)
        
        gene_impact: Dict[str, Any] = {}
        for gene, rows in per_gene.items():
            burden = min(1.0, len(rows) / 5.0)
            
            # Use pathogenicity score if available, otherwise fall back to impact_score
            # Pathogenicity scores are from 0-1, impact_scores are also 0-1
            severity = self._avg([
                float(r.get("pathogenicity_score", r.get("impact_score", 0.0))) for r in rows
            ], default=0.0)
            
            zygosity_factor = self._avg([self._zygosity_factor(r.get("genotype", "")) for r in rows], default=0.5)
            confidence_factor = self._avg([self._confidence_to_numeric(r.get("confidence", "Low")) for r in rows], default=0.4)
            qual_penalty = self._avg([self._quality_penalty(r) for r in rows], default=0.0)
            mean_pathogenicity = self._avg([float(r.get("pathogenicity_score", r.get("impact_score", 0.0))) for r in rows], default=0.0)
            mean_model_confidence = self._avg([self._confidence_to_numeric(str(r.get("model_confidence", r.get("confidence", "Low")))) for r in rows], default=0.35)
            
            raw_score = 100.0 * (
                0.35 * burden
                + 0.35 * severity
                + 0.15 * zygosity_factor
                + 0.15 * confidence_factor
            )
            uncertainty_penalty = min(20.0, (1.0 - confidence_factor) * 12.0 + qual_penalty * 8.0)
            normalized = self._clamp(raw_score - uncertainty_penalty)
            
            # Determine which variants are top drivers using pathogenicity score first
            sorted_rows = sorted(
                rows,
                key=lambda r: (
                    float(r.get("pathogenicity_score", 0.0) or 0.0),
                    float(r.get("impact_score", 0.0)),
                ),
                reverse=True,
            )
            top_drivers = sorted_rows[:3]
            
            # Calculate tier distribution
            tier_counts = {1: 0, 2: 0, 3: 0}
            for r in rows:
                tier = r.get("pathogenicity_tier") or 3
                if tier in tier_counts:
                    tier_counts[tier] += 1
            
            gene_impact[gene] = {
                "gene": gene,
                "variant_count": len(rows),
                "burden": round(burden, 4),
                "severity": round(severity, 4),
                "mean_pathogenicity_score": round(mean_pathogenicity, 4),
                "mean_model_confidence": round(mean_model_confidence, 4),
                "zygosity_factor": round(zygosity_factor, 4),
                "confidence_factor": round(confidence_factor, 4),
                "uncertainty_penalty": round(uncertainty_penalty, 2),
                "score": round(normalized, 2),
                "tier_distribution": tier_counts,
                "top_driving_variants": [
                    {
                        "variant_key": d.get("variant_key"),
                        "effect": d.get("predicted_effect_class"),
                        "impact_score": d.get("impact_score"),
                        "pathogenicity_score": d.get("pathogenicity_score"),
                        "pathogenicity_tier": d.get("pathogenicity_tier"),
                        "model_confidence": d.get("model_confidence"),
                        "evidence_summary": d.get("evidence_summary"),
                        "confidence": d.get("confidence"),
                        "pathogenicity_method": d.get("pathogenicity_method", "heuristic"),
                    }
                    for d in top_drivers
                ],
            }
        
        return {
            "genes": dict(
                sorted(
                    gene_impact.items(),
                    key=lambda item: (-float(item[1].get("score", 0.0)), item[0]),
                )
            ),
            "summary": {
                "gene_count": len(gene_impact),
                "max_gene_score": round(max([v["score"] for v in gene_impact.values()], default=0.0), 2),
                "mean_gene_score": round(self._avg([v["score"] for v in gene_impact.values()], default=0.0), 2),
            },
        }

    def score_pathway_impact(self, gene_impact: Dict[str, Any], pathway_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute pathway perturbation scores from impacted genes and pathway map."""
        genes = gene_impact.get("genes", {})
        if not genes:
            return {"pathways": [], "mechanism_hypotheses": ["No impacted genes available for pathway scoring."]}

        pathway_map = self._normalize_pathway_data(pathway_data)
        scored: List[Dict[str, Any]] = []
        for pathway_name, payload in pathway_map.items():
            pathway_genes = payload.get("genes", [])
            if not pathway_genes:
                continue
            hits = [g for g in pathway_genes if g in genes]
            if not hits:
                continue

            hit_ratio = len(hits) / max(1, len(pathway_genes))
            weighted_gene_impact = self._avg([genes[g]["score"] / 100.0 for g in hits], default=0.0)
            centrality = float(payload.get("centrality", 0.5))
            topology_bonus = min(0.12, 0.12 * self._clamp01(centrality))
            score = 100.0 * min(1.0, 0.45 * hit_ratio + 0.45 * weighted_gene_impact + topology_bonus)
            confidence = self._confidence_label((len(hits) / max(1, len(pathway_genes))) * 0.6 + 0.4 * weighted_gene_impact)

            scored.append(
                {
                    "pathway_name": pathway_name,
                    "hit_ratio": round(hit_ratio, 4),
                    "weighted_gene_impact": round(weighted_gene_impact, 4),
                    "topology_bonus": round(topology_bonus, 4),
                    "impact_score": round(self._clamp(score), 2),
                    "confidence": confidence,
                    "hit_genes": hits,
                    "total_genes": len(pathway_genes),
                    "mechanism_hypothesis": self._build_mechanism_hypothesis(pathway_name, hits),
                }
            )

        scored.sort(key=lambda row: (-float(row["impact_score"]), row["pathway_name"]))
        return {
            "pathways": scored,
            "mechanism_hypotheses": [row["mechanism_hypothesis"] for row in scored[:5]],
        }

    def generate_drug_candidates(
        self,
        gene_impact: Dict[str, Any],
        pathway_impact: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build therapy/repurposing candidates from impacted genes and pathway rationale."""
        top_genes = list(gene_impact.get("genes", {}).keys())[:8]
        candidates: Dict[str, Dict[str, Any]] = {}

        for gene in top_genes:
            uniprot_id = self._resolve_uniprot_id(gene, context)
            target_data = self._fetch_target_drug_data(uniprot_id, gene)
            for approved in target_data.get("fda_approved", []):
                name = approved.get("name", "").strip()
                if not name:
                    continue
                candidates[name.lower()] = self._build_candidate(
                    drug_name=name,
                    target_gene=gene,
                    approval_status="FDA Approved",
                    source_payload=approved,
                    repurposing=False,
                    pathway_impact=pathway_impact,
                )
            for trial_row in target_data.get("clinical_trials", []):
                name = trial_row.get("name") or trial_row.get("drugs") or "Investigational Candidate"
                candidates[name.lower()] = self._build_candidate(
                    drug_name=name,
                    target_gene=gene,
                    approval_status="Clinical Trial",
                    source_payload=trial_row,
                    repurposing=False,
                    pathway_impact=pathway_impact,
                )

            repurposed_rows = self._fetch_repurposing_candidates(gene)
            for rep in repurposed_rows:
                name = rep.get("name", "").strip()
                if not name:
                    continue
                key = name.lower()
                if key not in candidates:
                    candidates[key] = self._build_candidate(
                        drug_name=name,
                        target_gene=gene,
                        approval_status=rep.get("status", "Repurposing Candidate"),
                        source_payload=rep,
                        repurposing=True,
                        pathway_impact=pathway_impact,
                    )
                else:
                    candidates[key]["repurposing_flag"] = True
                    candidates[key]["evidence_quality_summary"] = "Mixed evidence (direct + repurposing)"

        return list(candidates.values())

    def rank_therapy_options(self, candidates: List[Dict[str, Any]], weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Rank therapy candidates with missing-component renormalization."""
        resolved = self._normalize_weights(weights or {})
        ranked: List[Dict[str, Any]] = []
        for cand in candidates:
            components = {
                "target_gene_match": self._safe_float(cand.get("target_gene_match"), 0.0),
                "pathway_correction_relevance": self._safe_float(cand.get("pathway_relevance"), 0.0),
                "evidence_quality": self._safe_float(cand.get("evidence_quality"), 0.0),
                "clinical_maturity": self._safe_float(cand.get("clinical_maturity"), 0.0),
                "safety_risk_penalty": self._safe_float(cand.get("safety_risk_penalty"), 0.0),
            }
            available = {k: v for k, v in components.items() if v is not None}
            present_weight_sum = sum(resolved[k] for k in available if k in resolved)
            renorm = {k: (resolved[k] / present_weight_sum) for k in available if present_weight_sum > 0}

            positive_score = 0.0
            safety_penalty = 0.0
            for key, value in available.items():
                contrib = renorm.get(key, 0.0) * self._clamp01(float(value))
                if key == "safety_risk_penalty":
                    safety_penalty += contrib
                else:
                    positive_score += contrib

            composite = 100.0 * max(0.0, positive_score - safety_penalty)
            completeness = (len(available) / len(self.DEFAULT_WEIGHTS)) * 100.0
            confidence = self._clamp(0.7 * completeness + 30.0 * self._clamp01(self._avg(list(available.values()), default=0.3)))
            cand_copy = dict(cand)
            cand_copy["composite_score"] = round(composite, 2)
            cand_copy["ranking_confidence"] = round(confidence, 2)
            cand_copy["completeness_pct"] = round(completeness, 2)
            cand_copy["renormalized_weights"] = {k: round(v, 4) for k, v in renorm.items()}
            ranked.append(cand_copy)

        return sorted(ranked, key=lambda row: (-float(row["composite_score"]), -float(row["ranking_confidence"]), str(row.get("drug_name", "")).lower()))

    def build_explainability_payload(
        self,
        ranked_candidates: List[Dict[str, Any]],
        gene_impact: Dict[str, Any],
        pathway_impact: Dict[str, Any],
        parsing_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build transparent explainability payload with drivers, risks, and next tests."""
        top_pathways = pathway_impact.get("pathways", [])[:3]
        top_variant_drivers: List[Dict[str, Any]] = []
        for gene_payload in list(gene_impact.get("genes", {}).values())[:5]:
            top_variant_drivers.extend(gene_payload.get("top_driving_variants", [])[:3])
        payload: Dict[str, Any] = {
            "global": {
                "disclaimer": self.DISCLAIMER,
                "ingestion_stats": parsing_stats,
                "top_pathways": top_pathways,
                "pathogenicity_context": {
                    "top_driving_variants": top_variant_drivers[:10],
                    "top_gene_pathogenicity": [
                        {
                            "gene": gene_payload.get("gene"),
                            "mean_pathogenicity_score": gene_payload.get("mean_pathogenicity_score"),
                            "mean_model_confidence": gene_payload.get("mean_model_confidence"),
                            "score": gene_payload.get("score"),
                        }
                        for gene_payload in list(gene_impact.get("genes", {}).values())[:5]
                    ],
                },
            },
            "therapy_options": [],
        }

        for cand in ranked_candidates[:20]:
            drivers = [
                f"Target-gene alignment score {cand.get('target_gene_match', 0):.2f}",
                f"Pathway relevance {cand.get('pathway_relevance', 0):.2f}",
                f"Evidence quality {cand.get('evidence_quality', 0):.2f}",
            ]
            risks = self._derive_risk_flags(cand, parsing_stats)
            validations = [
                "Orthogonal variant validation (deep sequencing or ddPCR).",
                "Functional assay for target-pathway correction effect.",
                "Model-system sensitivity assay for ranked therapy candidate.",
            ]
            payload["therapy_options"].append(
                {
                    "drug_name": cand.get("drug_name"),
                    "rank_score": cand.get("composite_score"),
                    "confidence": cand.get("ranking_confidence"),
                    "top_3_drivers": drivers[:3],
                    "top_3_risks": risks[:3],
                    "pathogenicity_support": [
                        {
                            "gene": gene_payload.get("gene"),
                            "top_variant": (gene_payload.get("top_driving_variants") or [{}])[0],
                        }
                        for gene_payload in list(gene_impact.get("genes", {}).values())[:3]
                    ],
                    "required_next_validation_tests": validations,
                }
            )
        return payload

    def export_case_report(
        self,
        parsed_vcf: Dict[str, Any],
        annotated_variants: List[Dict[str, Any]],
        gene_impact: Dict[str, Any],
        pathway_impact: Dict[str, Any],
        ranked_candidates: List[Dict[str, Any]],
        explainability: Dict[str, Any],
        sample_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Export report bundle for JSON/CSV/TXT/PDF-ready markdown workflows."""
        narrative_lines = [
            "# Variant-to-Therapy Research Case Report",
            "",
            f"- Generated: {datetime.utcnow().isoformat()}Z",
            f"- Parsed variants: {parsed_vcf.get('stats', {}).get('parsed', 0)}",
            f"- Retained variants: {parsed_vcf.get('stats', {}).get('retained', 0)}",
            f"- Impacted genes: {len(gene_impact.get('genes', {}))}",
            f"- Ranked therapies: {len(ranked_candidates)}",
            "",
            "## Key Assumptions",
            "- Consequence annotation may be incomplete; fallback rules were applied where annotations were missing.",
            "- Confidence and penalties are deterministic heuristics and should be experimentally validated.",
            "",
            "## Risk Flags",
            "- Data quality risk: variant-level QUAL/FILTER issues can alter ranking.",
            "- Biological plausibility risk: pathway maps may be incomplete.",
            "- Translational risk: preclinical evidence may not transfer to patient context.",
            "- Evidence sparsity risk: low trial maturity can reduce actionability confidence.",
            "",
            f"**{self.DISCLAIMER}**",
        ]

        # Include pathogenicity score columns if available
        variant_cols = [
            "variant_key", "gene", "predicted_effect_class", "impact_score",
            "confidence", "model_confidence", "pathogenicity_score", "pathogenicity_tier",
            "pathogenicity_method", "evidence_summary", "variant_type", "consequence", "genotype",
            "qual", "filter",
        ]
        # Check for pathogenicity fields in first variant
        if annotated_variants and any(k in annotated_variants[0] for k in ["pathogenicity_score", "pathogenicity_tier", "pathogenicity_method"]):
            # Insert pathogenicity columns after impact_score
            variant_cols = [
                "variant_key", "gene", "predicted_effect_class", "impact_score",
                "pathogenicity_score", "pathogenicity_tier", "pathogenicity_method",
                "model_confidence", "evidence_summary",
                "confidence", "variant_type", "consequence", "genotype",
                "qual", "filter",
            ]

        variants_csv = self._to_csv(annotated_variants, variant_cols)
        candidates_csv = self._to_csv(
            ranked_candidates,
            [
                "drug_name",
                "target_genes",
                "composite_score",
                "ranking_confidence",
                "completeness_pct",
                "approval_status",
                "repurposing_flag",
                "trial_evidence_summary",
                "evidence_quality_summary",
            ],
        )
        report_json = {
            "meta": {"sample_metadata": sample_metadata or {}, "disclaimer": self.DISCLAIMER},
            "ingestion": parsed_vcf,
            "variants": annotated_variants,
            "gene_impact": gene_impact,
            "pathway_impact": pathway_impact,
            "therapy_ranking": ranked_candidates,
            "explainability": explainability,
        }
        return {
            "json": report_json,
            "variants_csv": variants_csv,
            "therapies_csv": candidates_csv,
            "narrative_markdown": "\n".join(narrative_lines),
        }

    def _parse_info_field(self, info_raw: str) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if not info_raw or info_raw == ".":
            return info
        for token in info_raw.split(";"):
            if "=" in token:
                k, v = token.split("=", 1)
                info[k] = v
            else:
                info[token] = True
        return info

    def _parse_sample_fields(self, format_keys: List[str], sample_values: List[str]) -> Dict[str, Any]:
        sample_dict: Dict[str, Any] = {}
        if not format_keys or not sample_values:
            return sample_dict
        first_sample = sample_values[0].split(":")
        for idx, key in enumerate(format_keys):
            sample_dict[key] = first_sample[idx] if idx < len(first_sample) else ""
        return sample_dict

    def _extract_annotation_fields(self, info: Dict[str, Any]) -> Tuple[str, str, str, str]:
        ann = str(info.get("ANN", "") or info.get("CSQ", ""))
        if not ann:
            return "", "", "", ""
        first = ann.split(",")[0]
        parts = first.split("|")
        gene = parts[3] if len(parts) > 3 else ""
        consequence = parts[1] if len(parts) > 1 else ""
        transcript = parts[6] if len(parts) > 6 else ""
        protein = parts[10] if len(parts) > 10 else ""
        return gene, transcript, protein, consequence

    def _infer_variant_type(self, variant: Dict[str, Any], consequence: str) -> str:
        ref = str(variant.get("ref", ""))
        alt = str(variant.get("alt", ""))
        if "splice" in consequence:
            return "splice"
        if len(ref) == 1 and len(alt) == 1:
            return "SNV"
        if len(ref) != len(alt):
            return "indel"
        if "missense" in consequence or "frameshift" in consequence or "stop" in consequence:
            return "protein_altering"
        return "unknown"

    def _estimate_impact(self, variant_type: str, consequence: str) -> Tuple[str, float]:
        c = consequence.lower()
        if any(tag in c for tag in ["stop_gained", "frameshift", "splice_acceptor", "splice_donor", "start_lost"]):
            return "high", 0.9
        if any(tag in c for tag in ["missense", "inframe", "protein_altering"]):
            return "moderate", 0.65
        if any(tag in c for tag in ["synonymous", "utr", "intron"]):
            return "low", 0.25
        if variant_type == "indel":
            return "moderate", 0.55
        if variant_type == "SNV":
            return "low", 0.35
        return "unknown", 0.4

    def _estimate_confidence(self, variant: Dict[str, Any]) -> str:
        qual = self._safe_float(variant.get("qual"))
        filt = str(variant.get("filter", ""))
        has_gene = bool(variant.get("gene")) and variant.get("gene") != "UNKNOWN"
        has_consequence = bool(variant.get("consequence"))

        score = 0
        score += 1 if (qual is not None and qual >= 30.0) else 0
        score += 1 if filt in {"PASS", ".", ""} else 0
        score += 1 if has_gene else 0
        score += 1 if has_consequence else 0
        if score >= 4:
            return "High"
        if score >= 2:
            return "Med"
        return "Low"

    def _to_evidence_level(self, confidence: str) -> str:
        if confidence == "High":
            return "Level A"
        if confidence == "Med":
            return "Level B"
        return "Level C"

    def _infer_gene_from_id(self, info: Dict[str, Any]) -> str:
        for key in ("GENE", "SYMBOL", "GENEINFO"):
            if key in info:
                value = str(info.get(key, ""))
                if ":" in value:
                    return value.split(":", 1)[0]
                return value
        return ""

    def _zygosity_factor(self, genotype: str) -> float:
        gt = (genotype or "").replace("|", "/")
        if gt in {"1/1", "2/2"}:
            return 1.0
        if gt in {"0/1", "1/0", "0/2", "2/0", "1/2", "2/1"}:
            return 0.7
        if gt in {"0/0", ""}:
            return 0.3
        return 0.5

    def _confidence_to_numeric(self, label: str) -> float:
        normalized = str(label or "").strip().lower()
        return {
            "high": 0.9,
            "med": 0.65,
            "medium": 0.65,
            "low": 0.35,
        }.get(normalized, 0.35)

    def _quality_penalty(self, variant: Dict[str, Any]) -> float:
        qual = self._safe_float(variant.get("qual"), 20.0)
        filter_val = str(variant.get("filter", ""))
        qual_penalty = 0.4 if qual < 20.0 else 0.0
        filter_penalty = 0.6 if filter_val not in {"PASS", ".", ""} else 0.0
        return min(1.0, qual_penalty + filter_penalty)

    def _normalize_pathway_data(self, pathway_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        if not pathway_data:
            return {}
        if "pathways" in pathway_data and isinstance(pathway_data["pathways"], list):
            normalized: Dict[str, Dict[str, Any]] = {}
            for p in pathway_data["pathways"]:
                name = p.get("pathway_name") or p.get("name") or "Unknown pathway"
                genes = p.get("genes", [])
                if not genes:
                    text_blob = f"{p.get('description', '')} {name}".upper()
                    genes = re.findall(r"\b[A-Z0-9]{2,10}\b", text_blob)[:10]
                normalized[name] = {
                    "genes": list(dict.fromkeys([str(g).upper() for g in genes])),
                    "centrality": p.get("centrality", p.get("importance_score", 0.5)),
                }
            return normalized
        return {
            str(name): {"genes": [str(g).upper() for g in payload.get("genes", [])], "centrality": payload.get("centrality", 0.5)}
            for name, payload in pathway_data.items()
            if isinstance(payload, dict)
        }

    def _build_mechanism_hypothesis(self, pathway_name: str, hit_genes: List[str]) -> str:
        genes = ", ".join(hit_genes[:3])
        return f"Perturbation in {pathway_name} may be driven by {genes}, suggesting mechanism-directed intervention."

    def _resolve_uniprot_id(self, gene: str, context: Dict[str, Any]) -> str:
        mapping = context.get("gene_to_uniprot", {}) if isinstance(context, dict) else {}
        return str(mapping.get(gene, ""))

    def _fetch_target_drug_data(self, uniprot_id: str, gene: str) -> Dict[str, Any]:
        if self.api_client and uniprot_id:
            try:
                return self._run_async(self.api_client.fetch_drugbank_targets(uniprot_id, gene)) or {}
            except Exception:
                pass
        manual = get_manual_drug_database(gene, uniprot_id or "")
        return {
            "fda_approved": manual.get("fda_approved", []),
            "clinical_trials": manual.get("clinical_trials", []),
            "investigational": [],
        }

    def _fetch_repurposing_candidates(self, gene: str) -> List[Dict[str, Any]]:
        repurposing_seed = {
            "EGFR": [{"name": "Cetuximab", "status": "FDA Approved", "evidence": "Pathway overlap"}],
            "BRCA1": [{"name": "Olaparib", "status": "FDA Approved", "evidence": "DNA repair dependency"}],
            "BRCA2": [{"name": "Olaparib", "status": "FDA Approved", "evidence": "DNA repair dependency"}],
            "TP53": [{"name": "APR-246 (Eprenetapopt)", "status": "Clinical Trial", "evidence": "p53 reactivation hypothesis"}],
            "PIK3CA": [{"name": "Alpelisib", "status": "FDA Approved", "evidence": "PI3K pathway modulation"}],
        }
        return repurposing_seed.get(gene.upper(), [])

    def _build_candidate(
        self,
        drug_name: str,
        target_gene: str,
        approval_status: str,
        source_payload: Dict[str, Any],
        repurposing: bool,
        pathway_impact: Dict[str, Any],
    ) -> Dict[str, Any]:
        meta = get_drug_metadata(drug_name)
        phase = str(source_payload.get("phase", source_payload.get("max_phase", "N/A")))
        status = str(source_payload.get("status", "Unknown"))
        trial_summary = self._trial_summary(source_payload)
        evidence = self._evidence_quality_score(source_payload, repurposing)
        maturity = self._clinical_maturity_score(approval_status, phase, status)
        pathway_relevance = self._pathway_relevance(target_gene, pathway_impact)
        target_match = 0.9 if target_gene else 0.4
        safety_penalty = 0.25 if "withdrawn" in status.lower() or "terminated" in status.lower() else 0.1

        return {
            "drug_name": drug_name,
            "target_genes": [target_gene] if target_gene else [],
            "mechanism_rationale": f"Candidate linked to {target_gene} and impacted pathways.",
            "pathway_rationale": [p.get("pathway_name") for p in pathway_impact.get("pathways", [])[:3]],
            "approval_status": approval_status or meta.get("status", "Unknown"),
            "trial_evidence_summary": trial_summary,
            "repurposing_flag": repurposing,
            "contraindication_or_risk_notes": source_payload.get("risk_notes", "No structured contraindication data in current source."),
            "evidence_quality_summary": "High" if evidence >= 0.75 else "Moderate" if evidence >= 0.45 else "Low",
            "clinical_phase_or_status": f"{phase} / {status}",
            "target_gene_match": round(target_match, 4),
            "pathway_relevance": round(pathway_relevance, 4),
            "evidence_quality": round(evidence, 4),
            "clinical_maturity": round(maturity, 4),
            "safety_risk_penalty": round(self._clamp01(safety_penalty), 4),
            "external_ids": meta,
        }

    def _trial_summary(self, source_payload: Dict[str, Any]) -> str:
        phase = source_payload.get("phase", source_payload.get("max_phase", "N/A"))
        status = source_payload.get("status", "Unknown")
        title = source_payload.get("title", source_payload.get("indication", ""))
        return f"phase={phase}; status={status}; context={str(title)[:90]}"

    def _evidence_quality_score(self, source_payload: Dict[str, Any], repurposing: bool) -> float:
        score = 0.55
        if source_payload.get("chembl_id") or source_payload.get("nct_id"):
            score += 0.2
        if source_payload.get("first_approval") not in {"N/A", None, ""}:
            score += 0.1
        if repurposing:
            score -= 0.1
        return self._clamp01(score)

    def _clinical_maturity_score(self, approval_status: str, phase: str, status: str) -> float:
        status_l = str(status).lower()
        if "approved" in approval_status.lower():
            return 1.0
        phase_s = str(phase).upper()
        if "PHASE4" in phase_s or "4" == phase_s:
            return 0.9
        if "PHASE3" in phase_s or "3" == phase_s:
            return 0.75
        if "PHASE2" in phase_s or "2" == phase_s:
            return 0.6
        if "PHASE1" in phase_s or "1" == phase_s:
            return 0.4
        if "recruiting" in status_l or "active" in status_l:
            return 0.45
        return 0.3

    def _pathway_relevance(self, gene: str, pathway_impact: Dict[str, Any]) -> float:
        for row in pathway_impact.get("pathways", []):
            if gene in row.get("hit_genes", []):
                return self._clamp01(float(row.get("impact_score", 0.0)) / 100.0)
        return 0.25

    def _derive_risk_flags(self, candidate: Dict[str, Any], parsing_stats: Dict[str, Any]) -> List[str]:
        risks = []
        if parsing_stats.get("filtered", 0) > 0:
            risks.append("Data quality risk: one or more variants failed QUAL/FILTER thresholds.")
        if float(candidate.get("evidence_quality", 0.0)) < 0.5:
            risks.append("Evidence sparsity risk: weak direct target evidence.")
        if float(candidate.get("clinical_maturity", 0.0)) < 0.5:
            risks.append("Translational risk: low clinical maturity.")
        if float(candidate.get("pathway_relevance", 0.0)) < 0.4:
            risks.append("Biological plausibility risk: limited pathway correction alignment.")
        if not risks:
            risks.append("Residual model risk: deterministic heuristic assumptions may not hold in vivo.")
        return risks

    def _normalize_weights(self, custom_weights: Dict[str, float]) -> Dict[str, float]:
        merged = dict(self.DEFAULT_WEIGHTS)
        for key, value in custom_weights.items():
            if key in merged:
                merged[key] = max(0.0, float(value))
        total = sum(merged.values())
        if total <= 0:
            return dict(self.DEFAULT_WEIGHTS)
        return {k: v / total for k, v in merged.items()}

    def _confidence_label(self, val: float) -> str:
        if val >= 0.75:
            return "High"
        if val >= 0.5:
            return "Med"
        return "Low"

    def _to_csv(self, rows: List[Dict[str, Any]], columns: List[str]) -> str:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for key, val in list(flat.items()):
                if isinstance(val, (list, dict)):
                    flat[key] = json.dumps(val)
            writer.writerow({c: flat.get(c, "") for c in columns})
        return output.getvalue()

    def _run_async(self, coro: Any) -> Any:
        try:
            return asyncio.run(coro)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)
            finally:
                loop.close()

    @staticmethod
    def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if value in (None, "", "."):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _avg(values: List[float], default: float = 0.0) -> float:
        clean = [v for v in values if v is not None and not math.isnan(v)]
        if not clean:
            return default
        return sum(clean) / len(clean)

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
        return max(lower, min(upper, float(value)))

    def score_variant_pathogenicity(
        self,
        annotated_variants: List[Dict[str, Any]],
        use_prioritization: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Enhance annotated variants with research-backed pathogenicity scoring.
        
        Uses the VariantPrioritizer model stack (XGBoost/RF/LR) trained on
        missense variant features from GPN-MSA, CADD, phyloP, ESM-1b, etc.
        
        Scientific foundations:
        - Frazer et al. 2021 (ESM-1b embeddings for variant effect prediction)
        - Cheng et al. 2023 (AlphaMissense / precomputed conservation features)
        - Notin et al. 2022 (Tranception-style protein-level modeling)
        - Landrum et al. 2018 (ClinVar ground truth for labels)
        
        Args:
            annotated_variants: List of annotated variant dictionaries from
                               annotate_variant_effects()
            use_prioritization: Whether to apply research-backed scoring
        
        Returns:
            List of variants with added pathogenicity_score, tier, and metadata
        """
        if not use_prioritization or self._prioritizer is None:
            # Fallback: use existing impact scores linearly scaled to [0, 1]
            for variant in annotated_variants:
                score = self._clamp01(float(variant.get("impact_score", 0.0)))
                variant["pathogenicity_score"] = score
                variant["pathogenicity_tier"] = self._assign_tier_from_score(score)
                variant["pathogenicity_method"] = "heuristic_fallback"
                confidence_label = str(variant.get("confidence", "Low")).title()
                variant["model_confidence"] = confidence_label
                variant["evidence_summary"] = "Heuristic fallback derived from impact_score because the prioritizer was unavailable or disabled."
                variant["confidence_label"] = confidence_label
            return annotated_variants
        
        # Prepare features for prioritizer
        variant_features = []
        for variant in annotated_variants:
            bundle = self._build_prioritizer_feature_bundle(variant)
            variant_features.append((variant, bundle))
        
        # Batch score using prioritizer
        features_list = [bundle["features"] for _, bundle in variant_features]
        try:
            predictions = [
                self._prioritizer.predict_pathogenicity(bundle["features"], feature_type=bundle["feature_type"])
                for _, bundle in variant_features
            ]
        except Exception as e:
            warnings.warn(f"Pathogenicity prediction failed: {e}")
            # Fallback to heuristic
            return self.score_variant_pathogenicity(
                annotated_variants, use_prioritization=False
            )
        
        # Merge predictions back into variants
        for (variant, bundle), prediction in zip(variant_features, predictions):
            score = prediction.get("score")
            if score is None:
                score = self._clamp01(float(variant.get("impact_score", 0.0)))
                tier = self._assign_tier_from_score(score)
                method = "heuristic_fallback"
                confidence_label = str(variant.get("confidence", "Low")).title()
                fallback_reason = prediction.get("fallback_reason") or "models_unavailable"
            else:
                tier = prediction.get("tier")
                method = prediction.get("model_used", "unknown")
                confidence_label = str(prediction.get("confidence_label", "low")).title()
                fallback_reason = prediction.get("fallback_reason")

            variant["pathogenicity_score"] = self._clamp01(float(score))
            variant["pathogenicity_tier"] = tier or self._assign_tier_from_score(variant["pathogenicity_score"])
            variant["pathogenicity_method"] = method
            variant["model_confidence"] = confidence_label
            variant["evidence_summary"] = self._build_pathogenicity_evidence_summary(variant, prediction, bundle)
            variant["confidence_label"] = confidence_label
            variant["pathogenicity_evidence"] = prediction.get(
                "evidence_features_used", []
            )
            variant["pathogenicity_metadata"] = prediction.get("metadata", {})
            
            # Track fallback reason if applicable
            if fallback_reason:
                variant["pathogenicity_fallback_reason"] = fallback_reason
        
        return annotated_variants

    def _build_prioritizer_feature_bundle(self, variant: Dict[str, Any]) -> Dict[str, Any]:
        features = self._extract_prioritizer_features(variant)
        if self._prioritizer is None:
            return {"features": features, "feature_type": "minimal", "validation": {}}

        precomputed_validation = self._prioritizer.validate_features(features, "precomputed")
        protein_validation = self._prioritizer.validate_features(features, "protein")
        if precomputed_validation["valid"]:
            feature_type = "precomputed"
        elif protein_validation["valid"]:
            feature_type = "protein"
        else:
            feature_type = "minimal"
        return {
            "features": features,
            "feature_type": feature_type,
            "validation": {
                "precomputed": precomputed_validation,
                "protein": protein_validation,
            },
        }

    def _build_pathogenicity_evidence_summary(
        self,
        variant: Dict[str, Any],
        prediction: Dict[str, Any],
        bundle: Dict[str, Any],
    ) -> str:
        model_used = prediction.get("model_used") or "heuristic_fallback"
        feature_type = bundle.get("feature_type", "minimal")
        used = prediction.get("evidence_features_used", [])
        missing = prediction.get("missing_features", [])
        pieces = [
            f"{model_used} scored a {feature_type} variant",
            f"{len(used)} features used",
        ]
        if missing:
            pieces.append(f"{len(missing)} features imputed or unavailable")
        if variant.get("consequence"):
            pieces.append(f"consequence={variant.get('consequence')}")
        fallback_reason = prediction.get("fallback_reason")
        if fallback_reason:
            pieces.append(f"fallback={fallback_reason}")
        return "; ".join(pieces)

    def _extract_prioritizer_features(self, variant: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features for the variant prioritizer from annotated variant data.
        
        Attempts to provide precomputed features first, then falls back to
        protein-level features.
        """
        features = {}
        
        # Precomputed missense features (if available from external sources)
        # These would typically come from annotation databases
        info = variant.get("info", {})
        consequence = str(variant.get("consequence", "")).lower()

        def _set_numeric_feature(source_keys: List[str], target_key: str) -> None:
            for source_key in source_keys:
                if source_key in info:
                    try:
                        features[target_key] = float(info[source_key])
                        return
                    except (ValueError, TypeError):
                        return

        _set_numeric_feature(["gpn_msa_score"], "gpn_msa_score")
        
        # CADD score
        _set_numeric_feature(["CADD", "cadd_raw"], "cadd_raw")
        if "cadd_raw" in features:
            features["cadd_phred"] = features["cadd_raw"]
        
        # PhyloP (if available)
        _set_numeric_feature(["phyloP", "phyloP100way_vertebrate"], "phyloP100way_vertebrate")
        if "phyloP100way_vertebrate" in features:
            features["phyloP241way_mammalian"] = features["phyloP100way_vertebrate"]
        
        # PhastCons (if available)
        _set_numeric_feature(["phastCons", "phastCons100way_vertebrate"], "phastCons100way_vertebrate")
        if "phastCons100way_vertebrate" in features:
            features["phastCons241way_mammalian"] = features["phastCons100way_vertebrate"]
        
        # GERP++ (if available as proxy for conservation)
        if "GERP" in info:
            try:
                features["gerp_score"] = float(info["GERP"])
            except (ValueError, TypeError):
                pass
        
        # ESM-1b embedding stats (placeholder - would come from actual embedding)
        _set_numeric_feature(["esm1b_embedding_mean"], "esm1b_embedding_mean")
        _set_numeric_feature(["esm1b_embedding_max"], "esm1b_embedding_max")
        _set_numeric_feature(["esm1b_embedding_norm"], "esm1b_embedding_norm")
        
        # NT score (nucleotide diversity proxy)
        _set_numeric_feature(["nt_score", "NT"], "nt_score")
        
        # HyenaDNA embedding (placeholder)
        _set_numeric_feature(["hyena_dna_embedding_mean"], "hyena_dna_embedding_mean")
        
        # Protein-level / AA-change features (more commonly available)
        impact_score = variant.get("impact_score", 0.0)
        
        # AA position in protein
        protein_pos = 0
        if variant.get("protein"):
            # Extract position from protein notation like p.Arg150His
            import re
            match = re.search(r'p\.\D*(\d+)', variant["protein"])
            if match:
                protein_pos = int(match.group(1))
        features["aa_position"] = float(protein_pos)
        
        # Change type encoding
        consequence_map = {
            "missense": 1.0,
            "frameshift": 0.9,
            "stop_gained": 0.95,
            "splice": 0.85,
            "inframe": 0.7,
            "synonymous": 0.1,
            "utr": 0.2,
            "intron": 0.3,
        }
        features["aa_change_type"] = max(
            [consequence_map.get(c, 0.0) for c in consequence.split(",") if c]
            + [0.0]
        )
        
        # Domain score (based on impact)
        features["domain_score"] = impact_score
        
        # Conservation proxy (use phyloP if available, else impact-based)
        features["conservation_score"] = features.get("phyloP100way_vertebrate", impact_score)
        
        # BLOSUM62 score approximation
        ref = variant.get("ref", "")
        alt = variant.get("alt", "")
        features["blosum62_score"] = self._blosum62_score(ref, alt)
        
        # Grantham distance (chemical difference between amino acids)
        features["grantham_distance"] = self._grantham_distance(ref, alt)
        
        # SIFT score (if available)
        if "SIFT" in info:
            try:
                sift_val = float(info["SIFT"])
                # SIFT scores: 0=tolerated, 1=deleterious; invert for consistency
                features["sift_score"] = 1.0 - min(1.0, max(0.0, sift_val))
            except (ValueError, TypeError):
                features["sift_score"] = impact_score
        else:
            features["sift_score"] = impact_score
        
        # PolyPhen score (if available)
        if "PolyPhen" in info:
            try:
                polyphen_val = float(info["PolyPhen"])
                features["polyphen_score"] = polyphen_val
            except (ValueError, TypeError):
                features["polyphen_score"] = impact_score
        else:
            features["polyphen_score"] = impact_score
        
        return features

    def _blosum62_score(self, aa1: str, aa2: str) -> float:
        """Calculate normalized BLOSUM62 score between two amino acids."""
        # Simplified BLOSUM62 matrix (key: aa1+aa2, value: score)
        blosum62 = {
            ("A", "A"): 4, ("A", "R"): -1, ("A", "N"): -2, ("A", "D"): -2, ("A", "C"): 0,
            ("A", "Q"): -1, ("A", "E"): -1, ("A", "G"): 0, ("A", "H"): -2, ("A", "I"): -1,
            ("A", "L"): -1, ("A", "K"): -1, ("A", "M"): -1, ("A", "F"): -2, ("A", "P"): -1,
            ("A", "S"): 1, ("A", "T"): 0, ("A", "W"): -3, ("A", "Y"): -2, ("A", "V"): 0,
            ("R", "R"): 5, ("R", "K"): 2, ("R", "N"): 0, ("R", "D"): -2, ("R", "C"): -3,
            ("R", "Q"): 1, ("R", "E"): 0, ("R", "G"): -2, ("R", "H"): 0, ("R", "I"): -3,
            ("R", "L"): -2, ("R", "M"): -1, ("R", "F"): -3, ("R", "P"): -2, ("R", "S"): -1,
            ("S", "S"): 4,
        }
        
        if not aa1 or not aa2 or aa1 == aa2:
            return 1.0
        
        # Normalize: BLOSUM62 range is roughly -4 to 11; normalize to [0, 1]
        # Same amino acid = 1.0, very different = 0.0
        key = (aa1.upper(), aa2.upper())
        rev_key = (aa2.upper(), aa1.upper())
        
        score = blosum62.get(key, blosum62.get(rev_key, -1))
        # Normalize: (score - min) / (max - min) = (score + 4) / 15
        normalized = max(0.0, min(1.0, (score + 4.0) / 11.0))
        return normalized

    def _grantham_distance(self, aa1: str, aa2: str) -> float:
        """Calculate Grantham chemical distance between amino acids."""
        # Simplified Grantham distances (smaller = more similar)
        # Scale to [0, 1] where 1.0 = identical, 0.0 = very different
        if not aa1 or not aa2 or aa1.upper() == aa2.upper():
            return 1.0
        
        # Approximate Grantham distance (typical range: 0-215)
        # Conservative estimate: polar to non-polar ≈ 100
        distance_map = {
            ("V", "I"): 12, ("V", "L"): 14, ("V", "M"): 22,  # Hydrophobic cluster
            ("S", "T"): 10, ("S", "N"): 12, ("T", "S"): 10,  # Polar similar
            ("K", "R"): 12, ("D", "E"): 10,  # Charged similar
            ("F", "Y"): 10, ("F", "W"): 18,  # Aromatic
        }
        
        key = (aa1.upper(), aa2.upper())
        rev_key = (aa2.upper(), aa1.upper())
        distance = distance_map.get(key, distance_map.get(rev_key, 60))
        
        # Normalize: 1.0 (identical) to 0.0 (max distance ~215)
        normalized = max(0.0, min(1.0, 1.0 - (distance / 215.0)))
        return normalized

    def _assign_tier_from_score(self, score: Optional[float]) -> Optional[int]:
        """Assign tier (1-3) from pathogenicity score."""
        if score is None:
            return None
        if score >= 0.9:
            return 1
        if score >= 0.7:
            return 2
        if score >= 0.5:
            return 3
        return 3  # Below 0.5 is still Tier 3

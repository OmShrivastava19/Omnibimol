"""Multi-omics fusion service for drug response prediction."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import exp, sqrt
from statistics import mean, pstdev
from typing import Any


DEFAULT_PATHWAY_GENE_MAP: dict[str, set[str]] = {
    "PI3K_AKT_MTOR": {"PIK3CA", "AKT1", "AKT2", "AKT3", "MTOR", "PTEN"},
    "DNA_DAMAGE_REPAIR": {"BRCA1", "BRCA2", "ATM", "ATR", "CHEK1", "CHEK2", "PARP1"},
    "RTK_MAPK": {"EGFR", "ERBB2", "KRAS", "NRAS", "BRAF", "MAP2K1", "MAPK1"},
    "CELL_CYCLE": {"CDK4", "CDK6", "CCND1", "RB1", "E2F1", "CDKN2A"},
    "APOPTOSIS": {"BCL2", "BAX", "MCL1", "CASP3", "CASP8", "TP53"},
    "IMMUNE_SIGNALING": {"JAK1", "JAK2", "STAT1", "STAT3", "PDCD1", "CD274"},
    "HORMONE_SIGNALING": {"AR", "ESR1", "PGR", "FOXA1", "GATA3"},
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _sigmoid(value: float) -> float:
    # Numerically stable enough for our bounded score ranges.
    return 1.0 / (1.0 + exp(-value))


def _iter_gene_values(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for gene, value in raw.items():
        if gene is None:
            continue
        out[str(gene).upper()] = _safe_float(value, default=0.0)
    return out


def _drug_hash_value(drug_name: str, smiles: str) -> float:
    base = f"{drug_name.strip().lower()}|{smiles.strip()}"
    if not base.strip("|"):
        return 0.0
    digest = sha256(base.encode("utf-8")).hexdigest()
    # Convert first bytes to deterministic range [0, 1]
    return int(digest[:8], 16) / 0xFFFFFFFF


@dataclass(frozen=True)
class FusionSettings:
    calibration_a: float = 1.0
    calibration_b: float = 0.0
    transcriptomics_weight: float = 0.45
    genomics_weight: float = 0.35
    proteomics_weight: float = 0.20


class PathwayAggregator:
    """Aggregate gene-level omics into pathway-level summaries."""

    def __init__(self, pathway_gene_map: dict[str, set[str]] | None = None):
        self.pathway_gene_map = pathway_gene_map or DEFAULT_PATHWAY_GENE_MAP

    def aggregate(
        self,
        genomics: dict[str, Any] | None,
        transcriptomics: dict[str, Any] | None,
        proteomics: dict[str, Any] | None,
    ) -> dict[str, dict[str, float]]:
        genomic_mut = _iter_gene_values((genomics or {}).get("mutations"))
        genomic_cnv = _iter_gene_values((genomics or {}).get("cnv"))
        expr = _iter_gene_values(transcriptomics or {})
        prot = _iter_gene_values(proteomics or {})

        pathway_features: dict[str, dict[str, float]] = {}
        for pathway, genes in self.pathway_gene_map.items():
            if not genes:
                continue
            gene_list = list(genes)
            mut_signal = mean(abs(genomic_mut.get(g, 0.0)) for g in gene_list)
            cnv_signal = mean(abs(genomic_cnv.get(g, 0.0)) for g in gene_list)
            expr_signal = mean(expr.get(g, 0.0) for g in gene_list)
            prot_signal = mean(prot.get(g, 0.0) for g in gene_list)
            genomic_signal = (0.6 * mut_signal) + (0.4 * cnv_signal)
            pathway_features[pathway] = {
                "genomic_signal": genomic_signal,
                "transcriptomic_signal": expr_signal,
                "proteomic_signal": prot_signal,
                "mutation_burden": mut_signal,
                "cnv_burden": cnv_signal,
            }
        return pathway_features


class OmicsEncoder:
    """Attention-like weighted pooling over pathway-level modality signals."""

    def __init__(self, settings: FusionSettings):
        self.settings = settings

    def encode(
        self,
        pathway_features: dict[str, dict[str, float]],
        modality_mask: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        pathway_scores: dict[str, float] = {}
        for pathway, row in pathway_features.items():
            g = row.get("genomic_signal", 0.0)
            t = row.get("transcriptomic_signal", 0.0)
            p = row.get("proteomic_signal", 0.0)
            pathway_score = (
                self.settings.genomics_weight * modality_mask.get("genomics", 0.0) * g
                + self.settings.transcriptomics_weight * modality_mask.get("transcriptomics", 0.0) * t
                + self.settings.proteomics_weight * modality_mask.get("proteomics", 0.0) * p
            )
            pathway_scores[pathway] = pathway_score

        if not pathway_scores:
            return 0.0, {}

        abs_total = sum(abs(v) for v in pathway_scores.values()) or 1.0
        attention_weights = {k: abs(v) / abs_total for k, v in pathway_scores.items()}
        encoded_value = sum(pathway_scores[k] * attention_weights[k] for k in pathway_scores)
        return encoded_value, attention_weights


class DrugEncoder:
    """Deterministic lightweight drug representation."""

    def encode(self, drug_name: str, smiles: str | None, descriptors: dict[str, Any] | None) -> float:
        smiles_text = smiles or ""
        desc = descriptors or {}
        hash_component = _drug_hash_value(drug_name, smiles_text)
        desc_values = [_safe_float(v) for v in desc.values() if isinstance(v, (int, float, str))]
        descriptor_component = mean(desc_values) if desc_values else 0.0
        smiles_component = min(len(smiles_text), 120) / 120.0
        raw = (0.45 * hash_component) + (0.40 * descriptor_component) + (0.15 * smiles_component)
        return _clamp(raw, low=0.0, high=1.0)


class Calibrator:
    def __init__(self, settings: FusionSettings):
        self.settings = settings

    def sensitivity_to_probability(self, sensitivity_score: float) -> float:
        calibrated = self.settings.calibration_a * sensitivity_score + self.settings.calibration_b
        return _clamp(_sigmoid(calibrated))


class Explainer:
    """Pathway-focused biological explanation helper."""

    def summarize(
        self,
        pathway_scores: dict[str, float],
        pathway_features: dict[str, dict[str, float]],
        attention_weights: dict[str, float],
        modality_mask: dict[str, float],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        sorted_pathways = sorted(
            pathway_scores.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        top_pathways: list[dict[str, Any]] = []
        top_features: list[dict[str, Any]] = []

        for pathway, score in sorted_pathways[:5]:
            row = pathway_features.get(pathway, {})
            top_pathways.append(
                {
                    "pathway": pathway,
                    "contribution_score": round(score, 4),
                    "attention_weight": round(attention_weights.get(pathway, 0.0), 4),
                    "mutation_burden": round(row.get("mutation_burden", 0.0), 4),
                    "cnv_burden": round(row.get("cnv_burden", 0.0), 4),
                }
            )
            dominant_feature = max(
                (
                    ("genomic_signal", row.get("genomic_signal", 0.0)),
                    ("transcriptomic_signal", row.get("transcriptomic_signal", 0.0)),
                    ("proteomic_signal", row.get("proteomic_signal", 0.0)),
                ),
                key=lambda p: abs(p[1]),
            )
            top_features.append(
                {
                    "pathway": pathway,
                    "feature": dominant_feature[0],
                    "value": round(dominant_feature[1], 4),
                }
            )

        modality_terms: list[str] = []
        if modality_mask.get("transcriptomics", 0.0) > 0:
            modality_terms.append("transcriptomics")
        if modality_mask.get("genomics", 0.0) > 0:
            modality_terms.append("genomics")
        if modality_mask.get("proteomics", 0.0) > 0:
            modality_terms.append("proteomics")
        modality_phrase = ", ".join(modality_terms) if modality_terms else "available omics"

        if top_pathways:
            lead = top_pathways[0]["pathway"]
            second = top_pathways[1]["pathway"] if len(top_pathways) > 1 else top_pathways[0]["pathway"]
            explanation_text = (
                f"Predicted response is primarily driven by pathway activity in {lead} "
                f"and {second}, integrating {modality_phrase}. "
                "Transcriptomics is weighted as the primary signal while mutation and CNV burden "
                "shape pathway vulnerability estimates."
            )
        else:
            explanation_text = (
                "Prediction generated with limited pathway evidence; additional omics depth may improve confidence."
            )
        return top_pathways, top_features, explanation_text


class MultiOmicsFusionService:
    """Service for flexible multi-omics + drug fusion prediction."""

    def __init__(
        self,
        *,
        settings: FusionSettings | None = None,
        pathway_gene_map: dict[str, set[str]] | None = None,
    ):
        self.settings = settings or FusionSettings()
        self.aggregator = PathwayAggregator(pathway_gene_map)
        self.omics_encoder = OmicsEncoder(self.settings)
        self.drug_encoder = DrugEncoder()
        self.calibrator = Calibrator(self.settings)
        self.explainer = Explainer()

    def predict(
        self,
        *,
        drug_name: str,
        smiles: str | None,
        drug_descriptors: dict[str, Any] | None,
        sample_omics: dict[str, Any] | None = None,
        cohort_omics: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not sample_omics and not cohort_omics:
            raise ValueError("At least one of sample_omics or cohort_omics is required")

        if sample_omics:
            sample_scores = [self._predict_single(drug_name, smiles, drug_descriptors, sample_omics)]
        else:
            sample_scores = [
                self._predict_single(drug_name, smiles, drug_descriptors, row or {})
                for row in (cohort_omics or [])
            ]
            if not sample_scores:
                raise ValueError("cohort_omics must contain at least one sample")

        probability = mean(item["predicted_response_probability"] for item in sample_scores)
        sensitivity = mean(item["predicted_sensitivity_score"] for item in sample_scores)
        uncertainty = mean(item["uncertainty"] for item in sample_scores)

        pathway_merge: dict[str, float] = {}
        for item in sample_scores:
            for pathway in item["top_pathways"]:
                name = pathway["pathway"]
                pathway_merge[name] = pathway_merge.get(name, 0.0) + abs(pathway["contribution_score"])
        ranked = sorted(pathway_merge.items(), key=lambda x: x[1], reverse=True)

        modality_summary = self._merge_modality_usage(sample_scores)
        explanation = sample_scores[0]["explanation_text"] if len(sample_scores) == 1 else (
            "Cohort-level prediction combines per-sample pathway activity patterns and "
            f"highlights consistent pathway drivers including {', '.join(k for k, _ in ranked[:2]) or 'key pathways'}."
        )

        return {
            "predicted_response_probability": round(_clamp(probability), 4),
            "predicted_sensitivity_score": round(_clamp(sensitivity), 4),
            "uncertainty": round(_clamp(uncertainty), 4),
            "top_pathways": [
                {"pathway": pathway, "aggregate_contribution": round(score, 4)}
                for pathway, score in ranked[:5]
            ],
            "top_features": sample_scores[0]["top_features"][:5],
            "explanation_text": explanation,
            "modality_usage_summary": modality_summary,
        }

    def _predict_single(
        self,
        drug_name: str,
        smiles: str | None,
        drug_descriptors: dict[str, Any] | None,
        omics: dict[str, Any],
    ) -> dict[str, Any]:
        genomics = (omics or {}).get("genomics") or {}
        transcriptomics = (omics or {}).get("transcriptomics") or {}
        proteomics = (omics or {}).get("proteomics") or {}

        has_genomics = bool(genomics)
        has_transcriptomics = bool(transcriptomics)
        has_proteomics = bool(proteomics)
        if not has_genomics:
            raise ValueError("genomics input is required")

        modality_mask = {
            "genomics": 1.0 if has_genomics else 0.0,
            "transcriptomics": 1.0 if has_transcriptomics else 0.0,
            "proteomics": 1.0 if has_proteomics else 0.0,
        }

        pathway_features = self.aggregator.aggregate(genomics, transcriptomics, proteomics)
        omics_embedding, attention_weights = self.omics_encoder.encode(pathway_features, modality_mask)
        drug_embedding = self.drug_encoder.encode(drug_name, smiles, drug_descriptors)
        synergy_term = sqrt(abs(max(omics_embedding, 0.0) * drug_embedding))
        sensitivity = _clamp((0.55 * omics_embedding) + (0.30 * drug_embedding) + (0.15 * synergy_term))
        probability = self.calibrator.sensitivity_to_probability(sensitivity)

        pathway_scores = {
            pathway: (
                self.settings.genomics_weight * modality_mask["genomics"] * row["genomic_signal"]
                + self.settings.transcriptomics_weight * modality_mask["transcriptomics"] * row["transcriptomic_signal"]
                + self.settings.proteomics_weight * modality_mask["proteomics"] * row["proteomic_signal"]
            )
            for pathway, row in pathway_features.items()
        }
        top_pathways, top_features, explanation_text = self.explainer.summarize(
            pathway_scores=pathway_scores,
            pathway_features=pathway_features,
            attention_weights=attention_weights,
            modality_mask=modality_mask,
        )

        uncertainty = self._estimate_uncertainty(
            pathway_scores=pathway_scores,
            modality_mask=modality_mask,
        )
        return {
            "predicted_response_probability": round(probability, 4),
            "predicted_sensitivity_score": round(sensitivity, 4),
            "uncertainty": round(uncertainty, 4),
            "top_pathways": top_pathways,
            "top_features": top_features,
            "explanation_text": explanation_text,
            "modality_usage_summary": {
                "used_modalities": [k for k, v in modality_mask.items() if v > 0],
                "missing_modalities": [k for k, v in modality_mask.items() if v == 0],
            },
        }

    def _estimate_uncertainty(
        self,
        *,
        pathway_scores: dict[str, float],
        modality_mask: dict[str, float],
    ) -> float:
        if not pathway_scores:
            return 1.0
        spread = pstdev(pathway_scores.values()) if len(pathway_scores) > 1 else 0.0
        modality_count = sum(1 for val in modality_mask.values() if val > 0)
        modality_penalty = 0.15 if modality_count == 1 else 0.08 if modality_count == 2 else 0.0
        proteomics_bonus = -0.05 if modality_mask.get("proteomics", 0.0) > 0 else 0.0
        uncertainty = 0.35 + (spread * 0.4) + modality_penalty + proteomics_bonus
        return _clamp(uncertainty)

    def _merge_modality_usage(self, sample_scores: list[dict[str, Any]]) -> dict[str, Any]:
        used: set[str] = set()
        missing: set[str] = set()
        for item in sample_scores:
            summary = item.get("modality_usage_summary", {})
            used.update(summary.get("used_modalities", []))
            missing.update(summary.get("missing_modalities", []))
        return {
            "samples_evaluated": len(sample_scores),
            "used_modalities": sorted(used),
            "missing_modalities": sorted(missing - used),
        }

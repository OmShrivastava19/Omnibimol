"""ChemProt interaction scoring service for drug-target evidence ranking."""

from __future__ import annotations

import asyncio
import logging
import pickle
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Optional, Protocol

import httpx
import pandas as pd

from backend.core.config import Settings, get_settings


LOGGER = logging.getLogger(__name__)

CHEMPROT_KEYWORDS = (
    "bind",
    "binding",
    "inhibit",
    "inhibition",
    "activate",
    "activation",
    "interact",
    "interaction",
    "targets",
    "target",
    "associates",
    "association",
    "complex",
    "affinity",
    "modulates",
    "modulation",
)

GENE_SYMBOL_PATTERN = re.compile(r"\b[A-Z0-9]{2,7}\b")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


class ChemProtBackend(Protocol):
    """Model backend interface used by the service and tests."""

    def predict_lora_proba(self, records: list[dict[str, Any]]) -> list[float]:
        ...

    def predict_ensemble_proba(self, records: list[dict[str, Any]]) -> list[float] | None:
        ...

    def metadata(self) -> dict[str, Any]:
        ...

    def health_snapshot(self) -> dict[str, Any]:
        ...


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_text(text: str | None) -> str:
    return " ".join((text or "").strip().split())


def _split_sentences(text: str) -> list[str]:
    cleaned = _normalize_text(text)
    if not cleaned:
        return []
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(cleaned) if sentence.strip()]
    return sentences or [cleaned]


def _first_snippet(sentences: Iterable[str], chemical: str, protein: str) -> str:
    chemical_lower = chemical.lower()
    protein_lower = protein.lower()
    fallback = ""
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if chemical_lower in sentence_lower and protein_lower in sentence_lower:
            return sentence
        if not fallback and (chemical_lower in sentence_lower or protein_lower in sentence_lower):
            fallback = sentence
    return fallback


def _find_keyword_hits(text: str) -> list[str]:
    lowered = text.lower()
    return [keyword for keyword in CHEMPROT_KEYWORDS if keyword in lowered]


def _mention_distance(text: str, chemical: str, protein: str) -> tuple[int, bool, bool]:
    lowered = text.lower()
    chemical_lower = chemical.lower()
    protein_lower = protein.lower()
    chem_index = lowered.find(chemical_lower)
    prot_index = lowered.find(protein_lower)
    chem_present = chem_index >= 0
    prot_present = prot_index >= 0
    if chem_index < 0 or prot_index < 0:
        return len(text), chem_present, prot_present
    return abs(chem_index - prot_index), chem_present, prot_present


def extract_evidence_features(
    *,
    chemical: str,
    protein: str,
    abstract: str,
    pmid: str | None = None,
    disease_context: str | None = None,
) -> dict[str, Any]:
    """Extract deterministic ChemProt evidence features from one abstract."""

    text = _normalize_text(abstract)
    sentences = _split_sentences(text)
    snippet = _first_snippet(sentences, chemical, protein) or (sentences[0] if sentences else "")
    keyword_hits = _find_keyword_hits(text)
    entity_distance, chem_in_context, prot_in_context = _mention_distance(text, chemical, protein)
    token_count = len(text.split())
    cooccurrence = int(chem_in_context and prot_in_context)

    if disease_context:
        disease_context_lower = disease_context.lower()
        context_hit = int(disease_context_lower in text.lower())
    else:
        context_hit = 0

    return {
        "chemical": chemical,
        "protein": protein,
        "pmid": pmid,
        "snippet": snippet[:500],
        "keyword_hits": keyword_hits,
        "keyword_count": len(keyword_hits),
        "cooccurrence": cooccurrence,
        "entity_distance": entity_distance,
        "context_length": token_count,
        "chem_in_context": int(chem_in_context),
        "prot_in_context": int(prot_in_context),
        "disease_context_hit": context_hit,
        "bge_cosine_sim": 0.0,
    }


def _linear_probability(features: dict[str, Any]) -> float:
    """Fallback probability used when HF artifacts are unavailable."""

    distance_penalty = 1.0 - _clamp01(_safe_float(features.get("entity_distance"), 0.0) / 320.0)
    keyword_signal = _clamp01(_safe_float(features.get("keyword_count"), 0.0) / 6.0)
    context_signal = _clamp01(_safe_float(features.get("context_length"), 0.0) / 180.0)
    cooccurrence = _clamp01(_safe_float(features.get("cooccurrence"), 0.0))
    chem_present = _clamp01(_safe_float(features.get("chem_in_context"), 0.0))
    prot_present = _clamp01(_safe_float(features.get("prot_in_context"), 0.0))
    disease_hit = _clamp01(_safe_float(features.get("disease_context_hit"), 0.0))
    cosine = _clamp01(_safe_float(features.get("bge_cosine_sim"), 0.0))

    raw = (
        0.22 * cooccurrence
        + 0.16 * keyword_signal
        + 0.18 * distance_penalty
        + 0.10 * chem_present
        + 0.10 * prot_present
        + 0.08 * context_signal
        + 0.10 * disease_hit
        + 0.06 * cosine
    )
    return _clamp01(raw)


def _feature_contributions(features: dict[str, Any], final_score: float) -> dict[str, dict[str, Any]]:
    distance_signal = 1.0 - _clamp01(_safe_float(features.get("entity_distance"), 0.0) / 320.0)
    contributions = {
        "lora_prob": {
            "value": round(_safe_float(features.get("lora_prob"), 0.0), 4),
            "direction": "positive",
            "weight": 0.55,
        },
        "cooccurrence": {
            "value": round(_safe_float(features.get("cooccurrence"), 0.0), 4),
            "direction": "positive",
            "weight": 0.14,
        },
        "keyword_count": {
            "value": round(_safe_float(features.get("keyword_count"), 0.0), 4),
            "direction": "positive",
            "weight": 0.12,
        },
        "entity_distance": {
            "value": round(_safe_float(features.get("entity_distance"), 0.0), 4),
            "direction": "negative",
            "weight": 0.08,
        },
        "context_length": {
            "value": round(_safe_float(features.get("context_length"), 0.0), 4),
            "direction": "neutral",
            "weight": 0.03,
        },
        "chem_in_context": {
            "value": round(_safe_float(features.get("chem_in_context"), 0.0), 4),
            "direction": "positive",
            "weight": 0.04,
        },
        "prot_in_context": {
            "value": round(_safe_float(features.get("prot_in_context"), 0.0), 4),
            "direction": "positive",
            "weight": 0.04,
        },
        "bge_cosine_sim": {
            "value": round(_safe_float(features.get("bge_cosine_sim"), 0.0), 4),
            "direction": "positive",
            "weight": 0.04,
        },
    }

    for row in contributions.values():
        normalized = _clamp01(_safe_float(row["value"], 0.0))
        if row["direction"] == "negative":
            row["contribution"] = round((1.0 - normalized) * row["weight"] * final_score, 4)
        elif row["direction"] == "neutral":
            row["contribution"] = round(min(normalized / 256.0, 1.0) * row["weight"] * final_score, 4)
        else:
            row["contribution"] = round(normalized * row["weight"] * final_score, 4)
    return contributions


@dataclass
class ChemProtScoredEvidence:
    pmid: str | None
    snippet: str
    features: dict[str, Any]
    lora_prob: float
    final_score: float
    reranker_used: bool
    feature_contributions: dict[str, dict[str, Any]]


class RuleBasedChemProtBackend:
    """Deterministic CPU-only fallback backend used when HF artifacts are unavailable."""

    def __init__(self, metadata: dict[str, Any]):
        self._metadata = metadata

    def predict_lora_proba(self, records: list[dict[str, Any]]) -> list[float]:
        return [_linear_probability(record) for record in records]

    def predict_ensemble_proba(self, records: list[dict[str, Any]]) -> list[float] | None:
        return None

    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def health_snapshot(self) -> dict[str, Any]:
        return {**self._metadata, "backend": "rule_based", "loaded": True, "degraded": True}


class HuggingFaceChemProtBackend:
    """Lazy HF backend that loads the BiomedBERT base model plus the ChemProt LoRA adapter."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None
        self._ensemble_model = None
        self._load_error: str | None = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            try:
                import torch
                from peft import PeftModel
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except Exception as exc:  # pragma: no cover - import availability depends on runtime image
                self._load_error = f"missing_dependency:{exc}"
                self._loaded = True
                return

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.chemprot_adapter_repo_id,
                    revision=self.settings.chemprot_adapter_revision,
                    use_fast=True,
                    local_files_only=self.settings.chemprot_local_files_only,
                )
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    self.settings.chemprot_base_model_id,
                    revision=self.settings.chemprot_base_model_revision,
                    num_labels=2,
                    torch_dtype=torch.float32,
                    local_files_only=self.settings.chemprot_local_files_only,
                )
                model = PeftModel.from_pretrained(
                    base_model,
                    self.settings.chemprot_adapter_repo_id,
                    revision=self.settings.chemprot_adapter_revision,
                    local_files_only=self.settings.chemprot_local_files_only,
                )
                model.to("cpu")
                model.eval()
                self._tokenizer = tokenizer
                self._model = model
                self._ensemble_model = self._load_ensemble_model()
                self._loaded = True
                self._load_error = None
            except Exception as exc:  # pragma: no cover - exercised through fallback tests
                self._load_error = f"load_failed:{exc}"
                self._tokenizer = None
                self._model = None
                self._ensemble_model = None
                self._loaded = True

    def _load_ensemble_model(self) -> Any | None:
        if not self.settings.chemprot_enable_ensemble:
            return None

        candidates = [
            Path("models") / "ensemble_model.pkl",
            Path(__file__).resolve().parents[2] / "models" / "ensemble_model.pkl",
        ]
        for candidate in candidates:
            if candidate.exists():
                with candidate.open("rb") as handle:
                    return pickle.load(handle)

        try:
            from huggingface_hub import hf_hub_download

            local_path = hf_hub_download(
                repo_id=self.settings.chemprot_adapter_repo_id,
                filename="ensemble_model.pkl",
                revision=self.settings.chemprot_adapter_revision,
                local_files_only=self.settings.chemprot_local_files_only,
            )
            with open(local_path, "rb") as handle:
                return pickle.load(handle)
        except Exception:
            return None

    def predict_lora_proba(self, records: list[dict[str, Any]]) -> list[float]:
        self._load()
        if self._model is None or self._tokenizer is None:
            return [_linear_probability(record) for record in records]

        try:
            import torch

            texts = [record["text"] for record in records]
            encoded = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.settings.chemprot_max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self._model(**encoded)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
            return [float(probability) for probability in probabilities]
        except Exception as exc:
            LOGGER.warning("ChemProt LoRA inference failed; falling back to deterministic score: %s", exc)
            return [_linear_probability(record) for record in records]

    def predict_ensemble_proba(self, records: list[dict[str, Any]]) -> list[float] | None:
        self._load()
        if self._ensemble_model is None:
            return None

        try:
            feature_rows = pd.DataFrame(
                [
                    {
                        key: value
                        for key, value in record.items()
                        if key != "text"
                    }
                    for record in records
                ]
            )
            if hasattr(self._ensemble_model, "predict_proba"):
                predictions = self._ensemble_model.predict_proba(feature_rows)
                if hasattr(predictions, "tolist"):
                    predictions = predictions.tolist()
                return [float(row[1] if isinstance(row, (list, tuple)) else row) for row in predictions]
            if hasattr(self._ensemble_model, "predict"):
                predictions = self._ensemble_model.predict(feature_rows)
                return [_clamp01(float(row)) for row in predictions]
        except Exception as exc:
            LOGGER.warning("ChemProt ensemble reranker failed: %s", exc)
        return None

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "huggingface" if self._model is not None else "rule_based",
            "loaded": self._loaded,
            "load_error": self._load_error,
            "ensemble_loaded": self._ensemble_model is not None,
        }

    def health_snapshot(self) -> dict[str, Any]:
        state = self.metadata()
        state.update({
            "cpu_only": True,
            "adapter_id": self.settings.chemprot_adapter_repo_id,
            "base_model_id": self.settings.chemprot_base_model_id,
            "revision": self.settings.chemprot_adapter_revision,
        })
        return state


def _default_backend_factory(settings: Settings) -> ChemProtBackend:
    backend = HuggingFaceChemProtBackend(settings)
    backend._load()
    if backend._model is None:
        return RuleBasedChemProtBackend(backend.health_snapshot())
    return backend


class ChemProtInteractionService:
    """Public service facade used by routes and tests."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        backend_factory: Callable[[Settings], ChemProtBackend] | None = None,
        abstract_fetcher: Callable[..., Any] | None = None,
    ):
        self.settings = settings or get_settings()
        self._backend_factory = backend_factory or _default_backend_factory
        self._backend: ChemProtBackend | None = None
        self._backend_lock = threading.Lock()
        self._abstract_fetcher = abstract_fetcher or self._fetch_pubmed_abstracts

    def get_backend(self) -> ChemProtBackend:
        if self._backend is not None:
            return self._backend

        with self._backend_lock:
            if self._backend is None:
                try:
                    self._backend = self._backend_factory(self.settings)
                except Exception as exc:
                    LOGGER.warning("ChemProt backend initialization failed: %s", exc)
                    self._backend = RuleBasedChemProtBackend(self.health_snapshot())
        return self._backend

    def health_snapshot(self) -> dict[str, Any]:
        backend_state = self._backend.health_snapshot() if self._backend is not None else {
            "loaded": False,
            "degraded": True,
            "backend": "uninitialized",
            "ensemble_loaded": False,
            "load_error": None,
        }
        return {
            "status": "ready" if self.settings.chemprot_enabled else "disabled",
            "enabled": self.settings.chemprot_enabled,
            "cpu_only": True,
            "adapter_id": self.settings.chemprot_adapter_repo_id,
            "base_model_id": self.settings.chemprot_base_model_id,
            "max_length": self.settings.chemprot_max_length,
            "batch_size": self.settings.chemprot_batch_size,
            "timeout_sec": self.settings.chemprot_timeout_sec,
            "ensemble_enabled": self.settings.chemprot_enable_ensemble,
            "backend": backend_state,
        }

    def _build_text(self, *, chemical: str, protein: str, abstract: str, disease_context: str | None) -> str:
        parts = [f"Chemical: {chemical}", f"Protein: {protein}"]
        if disease_context:
            parts.append(f"Context: {disease_context}")
        parts.append(f"Abstract: {abstract}")
        return " [SEP] ".join(parts)

    async def _resolve_abstracts(
        self,
        *,
        chemical: str,
        disease_context: str | None,
        pmids: list[str] | None,
        abstracts: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        if abstracts:
            for row in abstracts:
                abstract_text = _normalize_text(str(row.get("abstract") or row.get("text") or row.get("snippet") or ""))
                if not abstract_text:
                    continue
                resolved.append(
                    {
                        "pmid": str(row.get("pmid") or row.get("PMID") or ""),
                        "abstract": abstract_text,
                        "title": row.get("title"),
                    }
                )
        if resolved:
            return resolved

        fetch_result = self._abstract_fetcher(
            chemical=chemical,
            disease_context=disease_context,
            pmids=pmids,
        )
        if asyncio.iscoroutine(fetch_result):
            fetch_result = await fetch_result
        return list(fetch_result or [])

    async def _fetch_pubmed_abstracts(
        self,
        *,
        chemical: str,
        disease_context: str | None,
        pmids: list[str] | None,
    ) -> list[dict[str, Any]]:
        query_terms = [f'"{chemical}"[Title/Abstract]']
        if disease_context:
            query_terms.append(f'"{disease_context}"[Title/Abstract]')
        search_term = " AND ".join(query_terms)

        async with httpx.AsyncClient(timeout=self.settings.chemprot_timeout_sec) as client:
            if pmids:
                ids = ",".join(pmids[:20])
            else:
                search_response = await client.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                    params={"db": "pubmed", "term": search_term, "retmode": "json", "retmax": self.settings.chemprot_batch_size * 3},
                )
                search_response.raise_for_status()
                ids = ",".join(search_response.json().get("esearchresult", {}).get("idlist", []))

            if not ids:
                return []

            fetch_response = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "pubmed", "id": ids, "retmode": "xml"},
            )
            fetch_response.raise_for_status()
            return self._parse_pubmed_xml(fetch_response.text)

    def _parse_pubmed_xml(self, xml_text: str) -> list[dict[str, Any]]:
        try:
            import xml.etree.ElementTree as ET

            root = ET.fromstring(xml_text)
        except Exception:
            return []

        abstracts: list[dict[str, Any]] = []
        for article in root.findall(".//PubmedArticle"):
            pmid_elem = article.find(".//PMID")
            title_elem = article.find(".//ArticleTitle")
            abstract_elem = article.find(".//AbstractText")
            pmid = pmid_elem.text if pmid_elem is not None and pmid_elem.text else ""
            abstract = abstract_elem.text if abstract_elem is not None and abstract_elem.text else ""
            title = title_elem.text if title_elem is not None and title_elem.text else ""
            if abstract or title:
                abstracts.append({"pmid": pmid, "title": title, "abstract": f"{title}. {abstract}".strip(". ")})
        return abstracts

    def _discover_candidate_proteins(self, abstracts: list[dict[str, Any]]) -> list[str]:
        symbols: set[str] = set()
        for row in abstracts:
            text = f"{row.get('title', '')} {row.get('abstract', '')}"
            for match in GENE_SYMBOL_PATTERN.findall(text):
                if len(match) >= 3 or match in {"EGFR", "TP53", "MTOR", "AKT1", "PIK3CA", "BRCA1", "BRCA2", "ABL1"}:
                    symbols.add(match)
        return sorted(symbols)

    def _score_evidence_batch(
        self,
        *,
        chemical: str,
        protein: str,
        disease_context: str | None,
        abstracts: list[dict[str, Any]],
    ) -> list[ChemProtScoredEvidence]:
        records: list[dict[str, Any]] = []
        feature_rows: list[dict[str, Any]] = []
        for row in abstracts:
            features = extract_evidence_features(
                chemical=chemical,
                protein=protein,
                abstract=row["abstract"],
                pmid=row.get("pmid"),
                disease_context=disease_context,
            )
            feature_rows.append(features)
            records.append(
                {
                    "text": self._build_text(
                        chemical=chemical,
                        protein=protein,
                        abstract=row["abstract"],
                        disease_context=disease_context,
                    ),
                    **features,
                }
            )

        backend = self.get_backend()
        lora_scores = backend.predict_lora_proba(records)
        ensemble_scores = backend.predict_ensemble_proba(records)
        metadata = backend.metadata()

        scored: list[ChemProtScoredEvidence] = []
        for idx, features in enumerate(feature_rows):
            lora_prob = _clamp01(lora_scores[idx] if idx < len(lora_scores) else _linear_probability(features))
            reranker_used = ensemble_scores is not None and idx < len(ensemble_scores)
            final_score = _clamp01(ensemble_scores[idx] if reranker_used else lora_prob)
            enriched = dict(features)
            enriched["lora_prob"] = round(lora_prob, 4)
            enriched["final_score"] = round(final_score, 4)
            contributions = _feature_contributions(enriched, final_score)
            scored.append(
                ChemProtScoredEvidence(
                    pmid=str(features.get("pmid") or "") or None,
                    snippet=str(features.get("snippet") or ""),
                    features=enriched,
                    lora_prob=lora_prob,
                    final_score=final_score,
                    reranker_used=bool(reranker_used),
                    feature_contributions=contributions,
                )
            )
        if metadata.get("load_error"):
            LOGGER.info("ChemProt backend degraded mode active: %s", metadata.get("load_error"))
        return scored

    async def score_request(
        self,
        *,
        chemical: str,
        disease_context: str | None = None,
        candidate_proteins: list[str] | None = None,
        abstracts: list[dict[str, Any]] | None = None,
        pmids: list[str] | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        if not self.settings.chemprot_enabled:
            raise ValueError("ChemProt scoring is disabled in configuration")

        resolved_abstracts = await self._resolve_abstracts(
            chemical=chemical,
            disease_context=disease_context,
            pmids=pmids,
            abstracts=abstracts,
        )
        resolved_proteins = candidate_proteins or self._discover_candidate_proteins(resolved_abstracts)

        ranked_targets: list[dict[str, Any]] = []
        for protein in resolved_proteins:
            target_evidence = self._score_evidence_batch(
                chemical=chemical,
                protein=protein,
                disease_context=disease_context,
                abstracts=resolved_abstracts,
            )
            if not target_evidence:
                continue

            interaction_probability = mean(evidence.lora_prob for evidence in target_evidence)
            final_score = mean(evidence.final_score for evidence in target_evidence)
            reranker_used = any(evidence.reranker_used for evidence in target_evidence)
            top_evidence = sorted(target_evidence, key=lambda row: row.final_score, reverse=True)[:3]

            ranked_targets.append(
                {
                    "protein": protein,
                    "interaction_probability": round(_clamp01(interaction_probability), 4),
                    "final_score": round(_clamp01(final_score), 4),
                    "reranker_used": reranker_used,
                    "evidence": [
                        {
                            "pmid": item.pmid,
                            "sentence": item.snippet,
                            "keyword_hits": item.features.get("keyword_hits", []),
                            "cooccurrence": bool(item.features.get("cooccurrence", 0)),
                            "feature_values": item.features,
                            "feature_contributions": item.feature_contributions,
                        }
                        for item in top_evidence
                    ],
                    "model_metadata": {
                        **self.get_backend().metadata(),
                        "model_id": self.settings.chemprot_base_model_id,
                        "adapter_id": self.settings.chemprot_adapter_repo_id,
                        "version": self.settings.chemprot_base_model_revision,
                        "revision": self.settings.chemprot_adapter_revision,
                        "cpu_only": True,
                    },
                }
            )

        ranked_targets.sort(
            key=lambda row: (-row["final_score"], -row["interaction_probability"], row["protein"].lower())
        )

        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        LOGGER.info(
            "chemprot_score latency_ms=%s batch_size=%s fallback_mode=%s targets=%s evidence=%s",
            elapsed_ms,
            self.settings.chemprot_batch_size,
            not bool(self.get_backend().metadata().get("backend") == "huggingface"),
            len(ranked_targets),
            len(resolved_abstracts),
        )

        top_row = ranked_targets[0] if ranked_targets else {}
        response = {
            "ranked_targets": ranked_targets,
            "interaction_probability": top_row.get("interaction_probability", 0.0),
            "final_score": top_row.get("final_score", 0.0),
            "reranker_used": top_row.get("reranker_used", False),
            "evidence": top_row.get("evidence", []),
            "model_metadata": top_row.get("model_metadata", {
                "model_id": self.settings.chemprot_base_model_id,
                "adapter_id": self.settings.chemprot_adapter_repo_id,
                "version": self.settings.chemprot_base_model_revision,
                "revision": self.settings.chemprot_adapter_revision,
                "cpu_only": True,
            }),
            "resolved_abstracts": resolved_abstracts,
            "resolved_candidate_proteins": resolved_proteins,
            "degraded_mode": bool(self.get_backend().metadata().get("backend") != "huggingface"),
            "latency_ms": elapsed_ms,
        }
        return response


_SERVICE_SINGLETON: ChemProtInteractionService | None = None
_SERVICE_LOCK = threading.Lock()


def get_chemprot_service() -> ChemProtInteractionService:
    global _SERVICE_SINGLETON
    if _SERVICE_SINGLETON is not None:
        return _SERVICE_SINGLETON
    with _SERVICE_LOCK:
        if _SERVICE_SINGLETON is None:
            _SERVICE_SINGLETON = ChemProtInteractionService()
    return _SERVICE_SINGLETON

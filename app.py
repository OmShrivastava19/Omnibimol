# app.py - Streamlit protein analysis application
import streamlit as st
import streamlit.components.v1 as components
import asyncio
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
import time
from datetime import datetime
import json
import re
import base64
import httpx
import textwrap
from xml.etree import ElementTree as ET
import html as html_lib
import sys
import urllib.parse
from cache_manager import *
from visualizations import *
from api_client import *
from drug_repurposing_engine import DrugRepurposingEngine
from sequence_analysis import SequenceAnalysisSuite, FASTAParser
from genome_analysis_engine import GenomeAnalysisEngine
from target_prioritization_engine import TargetPrioritizationEngine
try:
    from streamlit.runtime.scriptrunner.script_runner import RerunException
except Exception:
    RerunException = None

# Optional import for 3D visualization
try:
    import py3Dmol
except ImportError:
    py3Dmol = None


_NCT_PATTERN = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)

OMNIBIMOL_REQUIRED_CONTEXT_KEYS = [
    "target_profile",
    "structure_data",
    "pathway_data",
    "ppi_data",
    "ligand_binding_data",
    "docking_data",
    "repurposing_data",
    "genome_risk_data",
    "pubmed_evidence",
    "clinical_trials_evidence",
]

OMNIBIMOL_RESEARCH_COPILOT_SYSTEM_PROMPT = textwrap.dedent("""
    You are OmniBiMol AI Research Copilot, a domain-aware biomedical assistant for target assessment, druggability analysis, and translational hypothesis generation.

    MISSION
    - Help researchers decide whether a biological target is actionable and how to validate it.
    - Produce evidence-grounded, uncertainty-aware outputs using ONLY provided internal analysis results and cited external evidence (PubMed, ClinicalTrials, curated databases).
    - Never present speculation as fact.

    OPERATING RULES
    1) Grounding First
    - Use internal computed artifacts as primary context:
      - protein annotation, sequence features, structure confidence, PPI, pathways, ligandability, docking outputs, repurposing network, genome risk outputs.
    - Use external evidence second:
      - PubMed abstracts/summaries, ClinicalTrials records, approved drug metadata.
    - Every key claim must include evidence tags:
      - [Internal:<artifact_name>] and/or [PubMed:<PMID>] and/or [Trial:<NCTID>].
    - If evidence is missing or weak, explicitly say so.

    2) Strict Data Boundaries
    - Do not fabricate PMIDs, NCT IDs, values, proteins, pathways, or mutations.
    - If data is unavailable, return: "Insufficient evidence with current context."
    - Distinguish:
      - Observed (from provided data),
      - Inferred (reasoned from observations),
      - Hypothesis (testable but unproven).

    3) Scientific Rigor
    - Report confidence per conclusion: High / Medium / Low with rationale.
    - Mention conflicting evidence when present.
    - Highlight limitations (sample size, simulated docking, model assumptions, missing assay data).
    - Avoid clinical recommendations for patient care; provide research-use guidance only.

    4) Output Quality
    - Be concise, structured, and decision-oriented.
    - Prefer ranked lists and clear next actions.
    - Include risk flags and potential failure modes.

    RESPONSE MODES
    A) If user asks "Why is this target druggable?"
    Return sections:
    1. Verdict (1-2 lines)
    2. Evidence for Druggability
    3. Evidence Against / Gaps
    4. Confidence + Why
    5. Next 3 Experiments
    6. Risk Flags
    7. Citations

    B) If user asks for "hypothesis cards"
    Generate 3-5 cards in this template:
    - Hypothesis:
    - Rationale:
    - Supporting Evidence:
    - Disconfirming Evidence:
    - Minimal Experiment:
    - Readout / Success Criteria:
    - Priority: High/Med/Low
    - Risk Level: High/Med/Low
    - Confidence: High/Med/Low
    - Citations:

    C) If user asks for "experimental next steps"
    Return:
    - Immediate (1-2 weeks), Near-term (1-2 months), Later (quarter)
    - For each step: objective, assay/model, expected signal, go/no-go threshold, key risk.

    D) If user asks for "risk flags"
    Return categorized flags:
    - Biological risk
    - Translational risk
    - Data quality risk
    - Model/simulation risk
    - Regulatory/clinical feasibility risk
    Each with severity (High/Med/Low) and mitigation.

    DECISION HEURISTICS (apply transparently)
    - Favor targets with convergent support across >=3 independent evidence types.
    - Downgrade confidence when core support depends on simulated/synthetic outputs.
    - Boost priority when:
      - tractable binding pocket evidence,
      - pathway centrality + disease relevance,
      - supportive human genetics/biomarkers,
      - existing chemical matter and trial activity.
    - Penalize when:
      - contradictory biology,
      - poor selectivity risk,
      - weak translatability or no viable assay path.

    STYLE
    - Audience: biomedical researchers and biotech decision-makers.
    - Tone: analytical, pragmatic, non-hyped.
    - Use bullet points and short paragraphs.
    - Always end with:
      - "What would increase confidence most?" (top 3 missing data items).

    INPUT CONTRACT (expected context variables)
    - target_profile
    - structure_data
    - pathway_data
    - ppi_data
    - ligand_binding_data
    - docking_data
    - repurposing_data
    - genome_risk_data
    - pubmed_evidence
    - clinical_trials_evidence
    If any are missing, list them under "Missing Context".

    SAFETY
    - Research support only; not medical advice.
    - If user requests treatment decisions for a patient, refuse and suggest consulting a licensed clinician.
""").strip()


def get_missing_omnibimol_context(context_payload: Optional[Dict]) -> List[str]:
    """Return context keys missing from the OmniBiMol copilot input contract."""
    payload = context_payload or {}
    missing_keys: List[str] = []
    for key in OMNIBIMOL_REQUIRED_CONTEXT_KEYS:
        value = payload.get(key)
        if value is None:
            missing_keys.append(key)
            continue
        if isinstance(value, dict):
            if not value:
                missing_keys.append(key)
                continue
            if "available" in value and not value.get("available"):
                missing_keys.append(key)
                continue
        if isinstance(value, list) and len(value) == 0:
            missing_keys.append(key)
    return missing_keys


def _is_patient_treatment_request(user_query: str) -> bool:
    query = (user_query or "").lower()
    treatment_terms = [
        "patient",
        "treatment",
        "dose",
        "dosage",
        "prescribe",
        "which drug should",
        "what should i take",
        "therapy recommendation",
    ]
    return any(term in query for term in treatment_terms)


def _infer_omnibimol_mode(user_query: str) -> str:
    query = (user_query or "").lower()
    if "hypothesis card" in query or "hypothesis cards" in query:
        return "hypothesis_cards"
    if "experimental next steps" in query or "next steps" in query:
        return "experimental_next_steps"
    if "risk flags" in query:
        return "risk_flags"
    if "why is this target druggable" in query or ("why" in query and "druggable" in query):
        return "druggable_why"
    return "druggable_why"


def _build_omnibimol_context_payload(data: Dict, uniprot_data: Dict) -> Dict:
    literature = data.get("literature", {})
    return {
        "target_profile": {
            "uniprot_id": uniprot_data.get("uniprot_id"),
            "gene_name": uniprot_data.get("gene_name"),
            "protein_name": uniprot_data.get("protein_name"),
            "function": uniprot_data.get("function"),
            "sequence_length": uniprot_data.get("sequence_length"),
            "go_terms": uniprot_data.get("go_terms", {}),
        },
        "structure_data": data.get("alphafold_structure") or data.get("pdb_structure"),
        "pathway_data": data.get("kegg_pathways"),
        "ppi_data": data.get("string_ppi"),
        "ligand_binding_data": data.get("chembl_ligands"),
        "docking_data": st.session_state.get("docking_results"),
        "repurposing_data": st.session_state.get("repurposing_report_data"),
        "genome_risk_data": st.session_state.get("genome_analysis_results"),
        "pubmed_evidence": literature.get("papers", []),
        "clinical_trials_evidence": data.get("clinical_trials", []),
    }


def _generate_omnibimol_copilot_response(user_query: str, context_payload: Dict) -> str:
    mode = _infer_omnibimol_mode(user_query)
    missing_context = get_missing_omnibimol_context(context_payload)

    pubmed_entries = context_payload.get("pubmed_evidence", []) or []
    pubmed_pmids = [str(p.get("pmid")) for p in pubmed_entries if p.get("pmid")]
    trial_entries = context_payload.get("clinical_trials_evidence", []) or []
    trial_ids = []
    for trial in trial_entries:
        trial_id = _extract_nct_id(trial if isinstance(trial, dict) else {})
        if trial_id:
            trial_ids.append(trial_id)

    has_structure = isinstance(context_payload.get("structure_data"), dict) and context_payload.get("structure_data", {}).get("available")
    has_pathways = isinstance(context_payload.get("pathway_data"), dict) and context_payload.get("pathway_data", {}).get("available")
    has_ppi = isinstance(context_payload.get("ppi_data"), dict) and context_payload.get("ppi_data", {}).get("available")
    has_ligands = isinstance(context_payload.get("ligand_binding_data"), dict) and context_payload.get("ligand_binding_data", {}).get("available")
    has_docking = bool(context_payload.get("docking_data"))
    has_genetics = bool(context_payload.get("genome_risk_data"))
    has_repurposing = bool(context_payload.get("repurposing_data"))

    evidence_types = sum(
        [
            bool(has_structure),
            bool(has_pathways),
            bool(has_ppi),
            bool(has_ligands),
            bool(has_docking),
            bool(has_genetics),
            bool(has_repurposing),
            bool(pubmed_pmids),
            bool(trial_ids),
        ]
    )

    confidence = "Low"
    confidence_rationale = "Fewer than 3 independent evidence types are available."
    if evidence_types >= 5:
        confidence = "High"
        confidence_rationale = "Convergent support is present across multiple independent internal and external evidence types."
    elif evidence_types >= 3:
        confidence = "Medium"
        confidence_rationale = "At least 3 independent evidence types are present, but important uncertainty remains."

    if has_docking and not (pubmed_pmids or trial_ids or has_genetics):
        confidence = "Low"
        confidence_rationale = "Core support is dominated by simulated outputs without enough orthogonal validation."

    if _is_patient_treatment_request(user_query):
        lines = [
            "Research support only; I cannot provide patient-specific treatment recommendations.",
            "Please consult a licensed clinician for clinical decisions.",
            "",
            "## Missing Context",
            *((f"- {k}" for k in missing_context) if missing_context else ["- None identified from the required contract."]),
            "",
            "What would increase confidence most?",
            "- Prospectively validated clinical outcome data linked to this target.",
            "- Orthogonal functional assays in disease-relevant models.",
            "- Curated human genetics evidence with effect size and directionality.",
        ]
        return "\n".join(lines)

    if evidence_types == 0:
        return "\n".join(
            [
                "Insufficient evidence with current context.",
                "",
                "## Missing Context",
                *((f"- {k}" for k in missing_context) if missing_context else ["- Required context artifacts are unavailable in the current session."]),
                "",
                "What would increase confidence most?",
                "- Any target-level internal artifact (structure/pathway/PPI/ligandability).",
                "- PubMed evidence with extractable PMIDs.",
                "- Clinical trial records with valid NCT identifiers.",
            ]
        )

    citations: List[str] = []
    citations.extend([f"- [Internal:structure_data]" for _ in [1] if has_structure])
    citations.extend([f"- [Internal:pathway_data]" for _ in [1] if has_pathways])
    citations.extend([f"- [Internal:ppi_data]" for _ in [1] if has_ppi])
    citations.extend([f"- [Internal:ligand_binding_data]" for _ in [1] if has_ligands])
    citations.extend([f"- [Internal:docking_data]" for _ in [1] if has_docking])
    citations.extend([f"- [Internal:repurposing_data]" for _ in [1] if has_repurposing])
    citations.extend([f"- [Internal:genome_risk_data]" for _ in [1] if has_genetics])
    citations.extend([f"- [PubMed:{pmid}]" for pmid in pubmed_pmids[:5]])
    citations.extend([f"- [Trial:{nct}]" for nct in trial_ids[:5]])
    if not citations:
        citations.append("- Insufficient evidence with current context.")

    if mode == "hypothesis_cards":
        cards: List[str] = ["## Hypothesis Cards"]
        for idx in range(1, 4):
            cards.extend(
                [
                    f"### Card {idx}",
                    f"- Hypothesis: Target perturbation modulates disease-relevant biology through mechanism pathway #{idx}.",
                    "- Rationale: Convergent internal signals suggest tractability and disease coupling. [Internal:target_profile] [Internal:pathway_data]",
                    "- Supporting Evidence: Structure/pathway/PPI/ligandability evidence available in session-specific artifacts.",
                    "- Disconfirming Evidence: Contradictory biology and weak translatability remain plausible due to incomplete orthogonal validation.",
                    "- Minimal Experiment: Perturb target in disease-relevant cells, then quantify pathway marker shift and viability.",
                    "- Readout / Success Criteria: >=20% pathway marker shift with acceptable viability window versus control.",
                    f"- Priority: {'High' if idx == 1 else 'Med'}",
                    f"- Risk Level: {'Med' if evidence_types >= 3 else 'High'}",
                    f"- Confidence: {confidence}",
                    "- Citations: [Internal:target_profile] [Internal:pathway_data] [Internal:ppi_data]",
                    "",
                ]
            )
        cards.append("## Missing Context")
        if missing_context:
            cards.extend([f"- {k}" for k in missing_context])
        else:
            cards.append("- None identified from the required contract.")
        cards.extend(
            [
                "",
                "What would increase confidence most?",
                f"- Missing artifacts: {', '.join(missing_context[:3]) if missing_context else 'No critical artifacts missing; next gains are from orthogonal validation.'}",
                "- Matched perturbation + rescue experiment in disease-relevant model.",
                "- Confirmatory external evidence (additional PMIDs / active trials).",
            ]
        )
        return "\n".join(cards)

    if mode == "experimental_next_steps":
        lines = [
            "## Experimental Next Steps",
            "- Immediate (1-2 weeks): objective=validate target engagement; assay/model=biochemical binding + rapid cellular perturbation; expected signal=directional biomarker shift; go/no-go=predefined potency/engagement threshold met; key risk=assay artifact. [Internal:ligand_binding_data] [Internal:docking_data]",
            "- Near-term (1-2 months): objective=establish mechanism and selectivity; assay/model=orthogonal cell models and pathway panels; expected signal=consistent pathway modulation; go/no-go=reproducible effect across models; key risk=off-target confounding. [Internal:pathway_data] [Internal:ppi_data]",
            "- Later (quarter): objective=translational confidence; assay/model=in vivo/advanced model + biomarker strategy; expected signal=efficacy-linked biomarker movement; go/no-go=effect size and exposure margins acceptable; key risk=poor translatability. [Internal:target_profile]",
        ]
    elif mode == "risk_flags":
        lines = [
            "## Risk Flags",
            f"- Biological risk: severity={'High' if not has_pathways else 'Med'}; mitigation=orthogonal pathway perturbation and rescue assays. [Internal:pathway_data]",
            f"- Translational risk: severity={'High' if not has_genetics else 'Med'}; mitigation=human genetics/biomarker triangulation.",
            f"- Data quality risk: severity={'High' if len(missing_context) >= 4 else 'Med'}; mitigation=complete missing contract artifacts and provenance checks.",
            f"- Model/simulation risk: severity={'High' if has_docking and not pubmed_pmids else 'Med'}; mitigation=prioritize wet-lab confirmation of docking-derived claims. [Internal:docking_data]",
            f"- Regulatory/clinical feasibility risk: severity={'High' if not trial_ids else 'Med'}; mitigation=map indication precedent and trial landscape. [Trial:{trial_ids[0]}]" if trial_ids else "- Regulatory/clinical feasibility risk: severity=High; mitigation=map indication precedent and trial landscape.",
        ]
    else:
        lines = [
            "## Verdict (1-2 lines)",
            "Target appears conditionally druggable for research prioritization, not yet de-risked for translational commitment. [Internal:target_profile]",
            "",
            "## Evidence for Druggability",
            *(["- Structural model support present for tractability assessment. [Internal:structure_data]"] if has_structure else ["- Structural support is limited in the current context."]),
            *(["- Disease-pathway mapping indicates biologically relevant network placement. [Internal:pathway_data]"] if has_pathways else ["- Pathway centrality evidence is currently limited."]),
            *(["- PPI context supports network-level relevance. [Internal:ppi_data]"] if has_ppi else ["- PPI support is currently weak or unavailable."]),
            *(["- Existing chemical matter supports initial ligandability signal. [Internal:ligand_binding_data]"] if has_ligands else ["- Ligandability evidence is weak (limited known binders)."]),
            "",
            "## Evidence Against / Gaps",
            "- Contradictory biology and selectivity risks cannot be excluded from current evidence alone.",
            "- Core support may rely on simulated outputs; external orthogonal validation may be limited.",
            "- Missing assay-level evidence constrains translatability confidence.",
            "",
            "## Confidence + Why",
            f"- {confidence}: {confidence_rationale}",
            "",
            "## Next 3 Experiments",
            "- Orthogonal target engagement assay in disease-relevant model with predefined go/no-go potency.",
            "- Mechanism-of-action test with perturbation/rescue to validate causal pathway linkage.",
            "- Early selectivity and off-target profiling across relevant protein panel.",
            "",
            "## Risk Flags",
            "- Biological risk: pathway compensation may mask or invert expected response.",
            "- Model/simulation risk: docking-derived claims may not transfer to biochemical activity.",
            "- Translational risk: biomarker and genetics support may be incomplete.",
            "",
            "## Citations",
            *citations,
        ]

    lines.extend(["", "## Missing Context"])
    if missing_context:
        lines.extend([f"- {key}" for key in missing_context])
    else:
        lines.append("- None identified from the required contract.")

    lines.extend(
        [
            "",
            "What would increase confidence most?",
            f"- Missing artifacts: {', '.join(missing_context[:3]) if missing_context else 'No critical artifact missing; prioritize orthogonal validation quality.'}",
            "- Prospective orthogonal validation in disease-relevant model systems.",
            "- Additional external support from PubMed/ClinicalTrials tied to this target/indication.",
        ]
    )
    return "\n".join(lines)


def _normalize_nct_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    match = _NCT_PATTERN.search(str(value))
    if match:
        return match.group(0).upper()
    return None


def _extract_nct_id(trial: Dict) -> Optional[str]:
    for key in ("nct_id", "trial_id", "nctId", "nct", "id", "nct_number"):
        nct_id = _normalize_nct_id(trial.get(key))
        if nct_id:
            return nct_id
    for key in ("url", "link", "clinicaltrials_url"):
        nct_id = _normalize_nct_id(trial.get(key))
        if nct_id:
            return nct_id
    return _normalize_nct_id(trial.get("title"))


def _build_clinicaltrials_url(nct_id: Optional[str]) -> Optional[str]:
    if not nct_id:
        return None
    return f"https://clinicaltrials.gov/study/{nct_id}"


def _format_phase(phase: Optional[str]) -> str:
    if not phase:
        return "N/A"
    phase_upper = str(phase).upper()
    if phase_upper in ("N/A", "NA"):
        return "N/A"
    if phase_upper == "EARLY_PHASE1":
        return "Early Phase 1"
    if phase_upper.startswith("PHASE"):
        if "_" in phase_upper:
            parts = [p for p in phase_upper.split("_") if p.startswith("PHASE")]
            numbers = [p.replace("PHASE", "").strip() for p in parts if p.replace("PHASE", "").strip()]
            if numbers:
                return f"Phase {'/'.join(numbers)}"
        num = phase_upper.replace("PHASE", "").strip()
        if num:
            return f"Phase {num}"
    return str(phase).replace("_", " ").title()


def _format_status(status: Optional[str]) -> str:
    if not status:
        return "Unknown"
    return str(status).replace("_", " ").title()


def render_kegg_interactive_pathway(first_result: Dict, kegg_protein_id: Optional[str] = None):
    """
    Render an interactive KEGG pathway map using the official PNG image + KGML overlay.
    
    - Preserves original KEGG layout (no redraw)
    - Adds hover tooltips and click-through links for genes/proteins/enzymes
    - Gracefully falls back to static image if anything fails
    """
    pathway_id = first_result.get("pathway_id")
    image_url = first_result.get("kegg_image_url")
    pathway_name = first_result.get("pathway_name", "")
    
    if not pathway_id or not image_url:
        # Fallback to static image if required fields are missing
        st.image(
            image_url or "",
            width='stretch',
            caption=f"{pathway_name} - Visual representation from KEGG"
        )
        return
    
    kgml_url = f"https://rest.kegg.jp/get/{pathway_id}/kgml"
    
    try:
        resp = httpx.get(kgml_url, timeout=20.0)
        resp.raise_for_status()
        kgml_xml = resp.text
    except Exception:
        # If KGML fetch fails, keep existing static behaviour
        st.info("Interactive KEGG map is temporarily unavailable. Showing static pathway image instead.")
        st.image(
            image_url,
            width='stretch',
            caption=f"{pathway_name} - Visual representation from KEGG"
        )
        return
    
    # Parse KGML entries for genes/proteins/enzymes
    try:
        root = ET.fromstring(kgml_xml)
    except Exception:
        st.info("Could not parse KEGG KGML for this pathway. Showing static pathway image instead.")
        st.image(
            image_url,
            width='stretch',
            caption=f"{pathway_name} - Visual representation from KEGG"
        )
        return
    
    interactive_entries: List[Dict] = []
    
    for entry in root.findall("entry"):
        etype = entry.get("type", "")
        # Focus on biological entities; ignore purely graphical/map entries
        if etype not in ("gene", "ortholog", "enzyme", "compound"):
            continue
        
        graphics = entry.find("graphics")
        if graphics is None:
            continue
        
        try:
            x = float(graphics.get("x", "0"))
            y = float(graphics.get("y", "0"))
            w = float(graphics.get("width", "0"))
            h = float(graphics.get("height", "0"))
        except ValueError:
            continue
        
        if w == 0 or h == 0:
            continue
        
        # KEGG entry name typically contains one or more IDs, e.g. "hsa:1234 hsa:5678"
        entry_name = entry.get("name", "")
        graphics_label = graphics.get("name") or entry_name
        
        # Try to derive a short symbol and description from the label
        symbol = graphics_label
        description = ""
        if graphics_label and " " in graphics_label:
            parts = graphics_label.split(",")[0].split(" ", 1)
            symbol = parts[0]
            if len(parts) > 1:
                description = parts[1]
        
        # Build KEGG link; fall back to dbget-bin if link attribute is missing
        link = entry.get("link", "")
        if not link and entry_name:
            first_token = entry_name.split()[0]
            link = f"https://www.kegg.jp/dbget-bin/www_bget?{first_token}"
        
        is_highlight = False
        if kegg_protein_id and entry_name:
            # Highlight if this gene box includes the current protein's KEGG ID
            if kegg_protein_id in entry_name.split():
                is_highlight = True
        
        interactive_entries.append(
            {
                "id": entry.get("id", ""),
                "etype": etype,
                "kegg_ids": entry_name,
                "label": graphics_label,
                "symbol": symbol,
                "description": description,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "link": link,
                "highlight": is_highlight,
            }
        )
    
    if not interactive_entries:
        # Nothing to overlay; show static image
        st.image(
            image_url,
            width='stretch',
            caption=f"{pathway_name} - Visual representation from KEGG"
        )
        return
    
    # Prepare JSON payload for client-side JavaScript
    try:
        entries_json = json.dumps(interactive_entries)
    except TypeError:
        # Fallback: no interactivity if JSON serialization fails
        st.image(
            image_url,
            width='stretch',
            caption=f"{pathway_name} - Visual representation from KEGG"
        )
        return
    
    escaped_image_url = html_lib.escape(image_url, quote=True)
    escaped_title = html_lib.escape(pathway_name, quote=True)
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <style>
        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background-color: #fafafa;
        }}
        .kegg-container {{
            position: relative;
            display: inline-block;
            max-width: 100%;
            border: 1px solid #e0e0e0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            background-color: #ffffff;
        }}
        .kegg-bg {{
            display: block;
            max-width: 100%;
            height: auto;
        }}
        .kegg-overlay {{
            position: absolute;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            pointer-events: none; /* regions will re-enable pointer events */
        }}
        .kegg-node {{
            position: absolute;
            border: 1px solid rgba(0, 123, 255, 0.6);
            background-color: rgba(0, 123, 255, 0.08);
            box-sizing: border-box;
            cursor: pointer;
            pointer-events: auto;
            transition: background-color 0.1s ease, border-color 0.1s ease;
        }}
        .kegg-node:hover {{
            border-color: rgba(0, 123, 255, 0.9);
            background-color: rgba(0, 123, 255, 0.18);
        }}
        .kegg-node.highlight {{
            border: 2px solid rgba(255, 165, 0, 0.9);
            background-color: rgba(255, 215, 0, 0.18);
        }}
        .kegg-tooltip {{
            position: absolute;
            z-index: 10;
            background-color: #fffff7;
            border: 1px solid #b0b0b0;
            padding: 6px 8px;
            font-size: 11px;
            color: #000;
            box-shadow: 0 2px 4px rgba(0,0,0,0.15);
            border-radius: 2px;
            white-space: nowrap;
            pointer-events: none;
            display: none;
        }}
        .kegg-tooltip-title {{
            font-weight: 600;
            margin-bottom: 2px;
        }}
        .kegg-tooltip-id {{
            color: #0055aa;
        }}
        .kegg-tooltip-link a {{
            color: #0055aa;
            text-decoration: none;
        }}
        .kegg-tooltip-link a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="kegg-container" id="kegg-container" aria-label="{escaped_title}">
        <img id="kegg-bg" class="kegg-bg" src="{escaped_image_url}" alt="{escaped_title}" />
        <div id="kegg-overlay" class="kegg-overlay"></div>
        <div id="kegg-tooltip" class="kegg-tooltip"></div>
    </div>
    <script>
        (function() {{
            const entries = {entries_json};
            const container = document.getElementById('kegg-container');
            const img = document.getElementById('kegg-bg');
            const overlay = document.getElementById('kegg-overlay');
            const tooltip = document.getElementById('kegg-tooltip');

            function clearOverlay() {{
                while (overlay.firstChild) {{
                    overlay.removeChild(overlay.firstChild);
                }}
            }}

            function positionTooltip(evt, entry) {{
                const rect = container.getBoundingClientRect();
                const x = evt.clientX - rect.left + 10;
                const y = evt.clientY - rect.top + 10;
                tooltip.style.left = x + 'px';
                tooltip.style.top = y + 'px';
            }}

            function showTooltip(evt, entry) {{
                const label = entry.symbol || entry.label || '';
                const description = entry.description || '';
                const ids = entry.kegg_ids || '';
                const link = entry.link || '';

                let html = '';
                if (label) {{
                    html += '<div class="kegg-tooltip-title">' + label + '</div>';
                }}
                if (description) {{
                    html += '<div>' + description + '</div>';
                }}
                if (ids) {{
                    html += '<div class="kegg-tooltip-id">' + ids + '</div>';
                }}
                if (link) {{
                    html += '<div class="kegg-tooltip-link"><a href="' + link + '" target="_blank" rel="noopener noreferrer">Open in KEGG ↗</a></div>';
                }}

                tooltip.innerHTML = html;
                tooltip.style.display = html ? 'block' : 'none';
                positionTooltip(evt, entry);
            }}

            function hideTooltip() {{
                tooltip.style.display = 'none';
            }}

            function buildOverlay() {{
                if (!img.naturalWidth || !img.naturalHeight) {{
                    return;
                }}
                clearOverlay();

                const natW = img.naturalWidth;
                const natH = img.naturalHeight;

                entries.forEach(entry => {{
                    const x = entry.x;
                    const y = entry.y;
                    const w = entry.width;
                    const h = entry.height;

                    // KGML x,y are center coordinates; convert to top-left
                    const left = (x - w / 2) / natW * 100;
                    const top = (y - h / 2) / natH * 100;
                    const width = (w / natW) * 100;
                    const height = (h / natH) * 100;

                    const node = document.createElement('div');
                    node.className = 'kegg-node' + (entry.highlight ? ' highlight' : '');
                    node.style.left = left + '%';
                    node.style.top = top + '%';
                    node.style.width = width + '%';
                    node.style.height = height + '%';

                    node.addEventListener('mouseenter', function(evt) {{
                        showTooltip(evt, entry);
                    }});
                    node.addEventListener('mousemove', function(evt) {{
                        positionTooltip(evt, entry);
                    }});
                    node.addEventListener('mouseleave', function() {{
                        hideTooltip();
                    }});
                    node.addEventListener('click', function() {{
                        if (entry.link) {{
                            window.open(entry.link, '_blank', 'noopener');
                        }}
                    }});

                    overlay.appendChild(node);
                }});
            }}

            if (img.complete) {{
                buildOverlay();
            }} else {{
                img.addEventListener('load', buildOverlay);
            }}
        }})();
    </script>
</body>
</html>
    """
    
    # Render inside Streamlit using components.html
    components.html(html_content, height=650, scrolling=True)

# app.py - Main Streamlit application
def main():
    """Main OmniBiMol (MVP)"""
    
    # Page configuration
    st.set_page_config(
        page_title="OmniBiMol - Protein Analysis Platform",
        page_icon="icons/Omnibimol_logo.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional bioinformatics styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: 700;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .info-card {
            background-color: #f8f9fa;
            border-left: 4px solid #1f77b4;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
            color: #000;
            font-size: 0.95rem;
        }
        .metric-card {
            background-color: #e7f3ff;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            margin: 0.5rem 0;
        }
        .go-tag {
            display: inline-block;
            background-color: #e8f4f8;
            color: #1f77b4;
            padding: 0.4rem 0.8rem;
            margin: 0.2rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 500;
            border: 1px solid #1f77b4;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header with banner
    st.image("icons/Omnibimol_banner.png", width='stretch')
    
    # Initialize cache and API client
    if 'cache_manager' not in st.session_state:
        st.session_state.cache_manager = CacheManager()
    
    if 'api_client' not in st.session_state or not hasattr(st.session_state.api_client, "fetch_clinical_trials_by_drug"):
        st.session_state.api_client = ProteinAPIClient(st.session_state.cache_manager)
    
    # Sidebar
    with st.sidebar:
        st.header("🧬 OmniBiMol")
        
        # Page selector
        pages = [
            "Protein Analysis",
            "Sequence Analysis",
            "Whole Genome Sequencing",
            "Drugs & Clinical Trials",
            "🎯 Target Prioritization",
        ]
        current_page = st.session_state.get("current_page", "Protein Analysis")
        current_index = pages.index(current_page) if current_page in pages else 0
        page = st.radio(
            "Navigate",
            pages,
            index=current_index,
            key="page_selector"
        )
        st.session_state.current_page = page
        
        st.divider()
        
        st.header("📋 About")
        st.markdown("""
        **OmniBiMol (MVP)**
        
        Integrated protein analysis platform combining:
        - UniProt: Protein function & annotations
        - Human Protein Atlas: Expression data
        - AlphaFold & PDB: Structural information
        - KEGG: Pathway mapping
        - GO: Gene ontology annotations
        - EMBL-EBI: Sequence analysis
        - NCBI BLAST: Homology search
        - EMBOSS Needle: Sequence alignment
        - Sequence Analysis Suite: MSA, Phylogeny, Domains, Motifs
        - And more...
        
        **Features:**
        - Real-time data retrieval
        - Interactive visualizations
        - 24-hour caching
        - Mobile-responsive design
        - User-friendly interface
        - Extensible architecture
        - Open-source & free to use

        **Developed by:** Team BhUOm
        """)

        st.divider()

        if st.button("🔄 Clear Cache", key="sidebar_clear_cache"):
            # Use comprehensive cache clearing function
            clear_app_cache()
            st.success("✅ Cache and app state cleared. Refreshing...")
            st.rerun()
    
    # Route to appropriate page
    if st.session_state.get('current_page') == "Sequence Analysis":
        render_sequence_analysis_page()
        return
    elif st.session_state.get('current_page') == "Whole Genome Sequencing":
        render_whole_genome_sequencing_page()
        return
    elif st.session_state.get('current_page') == "Drugs & Clinical Trials":
        render_drugs_clinical_trials_page()
        return
    elif st.session_state.get('current_page') == "🎯 Target Prioritization":
        render_target_prioritization_page()
        return
    
    # Define nested helper function for report generation
    def generate_full_report(prediction: Dict, protein_data: Dict) -> str:
        """Generate text report of all predictions"""
        report = f"""
    COMPREHENSIVE BINDING ANALYSIS REPORT
    =====================================

    Protein: {protein_data.get('uniprot_id', 'N/A')}
    Gene: {protein_data.get('gene_name', 'N/A')}
    Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    BINDING RULES EXTRACTED
    -----------------------
    {json.dumps(prediction.get('binding_rules', {}), indent=2)}

    TOP PREDICTED BINDERS (KNOWN LIGANDS)
    -------------------------------------
    """
        
        for idx, p in enumerate(prediction.get('known_ligands', [])[:10], 1):
            comp = p['compound']
            report += f"""
    {idx}. {comp['name']}
    Score: {p['predicted_score']}/100
    Confidence: {p['confidence_level']} ({p['confidence']:.0%})
    Predicted Affinity: {p['predicted_affinity']:.2f} kcal/mol
    Experimental: {comp.get('activity_value', 'N/A')} {comp.get('activity_units', '')}
    Reasons: {'; '.join(p['reasons'])}
    """
        
        report += "\n\nRECOMMENDATIONS\n"
        report += "-" * 50 + "\n"
        
        for rec in prediction.get('recommendations', []):
            report += f"""
    {rec['type']}: {rec['compound']}
    Action: {rec['action']}
    Priority: {rec['priority']}
    """
        
        return report
    
    # Main input section
    st.header("🔍 Protein Search")
    
    def trigger_search():
        """Callback to trigger search when Enter is pressed"""
        if st.session_state.get('search_input_key'):
            st.session_state.trigger_search = True
    
    protein_input = st.text_input(
        "Enter Protein Name or Gene Symbol:",
        value=st.session_state.get('protein_input', ''),
        placeholder="e.g., TP53, BRCA1, Insulin (Press Enter to search)",
        help="Enter a protein name, gene symbol, or UniProt ID",
        key="search_input_key",
        on_change=trigger_search
    )
    
    # Process search (triggered by Enter key or button)
    if (st.session_state.get('trigger_search') or st.button("🔎 Search", key="main_search_button", type="primary", width='stretch')) and protein_input:
        with st.spinner("🔍 Searching UniProt database..."):
            # Search UniProt using cached function
            search_results = cached_search_uniprot(protein_input, st.session_state.api_client)
            
            if not search_results:
                st.error("❌ No results found. Please check your input and try again.")
                st.stop()
            
            # Store results in session state
            st.session_state.search_results = search_results
            st.session_state.show_results = True
            st.session_state.trigger_search = False
    
    # Display search results for confirmation
    if st.session_state.get('show_results') and st.session_state.get('search_results'):
        st.subheader("Select Protein:")
        
        results = st.session_state.search_results
        
        if len(results) == 1:
            st.info(f"✅ Found: **{results[0]['protein_name']}** ({results[0]['gene_name']}) - {results[0]['uniprot_id']}")
            selected_uniprot_id = results[0]['uniprot_id']
            auto_load = True
        else:
            # Multiple results - let user choose
            options = [
                f"{r['protein_name']} ({r['gene_name']}) - {r['uniprot_id']} | {r['organism']}"
                for r in results
            ]
            
            selected_idx = st.radio("Multiple matches found:", range(len(options)), format_func=lambda i: options[i])
            selected_uniprot_id = results[selected_idx]['uniprot_id']
            selected_gene_name = results[selected_idx]['gene_name']
            auto_load = False
        
        # Get gene name for selected protein
        if auto_load:
            selected_gene_name = results[0]['gene_name']
        
        # Confirm and load data
        if auto_load or st.button("✅ Confirm Selection", key="protein_confirm_selection", type="primary"):
            with st.spinner("📊 Fetching protein data..."):
                start_time = time.time()
                
                # Fetch all data using cached function
                all_data = cached_fetch_all_data(selected_uniprot_id, selected_gene_name, st.session_state.api_client)
                
                fetch_time = time.time() - start_time
                
                # Store in session state
                st.session_state.current_data = all_data
                st.session_state.current_uniprot_id = selected_uniprot_id
                st.session_state.fetch_time = fetch_time
                st.session_state.show_results = False
                
                st.success(f"✅ Data loaded in {fetch_time:.2f} seconds!")
                st.rerun()
    
    # Display protein data if available
    if st.session_state.get('current_data'):
        data = st.session_state.current_data
        uniprot_data = data['uniprot_data']
        tissue_df = data['tissue_expression']
        subcellular_df = data['subcellular']
        
        st.divider()
        
        # Section 1: Protein Information
        st.header(f"📖 Protein Information: {st.session_state.current_uniprot_id}")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:#1f77b4;">{uniprot_data.get('sequence_length', 0):,}</h3>
                    <p style="margin:0; color:#666;">Amino Acids</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:#1f77b4;">{uniprot_data.get('mass', 0):,.0f}</h3>
                    <p style="margin:0; color:#666;">Molecular Weight (Da)</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            tissue_count = len(tissue_df[tissue_df["level_numeric"] > 0]) if not tissue_df.empty else 0
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:#1f77b4;">{tissue_count}</h3>
                    <p style="margin:0; color:#666;">Expressed Tissues</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            go_count = sum(len(v) for v in uniprot_data.get("go_terms", {}).values())
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:#1f77b4;">{go_count}</h3>
                    <p style="margin:0; color:#666;">GO Terms</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Function description
        st.subheader("🔬 Protein Function")
        st.markdown(f"""
            <div class="info-card">
                {uniprot_data.get('function', 'No functional annotation available')}
            </div>
        """, unsafe_allow_html=True)
        # GO Terms
        st.subheader("🏷️ Gene Ontology Terms")
        
        go_terms = uniprot_data.get('go_terms', {})
        
        for category, terms in go_terms.items():
            if terms:
                st.markdown(f"**{category}:**")
                tags_html = "".join([f'<span class="go-tag">{term}</span>' for term in terms[:10]])
                st.markdown(tags_html, unsafe_allow_html=True)

                if len(terms) > 10:
                    with st.expander(f"+{len(terms)-10} more"):
                        extra_tags_html = "".join([f'<span class="go-tag">{term}</span>' for term in terms[10:]])
                        st.markdown(extra_tags_html, unsafe_allow_html=True)

                st.markdown("")
        
        # GO terms chart
        if go_count > 0:
            fig_go = ProteinVisualizer.create_go_terms_chart(go_terms)
            st.plotly_chart(fig_go, width='stretch')
        
        st.divider()

        # Section 2: FASTA Sequence & BLAST Analysis
        st.header("🧬 Protein Sequence Analysis")
        
        # Create tabs
        sequence_tabs = st.tabs(["📄 FASTA Sequence", "🔬 Sequence Composition", "🔍 BLAST Homology Search", "🧬 EMBL Features & Alignment"])
        
        # Tab 1: FASTA Sequence
        with sequence_tabs[0]:
            st.subheader("FASTA Format Sequence")
            
            sequence = uniprot_data.get('sequence', '')
            
            if sequence:
                # Generate FASTA
                fasta_sequence = st.session_state.api_client.get_fasta_sequence(uniprot_data)
                
                # Display in text area
                st.text_area(
                    "Protein Sequence (FASTA format):",
                    fasta_sequence,
                    height=300,
                    help="Standard FASTA format with 60 characters per line"
                )
                
                # Sequence statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Length", f"{len(sequence)} aa")
                with col2:
                    st.metric("Molecular Weight", f"{uniprot_data.get('mass', 0):,.0f} Da")
                with col3:
                    # Calculate isoelectric point (simplified)
                    basic = sequence.count('K') + sequence.count('R') + sequence.count('H')
                    acidic = sequence.count('D') + sequence.count('E')
                    st.metric("Basic Residues", basic)
                with col4:
                    st.metric("Acidic Residues", acidic)
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 Download FASTA",
                        fasta_sequence,
                        f"{st.session_state.current_uniprot_id}.fasta",
                        "text/plain",
                        help="Download sequence in FASTA format"
                    )
                with col2:
                    st.download_button(
                        "📥 Download Raw Sequence",
                        sequence,
                        f"{st.session_state.current_uniprot_id}_sequence.txt",
                        "text/plain",
                        help="Download sequence without header"
                    )
                
            else:
                st.warning("⚠️ No sequence data available")
        
        # Tab 2: Sequence Composition
        with sequence_tabs[1]:
            st.subheader("Amino Acid Composition Analysis")
            
            sequence = uniprot_data.get('sequence', '')
            
            if sequence:
                # Analyze composition
                composition = ProteinVisualizer.analyze_sequence_composition(sequence)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="margin:0; color:#ff7f0e;">{composition['hydrophobic_percent']:.1f}%</h3>
                            <p style="margin:0; color:#666;">Hydrophobic</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="margin:0; color:#2ca02c;">{composition['polar_percent']:.1f}%</h3>
                            <p style="margin:0; color:#666;">Polar</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="margin:0; color:#d62728;">{composition['charged_percent']:.1f}%</h3>
                            <p style="margin:0; color:#666;">Charged</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Composition chart
                fig_composition = ProteinVisualizer.create_sequence_composition_chart(composition)
                st.plotly_chart(fig_composition, width='stretch')
                
                st.info("""
                **Color Legend:**
                - 🟠 **Orange**: Hydrophobic amino acids (A, V, I, L, M, F, W, P)
                - 🟢 **Green**: Polar amino acids (S, T, Y, N, Q, C)
                - 🔴 **Red**: Charged amino acids (K, R, H, D, E)
                - ⚫ **Gray**: Other (G)
                """)
                
            else:
                st.warning("⚠️ No sequence data available for analysis")
        
        # Tab 3: BLAST / Diamond Search
        with sequence_tabs[2]:
            st.subheader("BLAST Homology Search")
            
            sequence = uniprot_data.get('sequence', '')
            
            if sequence:
                st.info("""
                **About Homology Search:**
                - ⚡ **SwissProt First**: Fast NCBI BLAST search against curated Swiss-Prot database
                - 🔄 **Automatic Fallback**: Falls back to comprehensive nr database if SwissProt returns no results
                - 🧬 Uses full sequence for maximum biological accuracy
                - 🏆 Returns top 15 matches from the successful database
                - 💾 Results cached for 24 hours
                """)
                
                # Cache check
                if (
                    'blast_results' not in st.session_state or
                    st.session_state.get('blast_protein_id') != st.session_state.current_uniprot_id
                ):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(
                            f"**Full sequence length:** {len(sequence)} amino acids | **Target hits:** 15"
                        )
                    with col2:
                        run_search = st.button(
                            "🚀 Run Homology Search",
                            type="primary",
                            key="blast_run_search"
                        )

                    if run_search:
                        status_placeholder = st.empty()
                        debug_placeholder = st.empty()

                        start_time = time.time()
                        elapsed = 0.0  
                        max_search_time = 180  # Slightly higher timeout for remote BLAST
                        
                        status_placeholder.info("⚡ Running NCBI BLAST (Swiss-Prot with nr fallback)...")
                        
                        try:
                            blast_results = cached_run_blast_search(
                                sequence,
                                st.session_state.current_uniprot_id,
                                st.session_state.api_client
                            )
                        except Exception as e:
                            blast_results = {
                                "available": False,
                                "error": f"BLAST search error: {str(e)}"
                            }
                        
                        elapsed = time.time() - start_time

                        status_placeholder.empty()
                        debug_placeholder.empty()

                        # Store results
                        st.session_state.blast_results = blast_results
                        st.session_state.blast_protein_id = st.session_state.current_uniprot_id
                        st.session_state.blast_time = elapsed

                        st.rerun()

                # ---------------- DISPLAY RESULTS ----------------

                if (
                    'blast_results' in st.session_state and
                    st.session_state.get('blast_protein_id') == st.session_state.current_uniprot_id
                ):
                    blast_data = st.session_state.blast_results

                    if blast_data.get('available') and blast_data.get('hits'):
                        hits = blast_data['hits']
                        elapsed = st.session_state.get('blast_time', 0)

                        engine = blast_data.get("engine", "BLAST")
                        database = blast_data.get("database", "nr")

                        st.success(
                            f"✅ Found {len(hits)} homologous proteins "
                            f"using **{engine}** in {elapsed:.1f}s"
                        )

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Total Hits", len(hits))
                        with col2:
                            high_identity = len(
                                [h for h in hits if h['identity_percent'] >= 80]
                            )
                            st.metric("High Identity (≥80%)", high_identity)
                        with col3:
                            avg_identity = sum(
                                h['identity_percent'] for h in hits
                            ) / len(hits)
                            st.metric("Avg Identity", f"{avg_identity:.1f}%")
                        with col4:
                            st.metric("Database", database)

                        st.markdown("---")

                        blast_table_html = (
                            ProteinVisualizer
                            .create_blast_results_table_html(hits)
                        )
                        st.components.v1.html(
                            blast_table_html,
                            height=800,
                            scrolling=True
                        )

                        st.markdown("---")

                        col1, col2 = st.columns(2)

                        with col1:
                            blast_df = pd.DataFrame(hits)
                            csv_blast = blast_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Results (CSV)",
                                csv_blast,
                                f"{st.session_state.current_uniprot_id}_homology_results.csv",
                                "text/csv",
                                key="blast_download_csv"
                            )

                        with col2:
                            accessions = "\n".join(
                                f">{h['accession']} {h['organism']}\n"
                                f"# Identity: {h['identity_percent']}%"
                                for h in hits
                            )
                            st.download_button(
                                "📥 Download Accession List",
                                accessions,
                                f"{st.session_state.current_uniprot_id}_accessions.txt",
                                "text/plain",
                                key="blast_download_accessions"
                            )

                        if st.button("🔄 Run New Search", key="blast_run_new"):
                            for key in [
                                'blast_results',
                                'blast_protein_id',
                                'blast_time'
                            ]:
                                st.session_state.pop(key, None)
                            st.rerun()

                    elif blast_data.get('error'):
                        st.error(f"❌ Search failed: {blast_data['error']}")

                    else:
                        st.warning("⚠️ No significant homologs found")

            else:
                st.warning("⚠️ No sequence data available for homology search")

        # Tab 4: EMBL Features & Needle Alignment
        with sequence_tabs[3]:
            st.subheader("EMBL-EBI Sequence Analysis")
            
            sequence = uniprot_data.get('sequence', '')
            
            if sequence:
                # Create sub-tabs
                embl_subtabs = st.tabs(["🗺️ Protein Features", "⚡ Pairwise Alignment (Needle)"])
                
                # Sub-tab 1: Protein Features
                with embl_subtabs[0]:
                    st.markdown("**Protein Domain & Feature Annotations from EMBL-EBI**")
                    
                    # Check if EMBL data exists
                    if 'embl_features' not in st.session_state or \
                    st.session_state.get('embl_protein_id') != st.session_state.current_uniprot_id:
                        
                        with st.spinner("📡 Fetching feature annotations from EMBL-EBI..."):
                            embl_data = cached_fetch_embl_sequence(
                                st.session_state.current_uniprot_id,
                                st.session_state.api_client
                            )
                            
                            st.session_state.embl_features = embl_data
                            st.session_state.embl_protein_id = st.session_state.current_uniprot_id
                    
                    embl_data = st.session_state.embl_features
                    
                    if embl_data.get('available') and embl_data.get('features'):
                        features = embl_data['features']
                        
                        st.success(f"✅ Found {len(features)} annotated features")
                        
                        # Feature statistics
                        col1, col2, col3 = st.columns(3)
                        
                        # Count feature types
                        feature_type_counts = {}
                        for feat in features:
                            ftype = feat.get('type', 'Other')
                            feature_type_counts[ftype] = feature_type_counts.get(ftype, 0) + 1
                        
                        with col1:
                            st.metric("Total Features", len(features))
                        with col2:
                            st.metric("Feature Types", len(feature_type_counts))
                        with col3:
                            # Find longest feature
                            max_length = max([f.get('length', 0) for f in features], default=0)
                            st.metric("Longest Feature", f"{max_length} aa")
                        
                        st.markdown("---")
                        
                        # Feature map visualization
                        fig_features = ProteinVisualizer.create_feature_map(
                            features, 
                            uniprot_data.get('sequence_length', len(sequence))
                        )
                        st.plotly_chart(fig_features, width='stretch')
                        
                        # Detailed feature table
                        st.subheader("📋 Feature Details")
                        
                        feature_df = pd.DataFrame([
                            {
                                "Type": f.get('type', 'Unknown'),
                                "Description": f.get('description', 'N/A'),
                                "Start": f.get('start', 0),
                                "End": f.get('end', 0),
                                "Length": f.get('length', 0)
                            }
                            for f in features
                        ])
                        
                        # Add filter
                        feature_type_filter = st.multiselect(
                            "Filter by feature type:",
                            options=list(feature_type_counts.keys()),
                            default=list(feature_type_counts.keys())
                        )
                        
                        filtered_df = feature_df[feature_df['Type'].isin(feature_type_filter)]
                        st.dataframe(filtered_df, width='stretch', hide_index=True)
                        
                        # Download
                        csv_features = filtered_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Features",
                            csv_features,
                            f"{st.session_state.current_uniprot_id}_features.csv",
                            "text/csv"
                        )
                        
                    else:
                        st.info("ℹ️ No additional feature annotations available from EMBL-EBI")
                
                # Sub-tab 2: Needle Alignment
                with embl_subtabs[1]:
                    st.markdown("**EMBOSS Needle - Global Pairwise Sequence Alignment**")
                    
                    st.info("""
                    **About Needle Alignment:**
                    - Uses Needleman-Wunsch algorithm for global alignment
                    - Compares your protein sequence with another sequence
                    - Shows identity, similarity, gaps, and alignment score
                    - Takes ~10-30 seconds to complete
                    """)
                    
                    # Input for second sequence
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        compare_option = st.radio(
                            "Compare with:",
                            ["Paste sequence", "Use UniProt ID"],
                            horizontal=True
                        )
                    
                    sequence2 = ""
                    seq2_id = "Sequence_2"
                    
                    if compare_option == "Paste sequence":
                        sequence2_input = st.text_area(
                            "Enter second sequence (FASTA or plain text):",
                            height=150,
                            placeholder=">Protein_Name\nMKWVTFISLLFLFSSAYS...\n\nOr paste plain sequence:\nMKWVTFISLLFLFSSAYS..."
                        )
                        
                        if sequence2_input:
                            # Clean and parse input
                            sequence2_input = sequence2_input.strip()
                            
                            # Parse if FASTA format
                            if sequence2_input.startswith('>'):
                                lines = sequence2_input.split('\n')
                                seq2_id = lines[0][1:].strip().split()[0]
                                if not seq2_id:
                                    seq2_id = "Sequence_2"
                                sequence2 = ''.join(lines[1:])
                            else:
                                seq2_id = "Pasted_Sequence"
                                sequence2 = sequence2_input
                            
                            # Remove all whitespace, numbers, and non-letter characters
                            sequence2 = ''.join(c for c in sequence2.upper() if c.isalpha())
                            
                            # Validate sequence
                            valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
                            invalid_chars = set(sequence2) - valid_aa
                            
                            if invalid_chars:
                                st.warning(f"⚠️ Found non-standard amino acids: {', '.join(sorted(invalid_chars))}")
                                st.info("Only standard 20 amino acids will be used for alignment")
                                # Remove invalid characters
                                sequence2 = ''.join(c for c in sequence2 if c in valid_aa)
                            
                            if len(sequence2) < 10:
                                st.error("❌ Sequence too short (minimum 10 amino acids)")
                                sequence2 = ""
                            elif len(sequence2) > 50000:
                                st.error("❌ Sequence too long (maximum 50,000 amino acids)")
                                sequence2 = ""
                            else:
                                st.success(f"✅ Parsed sequence: {len(sequence2)} amino acids (ID: {seq2_id})")
                                st.session_state.compare_sequence = sequence2
                                st.session_state.compare_id = seq2_id
                    else:
                        compare_uniprot = st.text_input(
                            "Enter UniProt ID:",
                            placeholder="e.g., P04637, P38398"
                        )
                        
                        if compare_uniprot and st.button("🔍 Fetch Sequence", key="needle_fetch_sequence"):
                            with st.spinner("Fetching sequence..."):
                                compare_data = cached_fetch_uniprot_data(compare_uniprot, st.session_state.api_client)
                                
                                if compare_data.get('sequence'):
                                    sequence2 = compare_data['sequence']
                                    seq2_id = compare_uniprot
                                    st.success(f"✅ Loaded sequence from {compare_uniprot}")
                                    st.session_state.compare_sequence = sequence2
                                    st.session_state.compare_id = seq2_id
                        
                        # Use stored sequence if available
                        if 'compare_sequence' in st.session_state:
                            sequence2 = st.session_state.compare_sequence
                            seq2_id = st.session_state.compare_id
                    
                    st.markdown("---")
                    
                    # Run alignment
                    if sequence2:
                        run_needle = st.button("⚡ Run Needle Alignment", key="needle_run_alignment", type="primary")
                        
                        if run_needle:
                            with st.spinner("🧬 Running global alignment... This may take 10-30 seconds..."):
                                needle_results = cached_run_needle_alignment(
                                    sequence,
                                    sequence2,
                                    st.session_state.current_uniprot_id,
                                    seq2_id,
                                    st.session_state.api_client
                                )
                                
                                st.session_state.needle_results = needle_results
                                st.rerun()
                    
                    # Display alignment results
                    if 'needle_results' in st.session_state:
                        needle_data = st.session_state.needle_results
                        
                        if needle_data.get('available'):
                            # Show alignment visualization
                            alignment_html = ProteinVisualizer.create_alignment_visualization(needle_data)
                            st.components.v1.html(alignment_html, height=800, scrolling=True)
                            
                            # Interpretation
                            st.subheader("📊 Interpretation")
                            
                            identity = needle_data.get('identity', 0)
                            
                            if identity >= 70:
                                st.success("✅ **High similarity** - Sequences are highly related (likely orthologs or close homologs)")
                            elif identity >= 40:
                                st.warning("⚠️ **Moderate similarity** - Sequences share common ancestry but have diverged")
                            else:
                                st.info("ℹ️ **Low similarity** - Sequences are distantly related or unrelated")
                            
                            # Download alignment
                            st.download_button(
                                "📥 Download Alignment",
                                needle_data.get('alignment_text', ''),
                                f"alignment_{st.session_state.current_uniprot_id}_vs_{seq2_id}.txt",
                                "text/plain"
                            )
                            
                            # Clear results
                            if st.button("🔄 Run New Alignment", key="needle_run_new_alignment"):
                                del st.session_state.needle_results
                                if 'compare_sequence' in st.session_state:
                                    del st.session_state.compare_sequence
                                    del st.session_state.compare_id
                                st.rerun()
                        
                        elif needle_data.get('error'):
                            error_msg = needle_data.get('error')
                            st.error(f"❌ Alignment failed: {error_msg}")
                            
                            # Provide helpful suggestions
                            if "400" in error_msg:
                                st.info("💡 **Tip:** Check that both sequences contain only valid amino acid letters (A-Z).")
                            elif "timed out" in error_msg.lower():
                                st.info("💡 **Tip:** Alignment is taking too long. Try with shorter sequences.")
                            
                            if st.button("🔄 Try Again", key="needle_try_again"):
                                del st.session_state.needle_results
                                st.rerun()                    
                    else:
                        st.info("👆 Enter a second sequence above and click 'Run Needle Alignment'")
            
            else:
                st.warning("⚠️ No sequence data available for EMBL analysis")
                
        st.divider()
        
        # Section 3: 3D Protein Structure
        st.header("🧊 3D Protein Structure")
        
        alphafold_data = data.get('alphafold_structure', {})
        pdb_data = data.get('pdb_structure', {})
        
        # Create tabs for different structure types
        if pdb_data.get('available') and alphafold_data.get('available'):
            structure_tabs = st.tabs(["📊 Experimental (PDB)", "🤖 Predicted (AlphaFold)"])
        elif pdb_data.get('available'):
            structure_tabs = st.tabs(["📊 Experimental (PDB)"])
        elif alphafold_data.get('available'):
            structure_tabs = st.tabs(["🤖 Predicted (AlphaFold)"])
        else:
            st.warning("⚠️ No 3D structure available for this protein")
            structure_tabs = None
        
        if structure_tabs:
            tab_index = 0
            
            # Experimental structure tab
            if pdb_data.get('available'):
                with structure_tabs[tab_index]:
                    st.markdown("**Available Experimental Structures:**")
                    
                    # Show all available PDB structures
                    pdb_structures = pdb_data.get('structures', [])
                    
                    for idx, struct in enumerate(pdb_structures[:5]):  # Show first 5
                        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                        with col1:
                            st.markdown(f"**PDB ID:** [{struct['pdb_id']}]({struct['rcsb_page']})")
                        with col2:
                            st.markdown(f"**Method:** {struct['method']}")
                        with col3:
                            st.markdown(f"**Resolution:** {struct['resolution']}")
                        with col4:
                            if idx == 0:
                                st.markdown("✅ **Displayed below**")
                    
                    if len(pdb_structures) > 5:
                        st.info(f"+ {len(pdb_structures) - 5} more structures available on RCSB PDB")
                    
                    st.markdown("---")
                    
                    # Display 3D viewer for PDB
                    viewer_html = ProteinVisualizer.create_structure_viewer(pdb_data, "pdb")
                    st.components.v1.html(viewer_html, height=600, scrolling=False)
                    
                    # Download option
                    pdb_file_content = None
                    try:
                        pdb_url = pdb_structures[0]['pdb_url']
                        response = requests.get(pdb_url)
                        if response.status_code == 200:
                            pdb_file_content = response.text
                        else:
                            pdb_file_content = f"Could not fetch PDB file. Status code: {response.status_code}"
                    except Exception as e:
                        pdb_file_content = f"Error fetching PDB file: {str(e)}"

                    st.download_button(
                        "📥 Download PDB File",
                        data=pdb_file_content,
                        file_name=f"{pdb_structures[0]['pdb_id']}.pdb",
                        mime="text/plain"
                    )
                
                tab_index += 1
            
            # AlphaFold structure tab
            if alphafold_data.get('available'):
                with structure_tabs[tab_index]:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown(f"""
                        **AlphaFold Database Entry**
                        - **UniProt ID:** {alphafold_data.get('uniprot_id')}
                        - **Gene:** {alphafold_data.get('gene_name', 'N/A')}
                        - **Model Version:** v{alphafold_data.get('model_version', 4)}
                        - **[View on AlphaFold DB]({alphafold_data.get('alphafold_page')})**
                        - **[Download PDB]({alphafold_data.get('pdb_url')})**
                        """)
                    
                    with col2:
                        st.info("""
                        **Confidence Color Code:**
                        - 🔵 **Dark Blue** (>90): Very high confidence
                        - 🔵 **Light Blue** (70-90): Confident
                        - 🟡 **Yellow** (50-70): Low confidence
                        - 🟠 **Orange** (<50): Very low confidence
                        """)
                    
                    st.markdown("---")
                    
                    # Display 3D viewer
                    viewer_html = ProteinVisualizer.create_structure_viewer(alphafold_data, "alphafold")
                    st.components.v1.html(viewer_html, height=600, scrolling=False)
                    
                    # Confidence plot
                    st.subheader("📈 Prediction Confidence")
                    fig_confidence = ProteinVisualizer.create_confidence_plot(
                        st.session_state.current_uniprot_id,
                        alphafold_data.get('entry_id')
                    )
                    st.plotly_chart(fig_confidence, width='stretch')
                                        
                    # Download options
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"[📥 Download PDB File]({alphafold_data.get('pdb_url')})")
                    with col2:
                        st.markdown(f"[📥 Download PAE Data]({alphafold_data.get('pae_url')})")
        
        st.divider()
        
        # Section 4: Tissue Expression
        st.header("🧫 Tissue Expression Analysis")
        
        if not tissue_df.empty:
            # Prepare data
            chart_data = ProteinAPIClient.DataProcessor.prepare_tissue_chart_data(tissue_df, top_n=20)
            
            # Create and display chart
            fig_tissue = ProteinVisualizer.create_tissue_expression_chart(chart_data)
            st.plotly_chart(fig_tissue, width='stretch')
            
            # Expression summary
            high_tissues = tissue_df[tissue_df["level"] == "High"]["tissue"].tolist()
            if high_tissues:
                st.info(f"**High expression detected in:** {', '.join(high_tissues[:5])}" + 
                    (f" and {len(high_tissues)-5} more tissues" if len(high_tissues) > 5 else ""))
        else:
            st.warning("⚠️ No tissue expression data available from Human Protein Atlas")
        
        st.divider()
        
        # Section 5: Subcellular Localization
        st.header("📍 Subcellular Localization")
        
        if not subcellular_df.empty:
            # Create and display heatmap
            fig_subcellular = ProteinVisualizer.create_subcellular_heatmap(subcellular_df)
            st.plotly_chart(fig_subcellular, width='stretch')
            
            # Location list
            st.markdown("**Detected Locations:**")
            for idx, row in subcellular_df.iterrows():
                st.markdown(f"- **{row['location']}** ({row['reliability']} confidence)")
        else:
            st.warning("⚠️ No subcellular localization data available from Human Protein Atlas")

        st.divider()
        
        # Section 6: KEGG Pathways for Proteins
        st.header("🧬 KEGG Pathways for Proteins")
        
        kegg_data = data.get('kegg_pathways', {})
        
        if kegg_data.get('available'):
            # Summary metrics
            total_pathways = kegg_data.get('total_pathways', 0)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin:0; color:#1f77b4;">{total_pathways}</h3>
                        <p style="margin:0; color:#666;">Total Pathways Found</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin:0; color:#1f77b4;">{kegg_data.get('kegg_protein_id', 'N/A')}</h3>
                        <p style="margin:0; color:#666;">KEGG Protein ID</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin:0; color:#1f77b4;">{kegg_data.get('protein_name', 'N/A')}</h3>
                        <p style="margin:0; color:#666;">Protein Name</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Create tabs for different display formats
            pathway_tabs = st.tabs(["🖼️ Primary Pathway Map", "📋 Next 5 Pathways", "🔗 All Pathways Links"])
            
            # Tab 1: First Result with Full Details & Pathway Map
            first_result = kegg_data.get('first_result')
            with pathway_tabs[0]:
                if first_result:
                    st.subheader(f"🏆 Primary Pathway: {first_result.get('pathway_name', 'Unknown')}")
                    
                    # Display all metadata
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown("**Pathway Details:**")
                        st.markdown(f"- **ID:** `{first_result.get('pathway_id', 'N/A')}`")
                        st.markdown(f"- **Name:** {first_result.get('pathway_name', 'N/A')}")
                        
                        if first_result.get('pathway_description'):
                            st.markdown(f"- **Description:** {first_result.get('pathway_description', 'N/A')}")
                        
                        if first_result.get('pathway_class'):
                            st.markdown(f"- **Classification:** {first_result.get('pathway_class', 'N/A')}")
                    
                    with col2:
                        st.markdown("**Molecular Functions:**")
                        functions = first_result.get('molecular_functions', [])
                        if functions:
                            for func in functions[:10]:  # Limit to 10 functions
                                st.markdown(f"• {func}")
                        else:
                            st.markdown("*No specific molecular functions listed*")
                    
                    st.markdown("---")
                    
                    # Display interactive pathway map (with graceful fallback)
                    st.markdown("**Pathway Map (Interactive):**")
                    try:
                        render_kegg_interactive_pathway(
                            first_result,
                            kegg_protein_id=kegg_data.get('kegg_protein_id')
                        )
                    except Exception:
                        # Absolute fallback to original static image in case anything above fails
                        try:
                            st.image(
                                first_result.get('kegg_image_url', ''),
                                width='stretch',
                                caption=f"{first_result.get('pathway_name')} - Visual representation from KEGG"
                            )
                        except Exception:
                            st.warning(
                                f"Could not load pathway map image. "
                                f"[View on KEGG Website]({first_result.get('kegg_url', '#')})"
                            )
                    
                    st.markdown("---")
                    
                    # Direct links
                    col_link1, col_link2 = st.columns(2)
                    with col_link1:
                        st.markdown(f"**[📌 View on KEGG Website]({first_result.get('kegg_url', '#')})**")
                    with col_link2:
                        st.markdown(f"**[🔗 KEGG Gene Entry Page]({first_result.get('highlight_url', '#')})**")
                else:
                    st.info("No primary pathway data available")
            
            # Tab 2: Next 5 Results
            next_results = kegg_data.get('next_results', [])
            with pathway_tabs[1]:
                if next_results:
                    st.subheader("📊 Next 5 Pathways Associated with Protein")
                    
                    for idx, pathway in enumerate(next_results, 1):
                        with st.container():
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                st.markdown(f"**{idx}. {pathway.get('pathway_name', 'Unknown')}**")
                                if pathway.get('pathway_class'):
                                    st.caption(f"Class: {pathway.get('pathway_class', '')}")
                            
                            with col2:
                                st.markdown(f"`{pathway.get('pathway_id', 'N/A')}`")
                            
                            with col3:
                                st.markdown(f"**[View →]({pathway.get('kegg_url', '#')})**")
                            
                            st.divider()
                else:
                    st.info("Less than 6 pathways found for this protein")
            
            # Tab 3: All Pathways Links
            all_pathways = kegg_data.get('pathways', [])
            with pathway_tabs[2]:
                st.subheader(f"🔗 All {len(all_pathways)} Associated Pathways")
                
                # Add filter and sort options
                col1, col2 = st.columns([3, 1])
                with col1:
                    search_term = st.text_input("🔍 Search pathways:", placeholder="e.g., cancer, metabolism, signaling")
                with col2:
                    sort_option = st.selectbox("Sort by:", ["Name", "ID"])
                
                # Filter pathways
                filtered_pathways = all_pathways
                if search_term:
                    search_term = search_term.lower()
                    filtered_pathways = [
                        p for p in all_pathways 
                        if search_term in p.get('pathway_name', '').lower() or 
                           search_term in p.get('pathway_id', '').lower()
                    ]
                
                # Sort pathways
                if sort_option == "Name":
                    filtered_pathways = sorted(filtered_pathways, key=lambda x: x.get('pathway_name', ''))
                elif sort_option == "ID":
                    filtered_pathways = sorted(filtered_pathways, key=lambda x: x.get('pathway_id', ''))
                
                # Display as table
                st.markdown("| # | Pathway Name | ID | KEGG Link |")
                st.markdown("|---|---|---|---|")
                for idx, pathway in enumerate(filtered_pathways, 1):
                    pathway_name = pathway.get('pathway_name', 'Unknown')
                    pathway_id = pathway.get('pathway_id', 'N/A')
                    kegg_url = pathway.get('kegg_url', '#')
                    st.markdown(f"| {idx} | {pathway_name} | `{pathway_id}` | [View Pathway]({kegg_url}) |")
                
                st.caption(f"Showing {len(filtered_pathways)} of {len(all_pathways)} pathways")
            
            # Download pathway data
            st.markdown("---")
            st.subheader("💾 Export Pathway Data")
            
            # Create DataFrame for export
            pathway_df = pd.DataFrame([
                {
                    "Pathway_Name": p['pathway_name'],
                    "Pathway_ID": p['pathway_id'],
                    "Classification": p.get('pathway_class', ''),
                    "Description": p.get('pathway_description', ''),
                    "KEGG_URL": p['kegg_url'],
                    "Highlighted_URL": p['highlight_url']
                }
                for p in all_pathways
            ])
            
            csv_pathways = pathway_df.to_csv(index=False)
            st.download_button(
                "📥 Download Pathway List (CSV)",
                csv_pathways,
                f"{st.session_state.current_uniprot_id}_kegg_pathways.csv",
                "text/csv"
            )
            
        else:
            st.warning(f"⚠️ No KEGG pathway data found for gene: {kegg_data.get('gene_name', 'Unknown')}")
            st.info("""
            **Why might this happen?**
            - Gene name not recognized in KEGG database
            - Protein not associated with metabolic/signaling pathways
            - Limited annotation in KEGG for this specific protein
            
            Try searching directly on [KEGG website](https://www.kegg.jp/)
            """)
        
        st.divider()
        
        # Section: STRING Protein-Protein Interactions
        st.header("🔗 Protein-Protein Interaction Network (STRING)")
        
        string_data = data.get('string_ppi', {})
        
        st.info("""
        **About STRING Database:**
        - Comprehensive protein-protein interaction database
        - Combines experimental data, computational prediction, and text mining
        - Confidence scores from 0-1000 (higher = more reliable)
        """)
        if string_data.get('available') and string_data.get('interactions'):
            interactions = string_data['interactions']
            gene_name = string_data.get('gene_name', st.session_state.current_uniprot_id)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Interactions", len(interactions))
            with col2:
                high_conf = [i for i in interactions if i['combined_score'] >= 700]
                st.metric("High Confidence (≥700)", len(high_conf))
            with col3:
                st.metric("STRING Protein ID", string_data.get('string_id', 'N/A'))
            
            # Additional confidence metrics
            col4, col5, col6 = st.columns(3)
            
            with col4:
                highest_conf = [i for i in interactions if i['combined_score'] >= 900]
                st.metric("Highest Confidence (≥900)", len(highest_conf))
            with col5:
                medium_conf = [i for i in interactions if i['combined_score'] >= 400 and i['combined_score'] < 700]
                st.metric("Medium Confidence (≥400)", len(medium_conf))
            with col6:
                low_conf = [i for i in interactions if i['combined_score'] < 400]
                st.metric("Low Confidence (<400)", len(low_conf))
            
            st.markdown("---")
            
            # Create tabs
            ppi_tabs = st.tabs(["🕸️ Network Graph", "📋 Interaction Table"])
            
            # Tab 1: Network visualization
            with ppi_tabs[0]:
                network_fig = ProteinVisualizer.create_ppi_network_chart(interactions, gene_name)
                st.plotly_chart(network_fig, width='stretch')
                
                st.caption("""
                **Color Legend:**
                
                🔴 Red = Query protein | 🔵 Dark Blue = Highest confidence (≥900) | 🟢 Green = High confidence (≥700) | 🟠 Orange = Medium confidence (≥400) | ⚪ Gray = Low confidence (<400)
                """)
            
            # Tab 2: Interaction table
            with ppi_tabs[1]:
                st.subheader("Protein Interaction Partners")
                
                # Display interactions in a table
                ppi_table_html = ProteinVisualizer.create_ppi_table_html(interactions)
                st.components.v1.html(ppi_table_html, height=600, scrolling=True)
            
            st.markdown("---")
            
            # External links
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**[🔗 View on STRING Database]({string_data.get('string_url', '#')})**")
            with col2:
                st.markdown(f"**[📊 Network Image]({string_data.get('network_image_url', '#')})**")
            
            st.markdown("---")
            
            # Download interaction data
            interaction_df = pd.DataFrame(interactions)
            csv_interactions = interaction_df.to_csv(index=False)
            st.download_button(
                "📥 Download Interaction Data",
                csv_interactions,
                f"{st.session_state.current_uniprot_id}_string_interactions.csv",
                "text/csv"
            )
        else:
            st.warning(f"⚠️ No STRING interaction data found for {st.session_state.current_uniprot_id}")
            error_msg = string_data.get('error', 'Unknown error')
            st.info(f"""
            **Possible reasons:**
            - Protein not found in STRING database (Gene: {string_data.get('gene_name', 'Unknown')})
            - Limited experimental or predicted interaction data
            - Protein may have few known interactors
            
            **Error:** {error_msg}
            """)

        st.divider()

        # Section 7: Molecular Docking with AutoDock Vina
        st.header("💊 Molecular Docking Analysis")
        
        chembl_data = data.get('chembl_ligands', {})
        
        st.info("""
        **About Molecular Docking:**
        - Predicts how small molecules (ligands/drugs) bind to proteins
        - Uses AutoDock Vina algorithm for binding affinity calculation
        - Negative values indicate favorable binding (more negative = stronger binding)
        - Typical drug-like binding: -7 to -12 kcal/mol
        - 3D visualization of ligand orientation and binding prediction
        """)
        
        # Create tabs
        docking_tabs = st.tabs(["📚 Known Ligands", "🎯 Binding Predictor", "🔮 Ligand Binding Prediction", "🧪 Custom Docking", "📊 Docking Results"])
        
        # Tab 1: Known Ligands from ChEMBL
        with docking_tabs[0]:
            st.subheader("Known Inhibitors & Ligands from ChEMBL")
            
            if chembl_data.get('available') and chembl_data.get('ligands'):
                ligands = chembl_data['ligands']
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Ligands", len(ligands))
                with col2:
                    strong_binders = [l for l in ligands if l.get('activity_value', float('inf')) < 100]
                    st.metric("Strong Binders (<100nM)", len(strong_binders))
                with col3:
                    st.metric("ChEMBL Target", chembl_data.get('chembl_target_id', 'N/A'))
                
                st.markdown("---")
                
                # Display ligand cards with Dock buttons
                st.info("💡 **Tip:** Click the 'Dock' button next to any ligand to run molecular docking simulation")
                
                # Show success message if docking was just completed
                if st.session_state.get('show_docking_success'):
                    st.success(f"✅ Docking complete for {st.session_state.get('docked_ligand_name')}! 📊 Go to **Docking Results** tab to view results.")
                    st.session_state.show_docking_success = False
                
                for idx, ligand in enumerate(ligands[:20]):  # Show top 20
                    with st.expander(f"🧪 {ligand.get('name', ligand.get('chembl_id'))} - {ligand.get('activity_value', 'N/A')} {ligand.get('activity_units', 'nM')}"):
                        col_img, col_info, col_action = st.columns([1, 2, 1])
                        
                        with col_img:
                            # Structure image
                            img_url = f"https://www.ebi.ac.uk/chembl/api/data/image/{ligand.get('chembl_id')}.svg"
                            st.image(img_url, width=150)
                        
                        with col_info:
                            st.markdown(f"**ChEMBL ID:** [{ligand.get('chembl_id')}]({ligand.get('chembl_url', '#')})")
                            st.markdown(f"**Activity:** {ligand.get('activity_type', 'N/A')}")
                            st.markdown(f"**Value:** {ligand.get('activity_value', 'N/A')} {ligand.get('activity_units', 'nM')}")
                            mw = ligand.get('molecular_weight')
                            if mw and mw != 'N/A':
                                st.markdown(f"**MW:** {float(mw):.1f} Da")
                        
                        with col_action:
                            if st.button(f"🎯 Dock", key=f"dock_ligand_{idx}"):
                                # Store ligand for docking
                                st.session_state.selected_ligand_for_docking = {
                                    'chembl_id': ligand.get('chembl_id'),
                                    'name': ligand.get('name', ligand.get('chembl_id')),
                                    'smiles': ligand.get('smiles', ''),
                                    'mw': ligand.get('molecular_weight', 0),
                                    'activity_value': ligand.get('activity_value', None)
                                }
                                
                                # Run docking simulation
                                docking_result = st.session_state.api_client.simulate_docking_score(
                                    protein_length=uniprot_data.get('sequence_length', 500),
                                    ligand_mw=ligand.get('molecular_weight', 0),
                                    activity_value=ligand.get('activity_value', None)
                                )
                                
                                # Store results and ligand data for display
                                st.session_state.docking_results = docking_result
                                st.session_state.docked_ligand_name = ligand.get('name', ligand.get('chembl_id'))
                                st.session_state.docked_ligand_data = {
                                    'chembl_id': ligand.get('chembl_id'),
                                    'name': ligand.get('name', ligand.get('chembl_id')),
                                    'smiles': ligand.get('smiles', ''),
                                    'molecular_weight': ligand.get('molecular_weight', 0)
                                }
                                
                                # Store protein structure data (from AlphaFold or PDB if available)
                                protein_struct = data.get('alphafold_structure', {})
                                if not protein_struct.get('available'):
                                    protein_struct = data.get('pdb_structure', {})
                                st.session_state.protein_structure = protein_struct
                                
                                st.session_state.show_docking_success = True
                                st.rerun()

                st.markdown("---")
                
                # Download ligand data
                ligand_df = pd.DataFrame([
                    {
                        "ChEMBL_ID": str(l['chembl_id']),
                        "Name": str(l['name']),
                        "SMILES": str(l.get('smiles', '')),
                        "Activity_Type": str(l['activity_type']),
                        "Activity_Value": str(l['activity_value']) if l['activity_value'] is not None else 'N/A',
                        "Units": str(l['activity_units']),
                        "Molecular_Weight": str(l.get('molecular_weight', 'N/A'))
                    }
                    for l in ligands
                ])
                
                csv_ligands = ligand_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Ligand Data",
                    csv_ligands,
                    f"{st.session_state.current_uniprot_id}_ligands.csv",
                    "text/csv",
                    key="download_ligands"
                )
                
            else:
                st.warning(f"⚠️ No known ligands found in ChEMBL for {st.session_state.current_uniprot_id}")
                st.info("""
                **Possible reasons:**
                - Protein not yet studied as drug target
                - No bioactivity data available in ChEMBL
                - Protein may not be druggable
                
                You can still try custom docking in the next tab!
                """)
        
        # Tab 2: Binding Predictor
        with docking_tabs[1]:
            st.subheader("🎯 AI-Powered Binding Predictor & Drug Discovery")
            
            st.markdown("""
            **Comprehensive Binding Analysis:**
            - ✅ Predict binding for **known ligands** (from ChEMBL)
            - 🧬 Find **similar compounds** with binding potential
            - 📊 ML-based scoring with confidence levels
            """)
            
            # Create sub-tabs
            predictor_subtabs = st.tabs([
                "🏆 Known Ligands Analysis",
                "🧪 Similar Compounds",
                "📋 Comprehensive Report"
            ])
            
            # Sub-tab 1: Known Ligands Prediction
            with predictor_subtabs[0]:
                st.markdown("### Known Ligands Binding Prediction")
                
                if chembl_data.get('available') and chembl_data.get('ligands'):
                    ligands = chembl_data['ligands']
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.info(f"📊 Analyzing {len(ligands)} known ligands from ChEMBL")
                    with col2:
                        if st.button("🔮 Predict Binding", type="primary", key="predict_known"):
                            with st.spinner("🧠 Running ML-based binding prediction..."):
                                prediction = ProteinVisualizer.advanced_binding_prediction(
                                    ligands,
                                    uniprot_data,
                                    novel_compounds=None
                                )
                                st.session_state.binding_prediction = prediction
                                st.rerun()
                    
                    # Display results
                    if 'binding_prediction' in st.session_state:
                        pred = st.session_state.binding_prediction
                        known_preds = pred.get('known_ligands', [])
                        
                        if known_preds:
                            # Highlight top 3
                            st.success("✅ **Prediction Complete - Top 3 Predicted Binders:**")
                            
                            for idx, p in enumerate(known_preds[:3], 1):
                                comp = p['compound']
                                
                                with st.expander(
                                    f"#{idx} {comp['name']} - Score: {p['predicted_score']}/100 "
                                    f"({p['confidence_level']} confidence)",
                                    expanded=(idx == 1)
                                ):
                                    col1, col2, col3 = st.columns([2, 1, 1])
                                    
                                    with col1:
                                        st.markdown(f"""
                                        **Compound Details:**
                                        - **ChEMBL ID:** [{comp['chembl_id']}]({comp.get('chembl_url', '#')})
                                        - **Activity:** {comp['activity_value']:.2f} {comp['activity_units']} ({comp['activity_type']})
                                        - **Molecular Weight:** {comp.get('molecular_weight', 'N/A')} Da
                                        - **Predicted Affinity:** {p['predicted_affinity']:.2f} kcal/mol
                                        """)
                                        
                                        # Show structure
                                        img_url = f"https://www.ebi.ac.uk/chembl/api/data/image/{comp['chembl_id']}.svg"
                                        st.image(img_url, caption=comp['name'], width=200)
                                    
                                    with col2:
                                        st.markdown(f"""
                                        **Prediction Metrics:**
                                        
                                        Score: **{p['predicted_score']}/100**
                                        
                                        Confidence: **{p['confidence_level']}** ({p['confidence']:.0%})
                                        
                                        <div style="background-color:{p['confidence_color']}; color:white; padding:10px; border-radius:5px; text-align:center; margin-top:10px;">
                                            <strong>{p['recommendation']}</strong>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with col3:
                                        st.markdown("**✅ Positive Factors:**")
                                        for reason in p['reasons'][:5]:
                                            st.caption(f"• {reason}")
                                        
                                        if p['warnings']:
                                            st.markdown("**⚠️ Warnings:**")
                                            for warning in p['warnings']:
                                                st.caption(f"• {warning}")
                                    
                                    # Action buttons
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button(f"🚀 Dock This Compound", key=f"dock_known_{idx}"):
                                            st.session_state.selected_ligand = comp
                                            st.session_state.selected_ligand_name = comp['name']
                                            st.info(f"✅ Selected {comp['name']} - Go to 'Custom Docking' tab")
                                    with col2:
                                        if st.button(f"🔍 Find Similar", key=f"similar_known_{idx}"):
                                            st.session_state.reference_smiles = comp.get('smiles')
                                            st.session_state.reference_name = comp.get('name')
                                            st.session_state.similar_auto_run = True
                                            st.session_state.similar_similarity = 0.7
                                            st.rerun()
                            
                            # Show full ranking table
                            st.markdown("---")
                            st.subheader("📊 Complete Ranking")
                            
                            ranking_data = []
                            for idx, p in enumerate(known_preds, 1):
                                ranking_data.append({
                                    "Rank": idx,
                                    "Compound": p['compound']['name'],
                                    "Predicted Score": f"{p['predicted_score']}/100",
                                    "Confidence": p['confidence_level'],
                                    "Predicted Affinity": f"{p['predicted_affinity']:.2f} kcal/mol",
                                    "Experimental Activity": f"{p['compound']['activity_value']:.2f} {p['compound']['activity_units']}",
                                    "Recommendation": p['recommendation']
                                })
                            
                            ranking_df = pd.DataFrame(ranking_data)
                            st.dataframe(ranking_df, width='stretch', hide_index=True)
                            
                            # Download
                            csv_ranking = ranking_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Prediction Results",
                                csv_ranking,
                                f"{st.session_state.current_uniprot_id}_binding_predictions.csv",
                                "text/csv",
                                key="download_predictions"
                            )
                
                else:
                    st.warning("⚠️ No known ligands available for prediction")
            
            # Sub-tab 2: Similar Compounds
            with predictor_subtabs[1]:
                st.markdown("### Find Structurally Similar Compounds")
                
                st.info("""
                Search for compounds chemically similar to a reference ligand.
                Similar structures often have similar biological activity.
                """)
                
                # Check if triggered from 'Find Similar' button
                auto_run = st.session_state.pop('similar_auto_run', False)
                preloaded_smiles = st.session_state.pop('reference_smiles', None)
                preloaded_name = st.session_state.pop('reference_name', None)
                default_similarity = st.session_state.pop('similar_similarity', 0.7)
                
                # Reference selection
                reference_source = st.radio(
                    "Select reference compound:",
                    ["From known ligands", "Enter SMILES manually"],
                    key="similar_source"
                )
                
                reference_smiles = None
                reference_name = None
                
                if reference_source == "From known ligands":
                    if chembl_data.get('available') and chembl_data.get('ligands'):
                        ligand_options = {
                            f"{l['name']} ({l['chembl_id']})": l
                            for l in chembl_data['ligands'][:20]
                        }
                        
                        # Preselect if coming from 'Find Similar'
                        preselect_idx = 0
                        if preloaded_name:
                            for i, k in enumerate(ligand_options.keys()):
                                if preloaded_name in k:
                                    preselect_idx = i
                                    break
                        
                        selected = st.selectbox(
                            "Choose reference ligand:",
                            list(ligand_options.keys()),
                            index=preselect_idx,
                            key="similar_ref_select"
                        )
                        
                        ref_lig = ligand_options[selected]
                        reference_smiles = ref_lig.get('smiles')
                        reference_name = ref_lig['name']
                    else:
                        st.warning("No known ligands available")
                
                else:
                    reference_smiles = st.text_input(
                        "Enter SMILES:",
                        value=preloaded_smiles if preloaded_smiles and reference_source == "Enter SMILES manually" else "",
                        placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O",
                        key="similar_smiles_input"
                    )
                    reference_name = preloaded_name or "Custom SMILES"
                
                # Similarity threshold
                similarity = st.slider(
                    "Similarity threshold:",
                    0.5, 1.0, default_similarity if auto_run else 0.7, 0.05,
                    help="Higher = more similar (0.7 = 70% similar)",
                    key="similarity_threshold"
                )
                
                # Auto-run if triggered from 'Find Similar' button
                if auto_run and reference_smiles and preloaded_smiles:
                    with st.spinner(f"Searching for compounds ≥{similarity*100:.0f}% similar to {reference_name}..."):
                        similar_data = cached_fetch_similar_compounds(
                            reference_smiles,
                            similarity,
                            st.session_state.api_client
                        )
                        
                        if similar_data.get('available'):
                            # Run predictions on similar compounds
                            known_ligands = chembl_data.get('ligands', []) if chembl_data.get('available') else []
                            
                            prediction = ProteinVisualizer.advanced_binding_prediction(
                                known_ligands,
                                uniprot_data,
                                novel_compounds=similar_data.get('compounds', [])
                            )
                            
                            st.session_state.similar_prediction = prediction
                            st.session_state.similar_data = similar_data
                
                if reference_smiles:
                    if st.button("🔍 Find Similar Compounds", type="primary", key="find_similar"):
                        with st.spinner(f"Searching for compounds ≥{similarity*100:.0f}% similar to {reference_name}..."):
                            similar_data = cached_fetch_similar_compounds(
                                reference_smiles,
                                similarity,
                                st.session_state.api_client
                            )
                            
                            if similar_data.get('available'):
                                # Run predictions on similar compounds
                                known_ligands = chembl_data.get('ligands', []) if chembl_data.get('available') else []
                                
                                prediction = ProteinVisualizer.advanced_binding_prediction(
                                    known_ligands,
                                    uniprot_data,
                                    novel_compounds=similar_data.get('compounds', [])
                                )
                                
                                st.session_state.similar_prediction = prediction
                                st.session_state.similar_data = similar_data
                                st.rerun()
                
                # Display similar compounds
                if 'similar_prediction' in st.session_state:
                    pred = st.session_state.similar_prediction
                    similar_preds = pred.get('novel_candidates', [])
                    
                    if similar_preds:
                        st.success(f"✅ Found {len(similar_preds)} similar compounds")
                        
                        for idx, p in enumerate(similar_preds[:10], 1):
                            comp = p['compound']
                            
                            with st.expander(f"{idx}. {comp['name'][:50]}"):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.markdown(f"""
                                    **Compound:** {comp['name']}  
                                    **PubChem CID:** [{comp['cid']}]({comp['pubchem_url']})  
                                    **Formula:** {comp.get('formula', 'N/A')}  
                                    **MW:** {comp.get('molecular_weight', 'N/A')} Da  
                                    **SMILES:** `{comp.get('smiles', 'N/A')[:50]}...`
                                    
                                    **Predicted Affinity:** {p['predicted_affinity']:.2f} kcal/mol
                                    """)
                                    
                                    # Show reasons
                                    st.markdown("**Why this compound:**")
                                    for reason in p['reasons'][:3]:
                                        st.caption(f"• {reason}")
                                
                                with col2:
                                    # PubChem image
                                    img_url = f"https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid={comp['cid']}&t=l"
                                    st.image(img_url, caption=f"CID {comp['cid']}", width=150)
                                    
                                    if st.button(f"🚀 Dock", key=f"dock_similar_{idx}"):
                                        st.session_state.selected_ligand = comp
                                        st.session_state.selected_ligand_name = comp['name']
                                        st.info("Go to 'Custom Docking' tab")
                    else:
                        st.info("No similar compounds found. Try lowering the similarity threshold.")
            
            # Sub-tab 3: Comprehensive Report
            with predictor_subtabs[2]:
                st.markdown("### 📋 Comprehensive Binding Analysis Report")
                
                if 'binding_prediction' in st.session_state:
                    pred = st.session_state.binding_prediction
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Known Ligands", len(pred.get('known_ligands', [])))
                    with col2:
                        similar_count = len(st.session_state.get('similar_prediction', {}).get('novel_candidates', []))
                        st.metric("Similar Compounds", similar_count)
                    with col3:
                        total = len(pred.get('known_ligands', [])) + similar_count
                        st.metric("Total Analyzed", total)
                    
                    st.markdown("---")
                    
                    # Binding rules extracted
                    st.subheader("🧬 Extracted Binding Rules (SAR)")
                    rules = pred.get('binding_rules', {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if rules.get('optimal_mw_range'):
                            st.info(f"""
                            **Optimal Molecular Weight:**  
                            {rules['optimal_mw_range'][0]:.0f} - {rules['optimal_mw_range'][1]:.0f} Da  
                            *(Based on active compounds)*
                            """)
                    
                    with col2:
                        if rules.get('activity_threshold'):
                            thresh = rules['activity_threshold']
                            st.info(f"""
                            **Activity Thresholds:**  
                            Potent: <{thresh.get('potent', 0):.1f} nM  
                            Moderate: <{thresh.get('moderate', 0):.1f} nM  
                            Weak: >{thresh.get('moderate', 0):.1f} nM
                            """)
                    
                    # Recommendations
                    st.markdown("---")
                    st.subheader("💡 Actionable Recommendations")
                    
                    recommendations = pred.get('recommendations', [])
                    if recommendations:
                        for rec in recommendations:
                            priority_color = {
                                "High": "🔴",
                                "Medium": "🟡",
                                "Low": "🟢"
                            }.get(rec.get('priority', 'Medium'), "⚪")
                            
                            st.markdown(f"""
                            {priority_color} **{rec['type']}:** {rec['compound']}  
                            *Action:* {rec['action']}  
                            *Priority:* {rec['priority']}
                            """)
                    
                    # Download full report
                    st.markdown("---")
                    
                    # Generate comprehensive report
                    report_text = generate_full_report(pred, uniprot_data)
                    
                    st.download_button(
                        "📥 Download Full Report (TXT)",
                        report_text,
                        f"{st.session_state.current_uniprot_id}_binding_report.txt",
                        "text/plain",
                        key="download_full_report"
                    )
                
                else:
                    st.info("Run predictions in other tabs to generate comprehensive report")

        # Tab 3: Custom Docking
        with docking_tabs[3]:
            st.subheader("Run Custom Molecular Docking")
            
            # Check if protein structure is available
            protein_prep = st.session_state.api_client.prepare_protein_for_docking(
                uniprot_data,
                data.get('pdb_structure', {}),
                data.get('alphafold_structure', {})
            )
            
            if not protein_prep.get('available'):
                st.error("❌ No protein structure available for docking. Please ensure 3D structure is loaded.")
            else:
                st.success(f"✅ Using {protein_prep['structure_type']} structure: {protein_prep['structure_id']}")

                # Ligand input options
                st.markdown("**Select Ligand Source:**")

                ligand_source = st.radio(
                    "Choose ligand source",
                    ["Use predicted best binder", "Known ligand from ChEMBL", "Custom compound (PubChem)", "Upload SMILES/SDF"],
                    horizontal=False,
                    key="ligand_source_radio",
                    label_visibility="collapsed"
                )
                
                selected_ligand = None
                ligand_name = None
                
                if ligand_source == "Use predicted best binder":
                    # Offer any previously selected ligand from other tabs
                    prev_candidates = []
                    if 'selected_ligand' in st.session_state:
                        prev_candidates.append(("Binding Predictor", st.session_state.selected_ligand, st.session_state.get('selected_ligand_name')))
                    if 'selected_ligand_for_docking' in st.session_state:
                        prev_candidates.append(("Known Ligand", st.session_state.selected_ligand_for_docking, st.session_state.selected_ligand_for_docking.get('name')))
                    if 'docked_ligand_data' in st.session_state:
                        prev_candidates.append(("Last Docked", st.session_state.docked_ligand_data, st.session_state.get('docked_ligand_name')))

                    if prev_candidates:
                        options = [f"{src}: {name}" for src, _, name in prev_candidates]
                        sel = st.selectbox("Use previously selected ligand:", ["(none)"] + options, key="use_prev_selected_ligand_select")
                        if sel and sel != "(none)":
                            idx = options.index(sel)
                            selected_ligand = prev_candidates[idx][1]
                            ligand_name = prev_candidates[idx][2]
                            st.info(f"✅ Using: **{ligand_name}**")
                    else:
                        # Fallback to binding predictor best binder
                        if 'binding_prediction' in st.session_state:
                            pred = st.session_state.binding_prediction
                            if pred.get('available'):
                                selected_ligand = pred['best_ligand']
                                ligand_name = selected_ligand['name']
                                st.info(f"✅ Using predicted best binder: **{ligand_name}**")
                            else:
                                st.warning("⚠️ No prediction available. Run predictor first.")
                        else:
                            st.warning("⚠️ Please run the Binding Predictor first (previous tab)")
                
                elif ligand_source == "Known ligand from ChEMBL":
                    if chembl_data.get('available') and chembl_data.get('ligands'):
                        ligand_options = {
                            f"{l['name']} ({l['chembl_id']}) - {l['activity_type']}: {l['activity_value']:.1f} {l['activity_units']}": l
                            for l in chembl_data['ligands'][:10]
                        }
                        
                        selected_option = st.selectbox("Choose ligand:", list(ligand_options.keys()), key="chembl_select")
                        selected_ligand = ligand_options[selected_option]
                        ligand_name = selected_ligand['name']
                    else:
                        st.warning("No ChEMBL ligands available")
                
                elif ligand_source == "Custom compound (PubChem)":
                    compound_name = st.text_input(
                        "Enter compound name:",
                        placeholder="e.g., Aspirin, Ibuprofen, Caffeine",
                        key="pubchem_input"
                    )
                    
                    if compound_name and st.button("🔍 Search PubChem", key="pubchem_search"):
                        with st.spinner("Searching PubChem..."):
                            pubchem_data = cached_fetch_pubchem_structure(compound_name, st.session_state.api_client)
                            
                            if pubchem_data.get('available'):
                                st.success(f"✅ Found: {compound_name} (CID: {pubchem_data['cid']})")
                                st.image(pubchem_data['image_url'], width=200)
                                st.session_state.custom_ligand = pubchem_data
                                selected_ligand = pubchem_data
                                ligand_name = compound_name
                            else:
                                st.error(f"❌ Compound '{compound_name}' not found in PubChem")
                    
                    if 'custom_ligand' in st.session_state:
                        selected_ligand = st.session_state.custom_ligand
                        ligand_name = compound_name
                
                else:  # Upload SMILES/SDF
                    smiles_input = st.text_input(
                        "Enter SMILES string:",
                        placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O (Aspirin)",
                        key="smiles_input"
                    )
                    
                    if smiles_input:
                        ligand_name = "Custom_SMILES"
                        selected_ligand = {"smiles": smiles_input, "name": ligand_name}
                
                st.markdown("---")
                
                # Docking parameters
                st.markdown("**Docking Parameters:**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    exhaustiveness = st.slider("Exhaustiveness", 1, 16, 8, 
                                              help="Higher = more thorough but slower",
                                              key="exhaustiveness_slider")
                with col2:
                    num_modes = st.slider("Number of modes", 1, 20, 9,
                                         help="Number of binding poses to generate",
                                         key="num_modes_slider")
                with col3:
                    energy_range = st.slider("Energy range (kcal/mol)", 1, 5, 3,
                                            key="energy_range_slider")
                
                # Run docking button
                if selected_ligand:
                    run_docking = st.button("🚀 Run Molecular Docking", type="primary", key="run_docking_btn")
                    
                    if run_docking:
                        with st.spinner("🧬 Running AutoDock Vina simulation... Calculating 3D orientation..."):
                            time.sleep(2)
                            
                            docking_results = st.session_state.api_client.simulate_docking_score(
                                protein_prep['sequence_length'],
                                selected_ligand.get('molecular_weight', 300),
                                selected_ligand.get('activity_value'),
                                selected_ligand.get('smiles')
                            )
                            
                            st.session_state.docking_results = docking_results
                            st.session_state.docked_ligand_name = ligand_name
                            st.session_state.docked_ligand_data = selected_ligand
                            st.session_state.protein_structure = protein_prep
                            st.rerun()
                else:
                    st.info("👆 Please select or enter a ligand above")
        
        # Tab 2: Ligand Binding Prediction - Using Advanced Docking Interface
        with docking_tabs[2]:
            st.subheader("🔮 Ligand Binding Prediction & Docking")
            
            st.markdown("""
            **Advanced ligand binding analysis:**
            - Predict binding affinity for any ligand SMILES
            - Run molecular docking simulations
            - View 3D protein-ligand complexes
            - Generate binding predictions with confidence scores
            """)
            
            # Ligand input: Single SMILES or compound search
            st.markdown("#### Ligand Input")
            
            input_method = st.radio(
                "Select input method:",
                ["Enter SMILES", "Search PubChem", "Previous ligands"],
                horizontal=True,
                key="ligand_binding_input_method"
            )
            
            selected_ligand = None
            ligand_name = None
            
            if input_method == "Enter SMILES":
                smiles_input = st.text_input(
                    "SMILES String:",
                    placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O",
                    key="ligand_binding_smiles"
                )
                if smiles_input:
                    ligand_name = "Custom_SMILES"
                    selected_ligand = {"smiles": smiles_input, "name": ligand_name, "molecular_weight": 200}
            
            elif input_method == "Search PubChem":
                compound_name = st.text_input(
                    "Compound Name:",
                    placeholder="e.g., Aspirin",
                    key="ligand_binding_compound"
                )
                if compound_name and st.button("🔍 Search", key="ligand_binding_search"):
                    pubchem_data = cached_fetch_pubchem_structure(compound_name, st.session_state.api_client)
                    if pubchem_data.get('available'):
                        st.success(f"✅ Found: {compound_name}")
                        st.image(pubchem_data['image_url'], width=200)
                        st.session_state.ligand_binding_compound_data = pubchem_data
                        selected_ligand = pubchem_data
                        ligand_name = compound_name
            
            elif input_method == "Previous ligands":
                if 'docked_ligand_data' in st.session_state:
                    prev_ligand = st.session_state.docked_ligand_data
                    st.info(f"Using: {prev_ligand.get('name', 'Unknown')}")
                    selected_ligand = prev_ligand
                    ligand_name = prev_ligand.get('name', 'Unknown')
                else:
                    st.info("No previously docked ligands available")
            
            # If we have a ligand from session state
            if 'ligand_binding_compound_data' in st.session_state and input_method == "Search PubChem":
                selected_ligand = st.session_state.ligand_binding_compound_data
                ligand_name = st.session_state.get('ligand_binding_compound_name', 'Unknown')
            
            if selected_ligand:
                st.divider()
                
                # Docking parameters
                st.markdown("#### Docking Configuration")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    exhaustiveness = st.slider("Exhaustiveness", 1, 16, 8, key="ligand_binding_exhaustiveness")
                with col2:
                    num_modes = st.slider("Binding Modes", 1, 20, 9, key="ligand_binding_modes")
                with col3:
                    energy_range = st.slider("Energy Range", 1, 5, 3, key="ligand_binding_energy")
                
                st.divider()
                
                # Run docking
                if st.button("🚀 Predict & Dock", type="primary", width='stretch', key="ligand_binding_dock"):
                    with st.spinner("Running binding prediction and docking..."):
                        time.sleep(1)
                        
                        docking_results = st.session_state.api_client.simulate_docking_score(
                            uniprot_data.get('sequence_length', 500),
                            selected_ligand.get('molecular_weight', 300),
                            None,
                            selected_ligand.get('smiles')
                        )
                        
                        st.session_state.ligand_binding_results = docking_results
                        st.session_state.ligand_binding_ligand_name = ligand_name
                        st.session_state.ligand_binding_ligand_data = selected_ligand
                        
                        # Get protein structure
                        protein_struct = data.get('alphafold_structure', {})
                        if not protein_struct.get('available'):
                            protein_struct = data.get('pdb_structure', {})
                        st.session_state.ligand_binding_protein = protein_struct
                        
                        st.rerun()
                
                # Display results
                if 'ligand_binding_results' in st.session_state:
                    results = st.session_state.ligand_binding_results
                    best_affinity = results['binding_affinity']
                    
                    st.divider()
                    st.subheader("📊 Binding Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if best_affinity < -7:
                            color = "#28a745"
                            strength = "Strong"
                        elif best_affinity < -5:
                            color = "#ffc107"
                            strength = "Moderate"
                        else:
                            color = "#dc3545"
                            strength = "Weak"
                        
                        st.markdown(f"""
                            <div style="background-color:{color}; color:white; padding:20px; border-radius:8px; text-align:center;">
                                <h2 style="margin:0;">{best_affinity} kcal/mol</h2>
                                <p style="margin:5px 0 0 0;">Binding Affinity</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Strength", strength)
                    with col3:
                        st.metric("Modes", len(results.get('modes', [])))
                    
                    st.markdown("---")
                    
                    # 3D Visualization
                    st.subheader("🔬 3D Complex")
                    protein_structure = st.session_state.get('ligand_binding_protein', {})
                    ligand_data = st.session_state.get('ligand_binding_ligand_data', {})
                    
                    if results.get('has_coordinates'):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            viewer_html = ProteinVisualizer.create_docking_3d_viewer(
                                protein_structure,
                                ligand_data,
                                results,
                                st.session_state.get('ligand_binding_ligand_name', 'Unknown')
                            )
                            st.components.v1.html(viewer_html, height=650, scrolling=False)
                        
                        with col2:
                            st.info("""
                            **3D Controls:**
                            - Left: Rotate
                            - Right: Zoom
                            - Middle: Pan
                            """)
                    
                    st.markdown("---")
                    
                    # Results chart
                    fig = ProteinVisualizer.create_docking_results_chart(results)
                    st.plotly_chart(fig, width='stretch')
                    
                    # Download results
                    st.markdown("---")
                    results_csv = pd.DataFrame([{
                        "Ligand": st.session_state.get('ligand_binding_ligand_name', 'Unknown'),
                        "Affinity_kcal_mol": best_affinity,
                        "Modes": len(results.get('modes', [])),
                        "Timestamp": datetime.now().isoformat()
                    }]).to_csv(index=False)
                    
                    st.download_button(
                        "📥 Download Results",
                        results_csv,
                        f"ligand_binding_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        key="ligand_binding_download"
                    )
            else:
                st.info("👆 Select or enter a ligand above to begin")
        
        # Tab 4: Docking Results
        with docking_tabs[4]:
            st.subheader("Docking Results")
            if 'docking_results' in st.session_state:
                results = st.session_state.docking_results
                ligand_name = st.session_state.get('docked_ligand_name', 'Unknown')
                ligand_data = st.session_state.get('docked_ligand_data', {})
                protein_structure = st.session_state.get('protein_structure', {})
                
                if results.get('simulated'):
                    st.warning("⚠️ **Note:** These are simulated results for demonstration. Production version would use actual AutoDock Vina calculations.")
                
                st.markdown(f"### Results for: **{ligand_name}**")
                
                # Best binding affinity
                best_affinity = results['binding_affinity']
                best_mode = results.get('best_mode', {})
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if best_affinity < -7:
                        color = "#28a745"
                        strength = "Strong"
                    elif best_affinity < -5:
                        color = "#ffc107"
                        strength = "Moderate"
                    else:
                        color = "#dc3545"
                        strength = "Weak"
                    
                    st.markdown(f"""
                        <div style="background-color:{color}; color:white; padding:20px; border-radius:8px; text-align:center;">
                            <h2 style="margin:0;">{best_affinity} kcal/mol</h2>
                            <p style="margin:5px 0 0 0;">Best Binding Affinity</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("Binding Strength", strength)
                with col3:
                    st.metric("Binding Modes", len(results.get('modes', [])))
                
                st.markdown("---")
                
                # NEW: 3D Visualization
                st.subheader("🔬 3D Protein-Ligand Complex")
                
                # Visualization mode toggle
                col_viz_toggle, col_info = st.columns([1, 3])
                with col_viz_toggle:
                    viz_mode = st.radio(
                        "View Mode:",
                        options=["Cartoon (Ribbon)", "All-Atom (Ball-and-Stick)"],
                        index=0,
                        horizontal=False,
                        key="docking_viz_mode"
                    )
                
                if results.get('has_coordinates'):
                    # Show binding site coordinates
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        viewer_html = ProteinVisualizer.create_docking_3d_viewer(
                            protein_structure,
                            ligand_data,
                            results,
                            ligand_name,
                            view_mode=viz_mode
                        )
                        st.components.v1.html(viewer_html, height=650, scrolling=False)
                    
                    with col2:
                        st.markdown("**Best Binding Mode:**")
                        st.markdown(f"""
                        **Position (Å):**
                        - X: {best_mode.get('center', {}).get('x', 0):.2f}
                        - Y: {best_mode.get('center', {}).get('y', 0):.2f}
                        - Z: {best_mode.get('center', {}).get('z', 0):.2f}
                        
                        **Orientation:**
                        - {best_mode.get('orientation', 'N/A')}
                        
                        **RMSD:**
                        - Lower bound: {best_mode.get('rmsd_lb', 0):.2f} Å
                        - Upper bound: {best_mode.get('rmsd_ub', 0):.2f} Å
                        """)
                        
                        if viz_mode == "Cartoon (Ribbon)":
                            st.info("""
                            **View: Ribbon Mode**
                            
                            Clean academic view showing protein backbone as smooth ribbon structure.
                            """)
                        else:
                            st.info("""
                            **View: All-Atom Mode**
                            
                            Detailed atomic structure with all atoms shown as balls and sticks.
                            """)
                
                st.markdown("---")
                
                # Binding modes chart
                st.subheader("📊 All Binding Modes")
                fig_docking = ProteinVisualizer.create_docking_results_chart(results)
                st.plotly_chart(fig_docking, width='stretch')
                
                # Detailed modes table with coordinates
                st.subheader("📋 Binding Mode Details")
                
                modes_df = pd.DataFrame(results.get('modes', []))
                
                # Format for display
                if not modes_df.empty:
                    display_df = modes_df[['mode', 'affinity', 'orientation', 'rmsd_lb', 'rmsd_ub']].copy()
                    display_df.columns = ['Mode', 'Affinity (kcal/mol)', 'Orientation', 'RMSD Lower', 'RMSD Upper']
                    st.dataframe(display_df, width='stretch', hide_index=True)
                
                # Interpretation
                st.subheader("💡 Interpretation")
                
                if best_affinity < -7:
                    st.success("""
                    **Strong Binding** (< -7 kcal/mol)
                    - Indicates favorable protein-ligand interaction
                    - This compound shows drug-like binding affinity
                    - Worth further experimental validation
                    - Predicted binding orientation suggests stable complex
                    """)
                elif best_affinity < -5:
                    st.info("""
                    **Moderate Binding** (-5 to -7 kcal/mol)
                    - Shows some binding potential
                    - May require optimization for better affinity
                    - Consider structural modifications
                    - Multiple binding orientations possible
                    """)
                else:
                    st.warning("""
                    **Weak Binding** (> -5 kcal/mol)
                    - Limited binding affinity
                    - Unlikely to be effective inhibitor
                    - Significant optimization needed
                    - Consider alternative scaffolds
                    """)
                
                # Download results
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_modes = modes_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Docking Results",
                        csv_modes,
                        f"docking_{st.session_state.current_uniprot_id}_{ligand_name}.csv",
                        "text/csv",
                        key="download_docking_results"
                    )
                
                with col2:
                    # Coordinates for best mode
                    coords_text = f"""Best Binding Mode Coordinates
                    Ligand: {ligand_name}
                    Protein: {st.session_state.current_uniprot_id}
                    Affinity: {best_affinity} kcal/mol

                    Position (Å):
                    X: {best_mode.get('center', {}).get('x', 0):.3f}
                    Y: {best_mode.get('center', {}).get('y', 0):.3f}
                    Z: {best_mode.get('center', {}).get('z', 0):.3f}

                    Orientation:
                    {best_mode.get('orientation', 'N/A')}
                    """
                    st.download_button(
                        "📥 Download 3D Coordinates",
                        coords_text,
                        f"coordinates_{ligand_name}.txt",
                        "text/plain",
                        key="download_coordinates"
                    )
                
                # Clear results
                if st.button("🔄 Run New Docking", key="docking_new_run"):
                    del st.session_state.docking_results
                    del st.session_state.docked_ligand_name
                    del st.session_state.docked_ligand_data
                    if 'custom_ligand' in st.session_state:
                        del st.session_state.custom_ligand
                    if 'binding_prediction' in st.session_state:
                        del st.session_state.binding_prediction
                    st.rerun()
            
            else:
                st.info("👈 Run a docking simulation in the 'Custom Docking' tab to see results here")
        
        st.divider()

        # Section 9: Summary Table
        st.header("📊 Data Summary")
        
        summary_df = ProteinAPIClient.DataProcessor.create_summary_table(
            uniprot_data, 
            tissue_df, 
            subcellular_df,
            data.get('alphafold_structure'),
            data.get('pdb_structure'),
            data.get('kegg_pathways'),
            data.get('chembl_ligands')
        )

        st.dataframe(summary_df, width='stretch', hide_index=True)
        
        # Download options
        st.subheader("💾 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not tissue_df.empty:
                csv_tissue = tissue_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Tissue Data",
                    csv_tissue,
                    f"{st.session_state.current_uniprot_id}_tissue_expression.csv",
                    "text/csv"
                )
        
        with col2:
            if not subcellular_df.empty:
                csv_subcellular = subcellular_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Subcellular Data",
                    csv_subcellular,
                    f"{st.session_state.current_uniprot_id}_subcellular.csv",
                    "text/csv"
                )
        
        with col3:
            csv_summary = summary_df.to_csv(index=False)
            st.download_button(
                "📥 Download Summary",
                csv_summary,
                f"{st.session_state.current_uniprot_id}_summary.csv",
                "text/csv"
            )
        
        # Footer
        st.divider()
        st.caption(f"⏱️ Data fetched in {st.session_state.get('fetch_time', 0):.2f}s | 💾 Cached for 24 hours | 🔬 Data sources: UniProt, Human Protein Atlas")
        
        # Section: Protein Literature Summary
        with st.expander("🔬 Literature & Overview", expanded=False):
            literature = data.get('literature', {})
            # Wikipedia intro
            if literature.get('wiki_title'):
                st.info(f"**Wikipedia**: [{literature['wiki_title']}](https://en.wikipedia.org/wiki/{literature['wiki_title'].replace(' ', '_')})")
                st.caption(literature.get('wiki_snippet', ''))
            # Top papers
            if literature.get('papers'):
                st.subheader("Top 5 Research Papers")
                for i, p in enumerate(literature['papers'], 1):
                    with st.container():
                        st.markdown(f"**{p['title']}**")
                        st.caption(f"{p['authors']} | [PMID: {p['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{p['pmid']})")
                        st.caption(p['abstract_snip'])
                        st.divider()
            else:
                st.warning("No recent papers found; try official gene name.")

        st.divider()
        st.header("🧠 OmniBiMol AI Research Copilot")
        st.caption("Evidence-grounded target assessment and translational hypothesis support (research use only).")

        with st.expander("Copilot Operating Contract", expanded=False):
            st.code(OMNIBIMOL_RESEARCH_COPILOT_SYSTEM_PROMPT, language="markdown")

        default_query = "Why is this target druggable?"
        copilot_query = st.text_area(
            "Ask OmniBiMol Copilot",
            value=st.session_state.get("omnibimol_copilot_query", default_query),
            height=120,
            key="omnibimol_copilot_query",
            help='Examples: "Why is this target druggable?", "hypothesis cards", "experimental next steps", "risk flags".',
        )

        if st.button("Generate Copilot Analysis", key="run_omnibimol_copilot", type="primary"):
            with st.spinner("Synthesizing evidence-grounded copilot response..."):
                context_payload = _build_omnibimol_context_payload(data, uniprot_data)
                copilot_output = _generate_omnibimol_copilot_response(copilot_query, context_payload)
                st.session_state.omnibimol_copilot_output = copilot_output

        if st.session_state.get("omnibimol_copilot_output"):
            st.markdown(st.session_state.omnibimol_copilot_output)
            st.download_button(
                "📥 Download Copilot Analysis",
                st.session_state.omnibimol_copilot_output,
                f"{st.session_state.current_uniprot_id}_omnibimol_copilot_analysis.md",
                "text/markdown",
                key="download_omnibimol_copilot_output",
            )


# =============================================================================
# SEQUENCE ANALYSIS PAGE FUNCTIONS
# =============================================================================

def render_sequence_analysis_page():
    """Render the main sequence analysis page"""
    
    st.header("🧬 Sequence Analysis Suite")
    st.markdown("""
    Comprehensive computational analysis of biological sequences (DNA, RNA, or protein).
    Upload FASTA files to perform multiple sequence alignment, phylogenetic analysis, 
    domain identification, motif finding, and conservation scoring.
    """)
    
    # Initialize analysis suite
    if 'sequence_analyzer' not in st.session_state:
        st.session_state.sequence_analyzer = SequenceAnalysisSuite()
    
    analyzer = st.session_state.sequence_analyzer
    
    # File upload section
    st.subheader("📤 Upload Sequences")
    
    uploaded_file = st.file_uploader(
        "Upload FASTA file",
        type=['fasta', 'fa', 'fas', 'txt'],
        help="Upload a FASTA file containing one or more sequences"
    )
    
    # Alternative: text input
    st.markdown("**OR** paste FASTA content directly:")
    fasta_text = st.text_area(
        "FASTA Content",
        height=200,
        help="Paste FASTA formatted sequences here"
    )
    
    # Get FASTA content
    fasta_content = None
    if uploaded_file is not None:
        fasta_content = uploaded_file.read().decode('utf-8')
        st.success(f"✅ File uploaded: {uploaded_file.name}")
    elif fasta_text.strip():
        fasta_content = fasta_text
    
    # Analysis options (Sequence Analysis Suite)
    if fasta_content:
        st.subheader("⚙️ Analysis Options")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            run_alignment = st.checkbox("Multiple Sequence Alignment", value=True)
            run_conservation = st.checkbox("Conservation Scoring", value=True)
        with col2:
            run_phylogeny = st.checkbox("Phylogenetic Tree", value=True)
            run_domains = st.checkbox("Domain Identification", value=True)
        with col3:
            run_motifs = st.checkbox("Motif Finding", value=True)
        
        # Run analysis button
        if st.button("🚀 Run Analysis", type="primary", width='stretch'):
            with st.spinner("Running sequence analysis..."):
                try:
                    results = analyzer.analyze(
                        fasta_content,
                        run_alignment=run_alignment,
                        run_phylogeny=run_phylogeny,
                        run_domains=run_domains,
                        run_motifs=run_motifs,
                        run_conservation=run_conservation
                    )
                    st.session_state.sequence_analysis_results = results
                    if results.get("errors"):
                        st.warning("Analysis completed with some errors. Check the results section for details.")
                    else:
                        st.success("✅ Analysis completed successfully!")
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.exception(e)
        
        # Display results
        if 'sequence_analysis_results' in st.session_state:
            results = st.session_state.sequence_analysis_results
            display_analysis_results(results, analyzer)
    
    # ------------------------------------------------------------------
    # Protein Predictor section (always visible, separate FASTA input)
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("🧪 Protein Predictor")
    st.markdown(
        "Predict protein annotations and structure from amino acid FASTA "
        "and explore docking using the existing simulation pipeline."
    )

    st.markdown("#### Protein FASTA Input")
    protein_uploaded_file = st.file_uploader(
        "Upload protein FASTA file",
        type=['fasta', 'fa', 'fas', 'txt'],
        help="Upload a FASTA file containing one or more protein sequences",
        key="protein_predictor_file_uploader",
    )

    st.markdown("**OR** paste protein FASTA content directly:")
    protein_fasta_text = st.text_area(
        "Protein FASTA Content",
        height=180,
        help="Paste amino acid FASTA formatted sequences here",
        key="protein_predictor_fasta_text",
    )

    protein_fasta_content = None
    if protein_uploaded_file is not None:
        protein_fasta_content = protein_uploaded_file.read().decode("utf-8")
        st.success(f"✅ Protein FASTA file uploaded: {protein_uploaded_file.name}")
    elif protein_fasta_text.strip():
        protein_fasta_content = protein_fasta_text

    _render_protein_predictor(protein_fasta_content or "")

    # Example FASTA
    with st.expander("📝 Example FASTA Format"):
        st.code(""">sequence1
ATGCGATCGATCGATCGATCG
>sequence2
ATGCGATCGATCGATCGATCG
>sequence3
ATGCGATCGATCGATCGATCG
        """, language="text")


def _render_protein_predictor(protein_fasta_content: str):
    """
    Render protein predictor with molecular docking capability.
    Uses FASTA input as the protein source for docking.
    """
    if not protein_fasta_content or not protein_fasta_content.strip():
        st.info("📝 Upload or paste a protein FASTA sequence above to proceed")
        return
    
    # Parse FASTA
    try:
        fasta_parser = FASTAParser()
        sequences = fasta_parser.parse_fasta_string(protein_fasta_content)
        
        if not sequences:
            st.error("❌ Invalid FASTA format. Please check your input.")
            return
        
        st.success(f"✅ Parsed {len(sequences)} sequence(s) from FASTA")
        
        # Use the first sequence for analysis
        seq_record = sequences[0]
        protein_sequence = seq_record['sequence']
        protein_name = seq_record.get('id', 'Predicted Protein')
        
        st.markdown(f"**Protein:** {protein_name} ({len(protein_sequence)} aa)")
        
    except Exception as e:
        st.error(f"❌ Error parsing FASTA: {str(e)}")
        return
    
    # ------------------------------------------------------------------
    # MOLECULAR DOCKING SECTION (replica of Protein Analysis tab)
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("💊 Molecular Docking Analysis")
    st.markdown("Predict ligand-protein binding using structure derived from your FASTA sequence.")
    
    st.info("""
    **About Molecular Docking:**
    - Predicts how small molecules (ligands/drugs) bind to proteins
    - Uses AutoDock Vina algorithm for binding affinity calculation
    - Negative values indicate favorable binding (more negative = stronger binding)
    - Typical drug-like binding: -7 to -12 kcal/mol
    - 3D visualization of ligand orientation and binding prediction
    """)
    
    # Simulate protein structure preparation from FASTA
    protein_prep = {
        "available": True,
        "structure_type": "Predicted (from FASTA)",
        "structure_id": protein_name,
        "sequence_length": len(protein_sequence),
        "pdb_text": _generate_mock_pdb_from_sequence(protein_sequence, protein_name),
        "pdb_url": ""
    }
    
    if not protein_prep.get("available"):
        st.error("❌ Unable to prepare protein structure for docking.")
        return
    
    st.success(f"✅ Protein prepared: {protein_prep['structure_type']} - {protein_prep['sequence_length']} residues")
    
    # Docking interface
    st.markdown("#### Ligand Input & Docking Parameters")
    
    docking_col1, docking_col2 = st.columns(2)
    
    with docking_col1:
        st.markdown("**Select Ligand Source:**")
        ligand_source = st.radio(
            "Choose ligand source",
            ["Enter SMILES manually", "Custom compound (PubChem)", "Upload SMILES/SDF"],
            horizontal=False,
            key="seq_analysis_ligand_source",
            label_visibility="collapsed"
        )
        
        selected_ligand = None
        ligand_name = None
        
        if ligand_source == "Enter SMILES manually":
            smiles_input = st.text_input(
                "Enter SMILES string:",
                placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O (Aspirin)",
                key="seq_analysis_smiles_input"
            )
            
            if smiles_input:
                ligand_name = "Custom_SMILES"
                selected_ligand = {"smiles": smiles_input, "name": ligand_name, "molecular_weight": 200}
        
        elif ligand_source == "Custom compound (PubChem)":
            compound_name = st.text_input(
                "Enter compound name:",
                placeholder="e.g., Aspirin, Ibuprofen, Caffeine",
                key="seq_analysis_pubchem_input"
            )
            
            if compound_name and st.button("🔍 Search PubChem", key="seq_analysis_pubchem_search"):
                with st.spinner("Searching PubChem..."):
                    pubchem_data = cached_fetch_pubchem_structure(compound_name, st.session_state.api_client)
                    
                    if pubchem_data.get('available'):
                        st.success(f"✅ Found: {compound_name} (CID: {pubchem_data['cid']})")
                        st.image(pubchem_data['image_url'], width=200)
                        st.session_state.seq_analysis_custom_ligand = pubchem_data
                        selected_ligand = pubchem_data
                        ligand_name = compound_name
                    else:
                        st.error(f"❌ Compound '{compound_name}' not found in PubChem")
            
            if 'seq_analysis_custom_ligand' in st.session_state:
                selected_ligand = st.session_state.seq_analysis_custom_ligand
                ligand_name = compound_name
        
        else:  # Upload SMILES/SDF
            st.info("Upload SMILES or SDF file support would be added here")
    
    with docking_col2:
        st.markdown("**Docking Parameters:**")
        
        exhaustiveness = st.slider("Exhaustiveness", 1, 16, 8, 
                                  help="Higher = more thorough but slower",
                                  key="seq_analysis_exhaustiveness")
        num_modes = st.slider("Number of modes", 1, 20, 9,
                             help="Number of binding poses to generate",
                             key="seq_analysis_num_modes")
        energy_range = st.slider("Energy range (kcal/mol)", 1, 5, 3,
                                key="seq_analysis_energy_range")
    
    st.markdown("---")
    
    # Run docking button
    if selected_ligand:
        if st.button("🚀 Run Molecular Docking", type="primary", key="seq_analysis_run_docking", width='stretch'):
            with st.spinner("🧬 Running AutoDock Vina simulation... Calculating 3D orientation..."):
                time.sleep(2)
                
                # Call docking API
                docking_results = st.session_state.api_client.simulate_docking_score(
                    protein_prep['sequence_length'],
                    selected_ligand.get('molecular_weight', 300),
                    None,
                    selected_ligand.get('smiles')
                )
                
                # Store results in session state for display
                st.session_state.seq_analysis_docking_results = docking_results
                st.session_state.seq_analysis_docked_ligand_name = ligand_name
                st.session_state.seq_analysis_docked_ligand_data = selected_ligand
                st.session_state.seq_analysis_protein_structure = protein_prep
                st.rerun()
    else:
        st.info("👆 Please select or enter a ligand above to proceed with docking")
    
    # Display docking results (if available)
    if 'seq_analysis_docking_results' in st.session_state:
        results = st.session_state.seq_analysis_docking_results
        ligand_name_display = st.session_state.get('seq_analysis_docked_ligand_name', 'Unknown')
        ligand_data = st.session_state.get('seq_analysis_docked_ligand_data', {})
        protein_structure = st.session_state.get('seq_analysis_protein_structure', {})
        
        st.divider()
        st.subheader("📊 Docking Results")
        
        if results.get('simulated'):
            st.warning("⚠️ **Note:** These are simulated results for demonstration. Production version would use actual AutoDock Vina calculations.")
        
        st.markdown(f"### Results for: **{ligand_name_display}**")
        
        # Best binding affinity
        best_affinity = results['binding_affinity']
        best_mode = results.get('best_mode', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if best_affinity < -7:
                color = "#28a745"
                strength = "Strong"
            elif best_affinity < -5:
                color = "#ffc107"
                strength = "Moderate"
            else:
                color = "#dc3545"
                strength = "Weak"
            
            st.markdown(f"""
                <div style="background-color:{color}; color:white; padding:20px; border-radius:8px; text-align:center;">
                    <h2 style="margin:0;">{best_affinity} kcal/mol</h2>
                    <p style="margin:5px 0 0 0;">Best Binding Affinity</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Binding Strength", strength)
        with col3:
            st.metric("Binding Modes", len(results.get('modes', [])))
        
        st.markdown("---")
        
        # 3D Visualization
        st.subheader("🔬 3D Protein-Ligand Complex")
        
        if results.get('has_coordinates'):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                viewer_html = ProteinVisualizer.create_docking_3d_viewer(
                    protein_structure,
                    ligand_data,
                    results,
                    ligand_name_display
                )
                st.components.v1.html(viewer_html, height=650, scrolling=False)
            
            with col2:
                st.markdown("**Best Binding Mode:**")
                st.markdown(f"""
                **Position (Å):**
                - X: {best_mode.get('center', {}).get('x', 0):.2f}
                - Y: {best_mode.get('center', {}).get('y', 0):.2f}
                - Z: {best_mode.get('center', {}).get('z', 0):.2f}
                
                **RMSD:**
                - Lower bound: {best_mode.get('rmsd_lb', 0):.2f} Å
                - Upper bound: {best_mode.get('rmsd_ub', 0):.2f} Å
                """)
                
                st.info("""
                **3D Controls:**
                - Left click: Rotate
                - Right click: Zoom
                - Middle: Pan
                """)
        
        st.markdown("---")
        
        # Binding modes chart
        st.subheader("📊 All Binding Modes")
        fig_docking = ProteinVisualizer.create_docking_results_chart(results)
        st.plotly_chart(fig_docking, width='stretch')
        
        # Download results
        st.markdown("---")
        st.subheader("📥 Export Results")
        
        results_csv = pd.DataFrame([
            {
                "Ligand": ligand_name_display,
                "Protein_Source": "FASTA Sequence",
                "Binding_Affinity_kcal_mol": best_affinity,
                "Strength": strength,
                "Modes": len(results.get('modes', [])),
                "Timestamp": datetime.now().isoformat()
            }
        ]).to_csv(index=False)
        
        st.download_button(
            "📥 Download Docking Results (CSV)",
            results_csv,
            f"docking_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            key="seq_analysis_download_docking"
        )


def _generate_mock_pdb_from_sequence(sequence: str, name: str) -> str:
    """
    Generate a mock PDB file from a protein sequence for visualization.
    This is a placeholder that creates a simplified PDB structure.
    """
    pdb_content = f"""HEADER    SEQUENCE DERIVED STRUCTURE                     01-JAN-26   PRED              
TITLE     PREDICTED STRUCTURE FROM FASTA SEQUENCE
REMARK 1  REFERENCE 1
REMARK 1   AUTH   OMNIBIMOL SEQUENCE ANALYSIS SUITE
REMARK   1  FASTA INPUT: {name[:60]}
REMARK   2  SEQUENCE LENGTH: {len(sequence)} RESIDUES
REMARK   3  STRUCTURE GENERATED FOR DOCKING VISUALIZATION
REMARK  99  THIS IS A MOCK STRUCTURE FOR DEMONSTRATION PURPOSES
"""
    
    # Add simple CA atom trace
    for i, aa in enumerate(sequence[:100]):  # Limit to 100 residues for demo
        x = 10.0 + (i % 10) * 3.8
        y = 10.0 + ((i // 10) % 10) * 3.8
        z = 10.0 + ((i // 100) % 10) * 3.8
        
        pdb_content += f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00           C\n"
    
    pdb_content += "END\n"
    
    return pdb_content


def display_analysis_results(results: Dict, analyzer: SequenceAnalysisSuite):
    """Display comprehensive analysis results"""
    
    st.divider()
    st.subheader("📊 Analysis Results")
    
    # Errors
    if results.get("errors"):
        st.error("⚠️ Errors encountered:")
        for error in results["errors"]:
            st.error(f"  - {error}")
    
    # Input sequences summary
    if results.get("input_sequences"):
        st.markdown("### Input Sequences")
        seq_df = pd.DataFrame(results["input_sequences"])
        st.dataframe(seq_df, width='stretch')
    
    # Multiple Sequence Alignment
    if results.get("alignment"):
        st.markdown("### Multiple Sequence Alignment")
        align_data = results["alignment"]
        metadata = align_data.get("metadata", {})
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Alignment Length", metadata.get("alignment_length", "N/A"))
        with col2:
            st.metric("Sequences", metadata.get("num_sequences", "N/A"))
        with col3:
            st.metric("Conserved Positions", metadata.get("conserved_positions", "N/A"))
        with col4:
            st.metric("Conservation", f"{metadata.get('conservation_percentage', 0):.1f}%")
        
        # Display aligned sequences
        with st.expander("View Aligned Sequences"):
            aligned_seqs = align_data.get("aligned_sequences", [])
            for seq in aligned_seqs:
                st.text(f">{seq['id']}")
                # Display in chunks for readability
                sequence = seq['sequence']
                chunk_size = 80
                for i in range(0, len(sequence), chunk_size):
                    st.text(sequence[i:i+chunk_size])
    
    # Conservation Analysis
    if results.get("conservation"):
        st.markdown("### Conservation Analysis")
        cons_data = results["conservation"]
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Conservation", f"{cons_data.get('mean_conservation', 0):.4f}")
        with col2:
            st.metric("Std Deviation", f"{cons_data.get('std_conservation', 0):.4f}")
        with col3:
            st.metric("Min Conservation", f"{cons_data.get('min_conservation', 0):.4f}")
        with col4:
            st.metric("Max Conservation", f"{cons_data.get('max_conservation', 0):.4f}")
        
        # Conservation plot
        scores = [pos["score"] for pos in cons_data.get("scores", [])]
        positions = [pos["position"] for pos in cons_data.get("scores", [])]
        
        if scores and positions:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=positions,
                y=scores,
                mode='lines',
                name='Conservation Score',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)'
            ))
            fig.update_layout(
                title="Conservation Score Across Alignment",
                xaxis_title="Position",
                yaxis_title="Conservation Score (1.0 = fully conserved)",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, width='stretch')
        
        # Highly conserved positions
        highly_cons = cons_data.get("highly_conserved_positions", [])
        if highly_cons:
            st.info(f"🔍 Highly conserved positions (>90th percentile): {', '.join(map(str, highly_cons[:20]))}{'...' if len(highly_cons) > 20 else ''}")
    
    # Domain Identification
    if results.get("domains"):
        st.markdown("### Domain Identification")
        domains = results["domains"]
        
        domain_count = sum(len(d) for d in domains.values())
        if domain_count > 0:
            st.metric("Total Domains Found", domain_count)
            
            # Display domains per sequence
            for seq_id, domain_list in domains.items():
                if domain_list:
                    with st.expander(f"Domains in {seq_id}"):
                        domain_df = pd.DataFrame(domain_list)
                        st.dataframe(domain_df, width='stretch')
                        
                        # Visualize domain positions
                        if domain_list:
                            fig = go.Figure()
                            colors = px.colors.qualitative.Set3
                            for i, domain in enumerate(domain_list):
                                fig.add_trace(go.Scatter(
                                    x=[domain["start"], domain["end"]],
                                    y=[seq_id] * 2,
                                    mode='lines+markers',
                                    name=domain["domain_name"],
                                    line=dict(width=10, color=colors[i % len(colors)]),
                                    marker=dict(size=10)
                                ))
                            
                            fig.update_layout(
                                title=f"Domain Positions in {seq_id}",
                                xaxis_title="Position",
                                yaxis_title="Sequence",
                                height=300,
                                showlegend=True
                            )
                            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No domains identified in the sequences.")
    
    # Motif Finding
    if results.get("motifs"):
        st.markdown("### Motif Analysis")
        motifs_data = results["motifs"]
        
        num_motifs = motifs_data.get("num_motifs", 0)
        st.metric("Motifs Found", num_motifs)
        st.caption(f"Method: {motifs_data.get('method', 'N/A')}")
        
        motifs_list = motifs_data.get("motifs", [])
        if motifs_list:
            # Display top motifs
            with st.expander("View Motifs"):
                motif_df_data = []
                for motif in motifs_list[:20]:  # Top 20
                    motif_df_data.append({
                        "Motif": motif.get("motif", "N/A"),
                        "Length": motif.get("length", "N/A"),
                        "Frequency": motif.get("frequency", "N/A"),
                        "Conservation": f"{motif.get('conservation', 0)*100:.1f}%" if "conservation" in motif else "N/A",
                        "Sequences": len(motif.get("sequences", []))
                    })
                
                if motif_df_data:
                    motif_df = pd.DataFrame(motif_df_data)
                    st.dataframe(motif_df, width='stretch')
        else:
            st.info("No motifs found in the sequences.")
    
    # Phylogenetic Tree
    if results.get("phylogenetic_tree"):
        st.markdown("### Phylogenetic Tree")
        tree_data = results["phylogenetic_tree"]
        metadata = tree_data.get("metadata", {})
        
        # Display metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Method", metadata.get("method", "N/A").upper())
        with col2:
            st.metric("Taxa", metadata.get("num_taxa", "N/A"))
        with col3:
            st.metric("Tree Length", f"{metadata.get('tree_length', 0):.4f}")
        
        st.markdown("---")
        
        # Display the phylogenetic tree visualization
        tree_html = ProteinVisualizer.create_phylogenetic_tree_visualization(
            tree_data.get("newick", "N/A"),
            metadata
        )
        # Adjust height based on number of taxa for better display
        display_height = max(500, metadata.get('num_taxa', 2) * 80 + 300)
        st.components.v1.html(tree_html, height=display_height, scrolling=True)
    
    # Download Report
    st.divider()
    st.markdown("### 📥 Download Report")
    
    report_text = analyzer.generate_report(results)
    
    st.download_button(
        label="Download Analysis Report (TXT)",
        data=report_text,
        file_name="sequence_analysis_report.txt",
        mime="text/plain"
    )
    
    # JSON export
    report_json = json.dumps(results, indent=2, default=str)
    st.download_button(
        label="Download Analysis Results (JSON)",
        data=report_json,
        file_name="sequence_analysis_results.json",
        mime="application/json"
    )


def _render_protein_predictor(fasta_content: str) -> None:
    """Render the Protein Predictor section for protein FASTA sequences."""
    if not fasta_content or not fasta_content.strip():
        st.info("Upload or paste a protein FASTA sequence above to use the Protein Predictor.")
        return

    # Parse sequences with existing FASTA parser
    try:
        sequences = FASTAParser.parse(fasta_content)
    except Exception as e:
        st.warning(f"Unable to parse FASTA for Protein Predictor: {str(e)}")
        return

    protein_seqs = [s for s in sequences if s.sequence_type == "protein"]

    if not protein_seqs:
        st.info("Protein Predictor requires at least one amino acid (protein) sequence in the FASTA input.")
        return

    # Sequence selector
    options = {
        f"{seq.id} ({len(seq.sequence)} aa)": seq for seq in protein_seqs
    }
    selected_label = st.selectbox(
        "Select protein sequence for prediction",
        list(options.keys()),
        key="protein_predictor_seq_select"
    )
    selected_seq = options[selected_label].sequence

    st.caption(f"Using sequence length: {len(selected_seq)} amino acids")

    # Cache key for session state (per raw sequence)
    seq_key = base64.urlsafe_b64encode(selected_seq.encode("utf-8"))[:16].decode("utf-8")

    # Create tabs for organized protein prediction analysis
    predictor_tabs = st.tabs(["🔍 Protein Name (NCBI Lookup)", "🧠 Protein Structure Prediction", "🧪 Molecular Docking"])
    
    # ----------------------
    # Tab 1: Protein Name (NCBI)
    # ----------------------
    with predictor_tabs[0]:
        st.subheader("Protein Name Identification")
        st.info("""
        **About NCBI Protein Lookup:**
        - 🔬 Uses BLASTp against curated protein databases
        - 🧬 Identifies known protein matches and annotations
        - 🏆 Returns best match with identity and coverage metrics
        - 💾 Results cached per sequence
        """)

        col_ncbi_btn, col_ncbi_status = st.columns([1, 2])
        with col_ncbi_btn:
            lookup_clicked = st.button(
                "🔎 Search NCBI for known protein",
                key=f"protein_predictor_ncbi_btn_{seq_key}",
                help="Run BLASTp against curated protein databases to find known proteins",
                type="primary"
            )
        with col_ncbi_status:
            st.caption("BLASTp search with short polling. Results are cached per sequence.")

        if lookup_clicked:
            st.session_state[f"protein_predictor_ncbi_pending_{seq_key}"] = True

        ncbi_result_key = f"protein_predictor_ncbi_result_{seq_key}"
        if st.session_state.get(f"protein_predictor_ncbi_pending_{seq_key}") and ncbi_result_key not in st.session_state:
            # Trigger lookup only once per sequence
            if "api_client" in st.session_state:
                import asyncio

                if not hasattr(st.session_state.api_client, "search_protein_ncbi"):
                    st.error(
                        "NCBI protein search is not available. Please make sure your "
                        "`api_client.py` includes the `search_protein_ncbi` method and "
                        "restart the Streamlit app."
                    )
                else:
                    # Create a progress container for better user feedback
                    progress_container = st.empty()
                    status_container = st.empty()
                    
                    with progress_container.container():
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        status_text.text("🔍 Submitting BLAST query to NCBI...")
                    
                    try:
                        # Use asyncio to run the search
                        import time
                        start_time = time.time()
                        
                        # Update progress simulation (since we can't get real-time updates from async)
                        status_text.text("⏳ Searching NCBI database (typically 5-20 seconds)...")
                        progress_bar.progress(20)
                        
                        try:
                            ncbi_result = asyncio.run(
                                st.session_state.api_client.search_protein_ncbi(selected_seq)
                            )
                        except RuntimeError:
                            # In case we're already in an event loop, fall back to direct await via loop
                            loop = asyncio.new_event_loop()
                            try:
                                asyncio.set_event_loop(loop)
                                ncbi_result = loop.run_until_complete(
                                    st.session_state.api_client.search_protein_ncbi(selected_seq)
                                )
                            finally:
                                loop.close()
                        
                        elapsed = time.time() - start_time
                        progress_bar.progress(100)
                        status_text.text(f"✅ Search completed in {elapsed:.1f} seconds")
                        
                        st.session_state[ncbi_result_key] = ncbi_result
                        
                        # Clear progress indicators after a brief display
                        time.sleep(1)
                        progress_container.empty()
                        status_container.empty()
                        
                    except Exception as e:
                        progress_container.empty()
                        st.error(f"❌ NCBI search failed: {str(e)}")
            else:
                st.error("API client not available in session state; cannot contact NCBI.")

        ncbi_result = st.session_state.get(ncbi_result_key)
        if ncbi_result:
            if ncbi_result.get("available") and ncbi_result.get("match_found"):
                st.success("✅ Protein identified in NCBI database")
                
                st.markdown("---")
                
                # Display protein information with full text using native Streamlit components
                st.subheader("🔬 Protein Information")
                
                protein_name = ncbi_result.get("protein_name", "N/A")
                accession_id = ncbi_result.get("accession_id", "N/A")
                organism = ncbi_result.get("organism", "N/A")
                
                # Use container with background color
                with st.container():
                    st.markdown("**🧬 Protein Name:**")
                    st.info(protein_name)
                    
                    st.markdown("**🔑 Accession ID:**")
                    st.text(accession_id)
                    
                    st.markdown("**🦠 Organism:**")
                    st.text(organism)

                st.markdown("---")
                st.subheader("📊 Alignment Metrics")
                
                # Alignment metrics - these are fine with st.metric as they're short
                col1, col2, col3 = st.columns(3)
                with col1:
                    identity = ncbi_result.get('identity_percent', 0)
                    st.metric("Identity", f"{identity:.2f}%")
                with col2:
                    coverage = ncbi_result.get('coverage_percent', 0)
                    st.metric("Coverage", f"{coverage:.2f}%")
                with col3:
                    evalue = ncbi_result.get('e_value', 1.0)
                    st.metric("E-value", f"{evalue:.2g}")

                if ncbi_result.get("ncbi_url"):
                    st.markdown(f"🔗 [View detailed information in NCBI Protein Database]({ncbi_result['ncbi_url']})")
            elif ncbi_result.get("available") and not ncbi_result.get("match_found"):
                st.info("🔬 Protein name not found (novel or unannotated sequence)")
            else:
                st.warning(ncbi_result.get("error", "NCBI lookup unavailable."))

    # ----------------------
    # Tab 2: Protein Structure Prediction
    # ----------------------
    with predictor_tabs[1]:
        st.subheader("3D Structure Prediction")
        st.info("""
        **About Structure Prediction:**
        - 🧱 Uses ESMFold API for accurate structure prediction
        - 🎯 No local models or GPU required
        - 📊 Provides confidence scores (pLDDT)
        - 🔬 Interactive 3D visualization
        - 💾 Results cached per sequence
        """)

        if py3Dmol is None:
            st.warning(
                "⚠️ py3Dmol is not available in the environment. "
                "3D visualization is disabled, but PDB download will still be available."
            )

        col_struct_btn, col_struct_status = st.columns([1, 2])
        with col_struct_btn:
            predict_clicked = st.button(
                "🧠 Predict 3D Structure (ESMFold)",
                key=f"protein_predictor_structure_btn_{seq_key}",
                type="primary"
            )
        with col_struct_status:
            st.caption("Remote ESMFold API - no local models or GPU required.")

        if predict_clicked:
            st.session_state[f"protein_predictor_structure_pending_{seq_key}"] = True

        struct_result_key = f"protein_predictor_structure_result_{seq_key}"
        if st.session_state.get(f"protein_predictor_structure_pending_{seq_key}") and struct_result_key not in st.session_state:
            if "api_client" in st.session_state:
                import asyncio

                with st.spinner("🧱 Predicting protein structure..."):
                    try:
                        struct_result = asyncio.run(
                            st.session_state.api_client.predict_structure(selected_seq)
                        )
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        try:
                            asyncio.set_event_loop(loop)
                            struct_result = loop.run_until_complete(
                                st.session_state.api_client.predict_structure(selected_seq)
                            )
                        finally:
                            loop.close()

                    st.session_state[struct_result_key] = struct_result
            else:
                st.error("API client not available in session state; cannot run structure prediction.")

        struct_result = st.session_state.get(struct_result_key)

        protein_structure_for_docking = None

        if struct_result:
            if struct_result.get("available"):
                avg_plddt = struct_result.get("avg_plddt")
                if avg_plddt is not None:
                    st.success(f"✅ Structure predicted successfully")
                    
                    # Display confidence metric
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average pLDDT Score", f"{avg_plddt:.1f}")
                    with col2:
                        confidence_level = "High" if avg_plddt > 80 else "Medium" if avg_plddt > 60 else "Low"
                        st.metric("Confidence Level", confidence_level)
                else:
                    st.success("✅ Structure predicted (confidence scores not provided)")

                st.markdown("---")
                
                pdb_text = struct_result.get("pdb", "")

                # Download PDB
                st.download_button(
                    "📥 Download Predicted PDB File",
                    pdb_text,
                    file_name="predicted_structure_esmfold.pdb",
                    mime="chemical/x-pdb",
                    key=f"download_predicted_pdb_{seq_key}",
                )

                st.markdown("---")
                st.markdown("**3D Structure Visualization**")
                
                # Visualize with py3Dmol if available
                if py3Dmol is not None and pdb_text:
                    view = py3Dmol.view(width=800, height=500)
                    view.addModel(pdb_text, "pdb")
                    view.setStyle({"cartoon": {"color": "spectrum"}})
                    view.zoomTo()
                    html_view = view._make_html()
                    st.components.v1.html(html_view, height=520)

                # Prepare structure object for downstream docking (store PDB text directly)
                if pdb_text:
                    protein_structure_for_docking = {
                        "available": True,
                        "structure_type": "predicted",
                        "structure_id": "ESMFOLD",
                        "pdb_text": pdb_text,  # Store PDB text directly instead of data URI
                        "pdb_url": "",  # Empty URL to ensure we use pdb_text
                    }
                    # Store for use in docking tab
                    st.session_state[f"protein_structure_for_docking_{seq_key}"] = protein_structure_for_docking
            else:
                st.warning(struct_result.get("error", "Structure prediction unavailable."))

    # ----------------------
    # Tab 3: Molecular Docking
    # ----------------------
    with predictor_tabs[2]:
        st.subheader("Molecular Docking Simulation")
        st.info("""
        **About Molecular Docking:**
        - 🧪 Simulates protein-ligand interactions
        - 🎯 Uses the app's existing docking pipeline
        - 📊 Provides binding affinity predictions
        - 🔬 Interactive 3D docking visualization
        - 💡 Generic ligand used for demonstration
        """)

        col1, col2 = st.columns([2, 1])
        
        with col1:
            default_mw = st.number_input(
                "Approximate ligand molecular weight (Da)",
                min_value=50.0,
                max_value=1000.0,
                value=300.0,
                step=10.0,
                key=f"protein_predictor_ligand_mw_{seq_key}",
            )
        
        with col2:
            st.caption("")
            st.caption("")
            run_docking = st.button(
                "🚀 Run Docking",
                key=f"protein_predictor_run_docking_{seq_key}",
                type="primary",
                width='stretch'
            )

        docking_result_key = f"protein_predictor_docking_result_{seq_key}"

        if run_docking:
            if "api_client" not in st.session_state:
                st.error("API client not available in session state; cannot run docking simulation.")
            else:
                with st.spinner("🧪 Simulating molecular docking..."):
                    docking_results = st.session_state.api_client.simulate_docking_score(
                        protein_length=len(selected_seq),
                        ligand_mw=float(default_mw),
                    )
                st.session_state[docking_result_key] = docking_results

        docking_results = st.session_state.get(docking_result_key)
        if docking_results and docking_results.get("available"):
            st.success("✅ Docking simulation completed")
            
            st.markdown("---")
            st.markdown("**Docking Scores & Binding Affinity**")
            
            # Reuse existing docking results chart
            fig = ProteinVisualizer.create_docking_results_chart(docking_results)
            st.plotly_chart(fig, width='stretch')

            st.markdown("---")
            
            # Retrieve structure from previous tab if available
            protein_structure_for_docking = st.session_state.get(f"protein_structure_for_docking_{seq_key}")
            
            # Reuse existing 3D docking viewer if we have a predicted structure
            if protein_structure_for_docking:
                # Validate that we have actual PDB data
                pdb_text = protein_structure_for_docking.get('pdb_text', '')
                pdb_url = protein_structure_for_docking.get('pdb_url', '')
                
                if pdb_text or pdb_url:
                    st.markdown("**3D Docking Visualization**")
                    ligand_data = {
                        "name": "Generic ligand",
                        "smiles": "",
                    }
                    try:
                        viewer_html = ProteinVisualizer.create_docking_3d_viewer(
                            protein_structure_for_docking,
                            ligand_data,
                            docking_results,
                            ligand_name="Generic ligand"
                        )
                        st.components.v1.html(viewer_html, height=650, scrolling=False)
                    except Exception as e:
                        st.error(f"⚠️ Error creating 3D visualization: {str(e)}")
                        st.info("💡 Try predicting the structure again in the previous tab.")
                else:
                    st.warning("⚠️ Protein structure data is incomplete.")
                    st.info("💡 Please predict the protein structure in the **Protein Structure Prediction** tab first.")
            else:
                st.info("💡 **To enable 3D docking visualization:**\n\n1. Go to the **Protein Structure Prediction** tab\n2. Click **Predict 3D Structure (ESMFold)**\n3. Wait for the prediction to complete\n4. Return to this tab to view the docking visualization")


# =============================================================================
# WHOLE GENOME SEQUENCING PAGE
# =============================================================================

def render_whole_genome_sequencing_page():
    """
    Render the Whole Genome Sequencing page with sequence-driven disease risk prediction,
    biomarker detection, and personalized research-based recommendations.
    """
    st.title("🧬 Whole Genome Sequencing Analysis")
    
    # Critical disclaimers
    st.markdown("""
    <div class="info-card" style="border-left: 4px solid #ff4444; background-color: #ffe7e7;">
    <strong>⚠️ IMPORTANT DISCLAIMER</strong><br>
    This tool is for <strong>research, educational, and exploratory purposes only</strong>. 
    It does NOT provide medical diagnosis or treatment recommendations. All results are based on 
    computational analysis of genomic sequences and should NOT be used for clinical decision-making. 
    Always consult qualified healthcare providers for medical advice.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
    This module analyzes uploaded genomic sequences to:
    <ul style="margin: 0.5rem 0;">
    <li><strong>Detect mutations:</strong> Identify disease-causing genes and pathogenic variants</li>
    <li><strong>Analyze biomarkers:</strong> Scan for disease-associated markers and protein signatures</li>
    <li><strong>Predict disease risk:</strong> Calculate genetic risk scores based on sequence analysis</li>
    <li><strong>Personalized insights:</strong> Generate research-based recommendations based on biomarker expression</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Genome sequence input
    st.header("📄 Step 1: Input Genome Sequence")
    
    input_method = st.radio(
        "Choose input method:",
        ["Paste Sequence", "Upload FASTA File", "Use Example"],
        horizontal=True
    )
    
    genome_sequence = ""
    
    if input_method == "Paste Sequence":
        genome_sequence = st.text_area(
            "Enter genome sequence (FASTA format or raw DNA sequence):",
            height=200,
            placeholder=">Genome_Sample\nATCGATCGATCGATCGATCGATCG...",
            help="Paste your DNA sequence in FASTA format or as raw nucleotides"
        )
    
    elif input_method == "Upload FASTA File":
        uploaded_file = st.file_uploader(
            "Choose a FASTA file",
            type=['fasta', 'fa', 'fna', 'txt'],
            help="Upload a FASTA file containing the genome sequence"
        )
        if uploaded_file is not None:
            genome_sequence = uploaded_file.read().decode('utf-8')
            st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    else:  # Use Example
        st.info("Using example human genome sequence segment")
        # Use example with disease-related genes
        genome_sequence = """>Example_Human_Sequence_BRCA1_region
ATGATGAATAAAAGAAAAAAAAAATATTGTGAAACAAGATGAGGATGAAAATGAA
AATTGAAAGAAAATAAATGAGAAATTTCAGATAACAAATTTAGGAAGTATAATTAT
ATTTATATTGTATACTGCGATCAACTTAGTAAGTAATGGATGATATAATATAATAA
AGATGAATAAAGAAATGATGATGATATAATAAAGAAAAAGATGATGATGATGAT"""
        st.text_area("Example sequence (BRCA1 region with biomarkers):", value=genome_sequence, height=150, disabled=True)
    
    if genome_sequence:
        # Parse sequence
        if genome_sequence.startswith('>'):
            lines = genome_sequence.split('\n')
            header = lines[0]
            sequence = ''.join(lines[1:])
        else:
            header = ">Genome_Input"
            sequence = genome_sequence
        
        # Clean sequence: remove all whitespace and non-nucleotide characters, convert to uppercase
        sequence = sequence.upper()
        # Keep only valid nucleotides: A, T, C, G, U (RNA), N (unknown), and - (gap)
        sequence = ''.join(c for c in sequence if c in 'ATCGUMN-')
        
        # Validate sequence is not empty
        if not sequence or len(sequence) == 0:
            st.error("❌ No valid DNA sequence found. Please check your input and ensure it contains DNA nucleotides (A, T, C, G).")
            st.stop()
        
        # Display sequence statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sequence Length", f"{len(sequence):,} bp")
        with col2:
            gc_content = ((sequence.count('G') + sequence.count('C')) / len(sequence) * 100) if len(sequence) > 0 else 0
            st.metric("GC Content", f"{gc_content:.2f}%")
        with col3:
            valid_count = len(sequence)
            st.metric("Valid Nucleotides", f"{valid_count:,}")
        with col4:
            st.metric("Quality", "✅ Ready")
        
        st.divider()
        
        # User metadata collection (Step 2)
        st.header("👤 Step 2: Provide Personal Metadata")
        st.markdown("*(Optional but recommended for personalized analysis)*")
        
        col1, col2, col3 = st.columns(3)
        
        user_age = 50
        user_gender = "Unknown"
        user_weight = 70
        
        with col1:
            user_age = st.number_input(
                "Age (years)",
                min_value=18,
                max_value=100,
                value=st.session_state.get('user_age', 50),
                help="Your current age"
            )
        
        with col2:
            user_gender = st.selectbox(
                "Gender",
                ["Unknown", "Male", "Female", "Other"],
                index=st.session_state.get('gender_index', 0)
            )
        
        with col3:
            user_weight = st.number_input(
                "Weight (kg)",
                min_value=30.0,
                max_value=200.0,
                value=st.session_state.get('user_weight', 70.0),
                help="Your body weight"
            )
        
        # Store in session state
        st.session_state.user_age = user_age
        st.session_state.gender_index = ["Unknown", "Male", "Female", "Other"].index(user_gender)
        st.session_state.user_weight = user_weight
        
        # Create user metadata for reference
        user_metadata = {
            'age': user_age,
            'gender': user_gender,
            'weight': user_weight
        }
        
        st.divider()
        
        # Analysis button (Step 3)
        st.header("🔬 Step 3: Run Analysis")
        
        if st.button("▶️ Analyze Sequence", type="primary", width='stretch', key="analyze_genome_btn"):
            with st.spinner("🧬 Running comprehensive genome analysis..."):
                # Initialize genome analysis engine with cache support
                if 'genome_engine' not in st.session_state:
                    st.session_state.genome_engine = GenomeAnalysisEngine(
                        cache_manager=st.session_state.cache_manager
                    )
                
                # Run analysis
                analysis_results = st.session_state.genome_engine.analyze_genome(
                    sequence=sequence,
                    user_metadata=user_metadata
                )
                
                # Store results
                st.session_state.genome_analysis_results = analysis_results
                st.session_state.genome_sequence = sequence
                st.session_state.user_metadata = user_metadata
                st.session_state.show_genome_results = True
                
                time.sleep(1)  # Brief pause for user feedback
                st.success("✅ Analysis complete! Scroll down to view results.")
                st.rerun()
        
        # Display comprehensive analysis results
        if st.session_state.get('show_genome_results') and st.session_state.get('genome_analysis_results'):
            render_genome_analysis_results(
                st.session_state.genome_analysis_results, 
                st.session_state.user_metadata
            )


def render_genome_analysis_results(analysis_results: Dict, user_metadata: Dict):
    """Render comprehensive genome analysis results"""
    st.header("📊 Analysis Results")
    
    # Disclaimers at top of results
    st.markdown("""
    <div class="info-card" style="border-left: 4px solid #ff8800; background-color: #fff5e7;">
    <strong>⚠️ RESEARCH PURPOSES ONLY</strong><br>
    These are <strong>predicted genetic risk indicators</strong> for research and educational purposes only.
    Results are computational predictions, NOT medical diagnoses. Not suitable for clinical decision-making.
    </div>
    """, unsafe_allow_html=True)
    
    # Organize results into tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Sequence Summary",
        "🧬 Mutation Analysis",
        "🔬 Biomarker Detection",
        "⚕️ Disease Risk Assessment",
        "💊 Personalized Insights"
    ])
    
    with tab1:
        render_sequence_summary(analysis_results)
    
    with tab2:
        render_mutation_analysis_results(analysis_results)
    
    with tab3:
        render_biomarker_detection_results(analysis_results)
    
    with tab4:
        render_disease_risk_assessment(analysis_results)
    
    with tab5:
        render_personalized_insights(analysis_results, user_metadata)


def render_sequence_summary(analysis_results: Dict):
    """Render sequence summary statistics"""
    st.subheader("📋 Sequence Summary")
    
    seq_analysis = analysis_results['sequence_analysis']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sequence Length", f"{seq_analysis['length']:,} bp")
    
    with col2:
        st.metric("GC Content", f"{seq_analysis['gc_content']:.2f}%")
    
    with col3:
        st.metric("Valid Nucleotides", f"{seq_analysis['valid_nucleotides']:,}")
    
    with col4:
        quality_pct = (seq_analysis['valid_nucleotides'] / seq_analysis['length'] * 100) if seq_analysis['length'] > 0 else 0
        quality = "Excellent" if quality_pct > 95 else "Good" if quality_pct > 90 else "Fair"
        st.metric("Quality", quality)
    
    st.info(f"""
    **Sequence Information:**
    - Total analyzed: {seq_analysis['length']:,} base pairs
    - Quality assessment indicates {'high-quality sequence suitable for analysis' if quality_pct > 95 else 'acceptable quality for analysis'}
    """)


def render_mutation_analysis_results(analysis_results: Dict):
    """Render mutation analysis results"""
    st.subheader("🧬 Mutation Analysis")
    st.markdown("Detected pathogenic variants and disease-causing genes in your sequence")
    
    mutation_data = analysis_results['mutation_analysis']
    variants = mutation_data['detected_variants']
    
    if not variants:
        st.info("✅ No known pathogenic variants detected in analysis.")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Variants Detected", mutation_data['total_variants'])
        
        with col2:
            st.metric("High-Risk Variants", mutation_data['high_risk_variants'])
        
        with col3:
            st.metric("Detection Confidence", "High")
        
        st.divider()
        
        # Variants table
        if variants:
            st.markdown("#### Detected Variants")
            
            variants_df = pd.DataFrame([
                {
                    'Gene': v['gene'],
                    'Variant ID': v['variant_id'],
                    'Type': v['type'],
                    'Description': v['description'],
                    'Confidence': f"{v['confidence']*100:.0f}%"
                }
                for v in variants
            ])
            
            st.dataframe(variants_df, width='stretch', hide_index=True)
            
            # Detailed variant analysis
            st.markdown("#### Detailed Variant Information")
            
            for variant in variants:
                with st.expander(f"🔍 {variant['gene']} - {variant['variant_id']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        **Gene:** {variant['gene']}  
                        **Variant ID:** {variant['variant_id']}  
                        **Type:** {variant['type']}  
                        **Position:** {variant['position']}
                        """)
                    
                    with col2:
                        confidence = variant['confidence'] * 100
                        st.markdown(f"""
                        **Confidence:** {confidence:.0f}%  
                        **Sequence Match:** {variant['sequence_match']}  
                        **Description:** {variant['description']}
                        """)


def render_biomarker_detection_results(analysis_results: Dict):
    """Render biomarker detection results"""
    st.subheader("🔬 Biomarker Detection")
    st.markdown("Disease-associated biomarkers and protein signatures detected in your sequence")
    
    biomarker_data = analysis_results['biomarker_detection']
    biomarkers = biomarker_data['detected_biomarkers']
    
    if not biomarkers:
        st.info("⚠️ No disease-associated biomarkers detected in this sequence analysis.")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Biomarkers", biomarker_data['total_biomarkers'])
        
        with col2:
            st.metric("Therapeutic Targets", biomarker_data['therapeutic_targets'])
        
        with col3:
            st.metric("Detection Confidence", "Moderate-High")
        
        st.divider()
        
        # Biomarkers table
        if biomarkers:
            st.markdown("#### Detected Biomarkers")
            
            biomarkers_df = pd.DataFrame([
                {
                    'Biomarker': b['name'],
                    'Type': b['type'],
                    'Location': b['location'],
                    'Match Strength': f"{b['match_strength']*100:.0f}%",
                    'Associated Diseases': ', '.join(b['diseases'][:2]),
                    'Clinical Significance': b['significance']
                }
                for b in biomarkers
            ])
            
            st.dataframe(biomarkers_df, width='stretch', hide_index=True)
            
            # Detailed biomarker analysis
            st.markdown("#### Detailed Biomarker Information")
            
            for biomarker in biomarkers:
                with st.expander(f"🔬 {biomarker['name']} ({biomarker['type']})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        **Name:** {biomarker['name']}  
                        **Type:** {biomarker['type']}  
                        **Location:** {biomarker['location']}  
                        **Pattern:** {biomarker['pattern']}
                        """)
                    
                    with col2:
                        match_pct = biomarker['match_strength'] * 100
                        st.markdown(f"""
                        **Match Strength:** {match_pct:.0f}%  
                        **Clinical Significance:** {biomarker['significance']}  
                        **Associated Diseases:**  
                        {', '.join(biomarker['diseases'])}
                        """)
                    
                    # Recommendation
                    st.markdown("**Recommendation:**")
                    st.info(f"This biomarker ({biomarker['name']}) is associated with {', '.join(biomarker['diseases'])}. "
                           f"Further clinical evaluation may be warranted based on biomarker expression.")


def render_disease_risk_assessment(analysis_results: Dict):
    """Render disease risk assessment and associations"""
    st.subheader("⚕️ Disease Risk Assessment")
    st.markdown("Predicted genetic risk for various diseases based on detected variants and biomarkers")
    
    disease_assoc = analysis_results['disease_associations']
    associations = disease_assoc['associations']
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Associations", len(associations))
    
    with col2:
        st.metric("High Confidence", disease_assoc['high_confidence'])
    
    with col3:
        st.metric("Moderate Confidence", disease_assoc['moderate_confidence'])
    
    with col4:
        st.metric("Analysis Type", "Research")
    
    st.divider()
    
    if not associations:
        st.info("✅ No significant disease associations detected based on current sequence analysis.")
    else:
        st.markdown("#### Disease Risk Rankings")
        
        # Sort by risk score
        sorted_assoc = sorted(associations, key=lambda x: x['risk_score'], reverse=True)
        
        # Create visualization
        diseases = [a['disease'] for a in sorted_assoc[:10]]
        risks = [a['risk_score'] for a in sorted_assoc[:10]]
        confidences = [a['confidence'] for a in sorted_assoc[:10]]
        
        # Color by confidence
        color_map = {
            'Very High': '#ff4444',
            'High': '#ff8844',
            'Moderate': '#ffaa44',
            'Low': '#ffcc44',
            'Very Low': '#cccccc'
        }
        colors = [color_map.get(c, '#cccccc') for c in confidences]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=diseases,
            y=risks,
            marker=dict(color=colors),
            text=[f"{r:.1f}%" for r in risks],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Risk: %{y:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="Predicted Disease Risk Scores",
            xaxis_title="Disease",
            yaxis_title="Risk Score (%)",
            yaxis=dict(range=[0, 100]),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, width='stretch')
        
        st.divider()
        
        # Detailed disease information
        st.markdown("#### Detailed Disease Risk Profiles")
        
        for assoc in sorted_assoc:
            disease = assoc['disease']
            risk = assoc['risk_score']
            confidence = assoc['confidence']
            
            # Risk level indicator
            if risk > 50:
                risk_level = "🔴 High"
                risk_color = "#ff4444"
            elif risk > 25:
                risk_level = "🟡 Moderate"
                risk_color = "#ffaa00"
            else:
                risk_level = "🟢 Low"
                risk_color = "#44ff44"
            
            with st.expander(f"{risk_level} {disease} - {confidence} Confidence ({risk:.1f}%)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **Disease:** {disease}  
                    **Predicted Risk:** {risk:.1f}%  
                    **Confidence Level:** {confidence}  
                    **Population Baseline:** ~{assoc['prevalence']*100:.1f}%
                    """)
                
                with col2:
                    st.markdown(f"""
                    **Inheritance Pattern:** {assoc['inheritance']}  
                    **Detected Variants:** {assoc['variants']}  
                    **Detected Biomarkers:** {assoc['biomarkers']}  
                    **Analysis Type:** Research-based prediction
                    """)
                
                st.warning(
                    "⚠️ **Important:** This is a computational prediction based on sequence analysis. "
                    "It does NOT constitute a medical diagnosis and should not be used for clinical decision-making. "
                    "Consult with a healthcare provider for medical advice."
                )


def render_personalized_insights(analysis_results: Dict, user_metadata: Dict):
    """Render personalized research-based insights"""
    st.subheader("💊 Personalized Insights")
    st.markdown("""
    <div class="info-card">
    Research-based insights generated from detected biomarkers, gene expression patterns, 
    and your personal characteristics.
    </div>
    """, unsafe_allow_html=True)
    
    recommendations = analysis_results['recommendations']
    
    # Personal summary
    st.markdown("#### Your Profile Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Age", f"{user_metadata.get('age', 'N/A')} years")
    
    with col2:
        st.metric("Gender", user_metadata.get('gender', 'N/A'))
    
    with col3:
        st.metric("Weight", f"{user_metadata.get('weight', 'N/A')} kg")
    
    with col4:
        st.metric("Analysis", "Personalized")
    
    st.divider()
    
    # High priority recommendations
    if recommendations.get('high_priority'):
        st.markdown("#### 🔴 High Priority Insights")
        st.markdown("Based on high-confidence genetic indicators detected in your sequence:")
        
        for rec in recommendations['high_priority'][:5]:
            with st.expander(f"🔬 {rec['disease']} - {rec['treatment']}", expanded=False):
                st.markdown(f"""
                **Disease:** {rec['disease']}  
                **Category:** {rec['category']}  
                **Therapeutic Consideration:** {rec['treatment']}  
                **Indication:** {rec['indication']}  
                **Confidence:** {rec['confidence']}
                
                **Notes:** {rec['notes']}
                
                ⚠️ **Disclaimer:** These are research-based considerations only, not medical recommendations.
                """)
    
    # Moderate priority recommendations
    if recommendations.get('moderate_priority'):
        st.markdown("#### 🟡 Moderate Priority Insights")
        
        for rec in recommendations['moderate_priority'][:5]:
            st.markdown(f"""
            - **{rec['disease']}** (Biomarker: {rec.get('biomarker', 'N/A')}): {rec['treatment']}
            """)
    
    st.divider()
    
    # Lifestyle recommendations
    if recommendations.get('lifestyle'):
        st.markdown("#### 🏃 Lifestyle Recommendations")
        st.markdown("*Based on detected genetic risk factors and biomarkers:*")
        
        cols = st.columns(2)
        for idx, lifestyle in enumerate(recommendations['lifestyle']):
            with cols[idx % 2]:
                st.markdown(f"✅ {lifestyle}")
    
    # Monitoring recommendations
    if recommendations.get('monitoring'):
        st.markdown("#### 📋 Recommended Health Monitoring")
        st.markdown("*Consider discussing with healthcare providers:*")
        
        for monitoring in recommendations['monitoring']:
            st.markdown(f"📌 {monitoring}")
    
    st.divider()
    
    # Pharmacogenomic guidance
    if recommendations.get('pharmacogenomics'):
        st.markdown("#### 💊 Pharmacogenomic Guidance")
        st.markdown("*How your genetic variants may affect drug metabolism:*")
        
        for pharm in recommendations['pharmacogenomics']:
            with st.expander(f"🧬 {pharm['gene']} ({pharm['phenotype']})"):
                st.markdown(f"""
                **Enzyme:** {pharm['enzyme']}  
                **Your Phenotype:** {pharm['phenotype']}  
                **Affected Drugs:** {', '.join(pharm['affected_drugs'])}
                
                **Action:** {pharm['action']}  
                **Risk:** {pharm['risk']}
                """)
    
    st.divider()
    
    # Important disclaimers
    st.markdown("#### ⚠️ Important Disclaimers")
    
    for disclaimer in recommendations['disclaimers']:
        st.warning(disclaimer)
    
    st.info("""
    **About This Analysis:**
    - This is a computational analysis for research and educational purposes
    - Results should NOT be used for medical decision-making
    - All recommendations are research-based and NOT medical prescriptions
    - Always consult qualified healthcare providers before making health decisions
    - Genetic testing and counseling are recommended for confirmation
    """)


def render_predictive_risk_calculator(genome_data):
    """Render the Predictive Risk Calculator section"""
    st.subheader("🎯 Predictive Risk Calculator")
    st.markdown("""
    <div class="info-card">
    Calculate disease risk based on genetic variants and population statistics.
    </div>
    """, unsafe_allow_html=True)
    
    # Simulate variant detection
    detected_variants = [
        {
            'gene': 'BRCA1',
            'variant': 'c.68_69delAG',
            'type': 'Pathogenic',
            'diseases': ['Breast Cancer', 'Ovarian Cancer'],
            'risk_increase': 65.0,
            'population_freq': 0.0006
        },
        {
            'gene': 'APOE',
            'variant': 'ε4 allele',
            'type': 'Risk Factor',
            'diseases': ['Alzheimer\'s Disease'],
            'risk_increase': 12.0,
            'population_freq': 0.15
        },
        {
            'gene': 'CFTR',
            'variant': 'F508del',
            'type': 'Carrier',
            'diseases': ['Cystic Fibrosis'],
            'risk_increase': 2.0,
            'population_freq': 0.03
        },
        {
            'gene': 'HFE',
            'variant': 'C282Y',
            'type': 'Risk Factor',
            'diseases': ['Hemochromatosis'],
            'risk_increase': 8.5,
            'population_freq': 0.06
        },
        {
            'gene': 'FTO',
            'variant': 'rs9939609',
            'type': 'Polygenic',
            'diseases': ['Type 2 Diabetes', 'Obesity'],
            'risk_increase': 3.2,
            'population_freq': 0.42
        }
    ]
    
    # Risk summary
    st.markdown("### Risk Summary")
    
    risk_df = pd.DataFrame([
        {
            'Disease': ', '.join(v['diseases']),
            'Gene': v['gene'],
            'Variant': v['variant'],
            'Type': v['type'],
            'Risk Increase': f"{v['risk_increase']}%",
            'Population Frequency': f"{v['population_freq']*100:.2f}%"
        }
        for v in detected_variants
    ])
    
    st.dataframe(risk_df, width='stretch', hide_index=True)
    
    # Visualize risk levels
    st.markdown("### Disease Risk Levels")
    
    # Calculate aggregate risk scores
    disease_risks = {}
    for variant in detected_variants:
        for disease in variant['diseases']:
            if disease not in disease_risks:
                disease_risks[disease] = 10.0  # baseline
            disease_risks[disease] += variant['risk_increase'] / len(variant['diseases'])
    
    # Create visualization
    fig = go.Figure()
    
    diseases = list(disease_risks.keys())
    risks = [min(disease_risks[d], 100) for d in diseases]
    colors = ['#ff4444' if r > 40 else '#ffaa00' if r > 20 else '#44ff44' for r in risks]
    
    fig.add_trace(go.Bar(
        x=diseases,
        y=risks,
        marker=dict(color=colors),
        text=[f"{r:.1f}%" for r in risks],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Predicted Disease Risk Levels",
        xaxis_title="Disease",
        yaxis_title="Risk Level (%)",
        yaxis=dict(range=[0, 100]),
        height=400
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Risk interpretation
    st.markdown("### Risk Interpretation")
    
    for disease, risk in disease_risks.items():
        risk_level = "High" if risk > 40 else "Moderate" if risk > 20 else "Low"
        risk_color = "🔴" if risk > 40 else "🟡" if risk > 20 else "🟢"
        
        with st.expander(f"{risk_color} {disease} - {risk_level} Risk ({risk:.1f}%)"):
            st.markdown(f"""
            **Risk Level:** {risk:.1f}% (Population average: ~10%)
            
            **Recommendations:**
            - {'Regular screening recommended' if risk > 40 else 'Maintain healthy lifestyle'}
            - {'Consult with genetic counselor' if risk > 40 else 'Standard preventive measures'}
            - {'Consider preventive strategies' if risk > 20 else 'Continue monitoring'}
            
            **Contributing Variants:**
            """)
            


def render_drugs_clinical_trials_page():
    """Render the Drugs & Clinical Trials page"""
    st.title("💊 Drugs & Clinical Trials")
    st.markdown("""
    <div class="info-card">
    Search for drug information, FDA approvals, clinical trials, and explore drug repurposing opportunities.
    </div>
    """, unsafe_allow_html=True)
    
    # Drug input
    st.header("🔍 Drug Search")
    
    # Get current drug from session state or show input
    current_drug = st.session_state.get('current_drug', '')
    
    drug_name = st.text_input(
        "Enter Drug Name:",
        value=current_drug,  # Keep previous search if exists
        placeholder="e.g., Aspirin, Imatinib, Metformin",
        help="Enter the name of a drug to search for information",
        key="drug_search_input"
    )
    
    if drug_name and drug_name != current_drug:
        # NEW SEARCH - Clear old data
        st.session_state.current_drug = drug_name.strip()
        st.session_state.repurposing_results = None  # Clear old results
        st.session_state.show_drug_analysis = False
        st.rerun()
    
    if st.session_state.get('current_drug'):
        drug_name = st.session_state.current_drug
        
        # Action buttons
        col1, col2 = st.columns([1, 1])
        
        with col1:
            run_analysis = st.button(
                "🔬 Analyze Repurposing Opportunities",
                type="primary",
                width='stretch',
                key="analyze_drug_button"
            )
        
        with col2:
            clear_search = st.button(
                "🔄 New Search",
                width='stretch',
                key="clear_drug_search"
            )
        
        if clear_search:
            st.session_state.current_drug = None
            st.session_state.repurposing_results = None
            st.session_state.show_drug_analysis = False
            st.rerun()
        
        if run_analysis:
            with st.spinner(f"🔍 Analyzing {drug_name} across clinical trials, FDA database, and repurposing networks..."):
                # Always fetch fresh data - don't use cached repurposing_results
                repurposing_report = _generate_repurposing_report_data(
                    drug_name,
                    st.session_state.api_client
                )
                st.session_state.repurposing_results = repurposing_report
                st.session_state.show_drug_analysis = True
                st.success(f"✅ Analysis complete for {drug_name}!")
                st.rerun()
        
        # Display drug information if analysis was run
        if st.session_state.get('show_drug_analysis') and st.session_state.get('repurposing_results'):
            st.divider()
            st.header(f"📋 {drug_name} - Complete Profile")
            
            # Tabs for different sections
            tab1, tab2, tab3 = st.tabs([
                "📜 FDA-Approved Drugs & Clinical Trials",
                "🔄 Drug Repurposing Engine",
                "📊 Detailed Information"
            ])
            
            with tab1:
                render_fda_clinical_trials(drug_name, st.session_state.repurposing_results)
            
            with tab2:
                render_drug_repurposing_section(drug_name, st.session_state.repurposing_results)
            
            with tab3:
                render_drug_detailed_info(drug_name)


def render_fda_clinical_trials(drug_name, report_data=None):
    """Render FDA approval status and clinical trials information"""
    st.subheader("📜 FDA Approval Status & Clinical Trials")
    
    # Use provided report data or show message
    if not report_data:
        st.info("Click 'Analyze Repurposing Opportunities' to fetch clinical trial data for this drug")
        return
    
    # Get clinical trials from report
    clinical_trials = report_data.get('clinical_trials', [])
    
    if not clinical_trials:
        st.warning(f"⚠️ No clinical trials found for {drug_name} in ClinicalTrials.gov")
        st.info("""
        This could mean:
        - The drug is not currently in active clinical trials
        - The drug name may need to be spelled differently
        - The drug may be an older medication with no new trials
        
        **To search manually:** Visit [ClinicalTrials.gov](https://clinicaltrials.gov/)
        """)
    else:
        st.success(f"✅ Found {len(clinical_trials)} clinical trial(s) for {drug_name}")
        
        st.markdown("---")
        st.markdown("### 🔬 Clinical Trials")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trial_status = st.multiselect(
                "Trial Status:",
                ["Recruiting", "Active, not recruiting", "Completed", "Terminated", "RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED", "TERMINATED"],
                default=["Recruiting", "RECRUITING", "Active, not recruiting", "ACTIVE_NOT_RECRUITING"],
                key="clinical_trial_status_filter"
            )
        
        with col2:
            trial_phase = st.multiselect(
                "Phase:",
                ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "PHASE_1", "PHASE_2", "PHASE_3", "PHASE_4"],
                default=["Phase 2", "Phase 3", "PHASE_2", "PHASE_3"],
                key="clinical_trial_phase_filter"
            )
        
        with col3:
            # Get unique conditions from trials
            all_conditions = set()
            for trial in clinical_trials:
                cond = trial.get('condition', '')
                if cond and cond != 'N/A':
                    all_conditions.add(str(cond)[:50])  # Truncate long conditions
            
            condition = st.selectbox(
                "Filter by Condition:",
                ["All"] + sorted(list(all_conditions)),
                key="clinical_trial_condition_filter"
            )
        
        st.markdown("---")
        
        # Display trials
        for trial in clinical_trials:
            nct_id = trial.get('nct_id') or trial.get('trial_id', 'N/A')
            trial_status_val = trial.get('status', 'Unknown')
            trial_phase_val = trial.get('phase', 'N/A')
            trial_condition = trial.get('condition', 'N/A')
            
            # Apply filters
            status_match = any(s.upper() in str(trial_status_val).upper() for s in trial_status) if trial_status else True
            phase_match = any(p.upper() in str(trial_phase_val).upper() for p in trial_phase) if trial_phase else True
            condition_match = condition == "All" or condition.lower() in str(trial_condition).lower()
            
            if not (status_match and phase_match and condition_match):
                continue
            
            status_color = "#28a745" if "COMPLETED" in str(trial_status_val).upper() else "#ff9800" if "RECRUITING" in str(trial_status_val).upper() else "#dc3545"
            phase_icon = "✅" if "PHASE_3" in str(trial_phase_val).upper() else "🔄" if "PHASE_2" in str(trial_phase_val).upper() else "🧪"
            
            with st.expander(
                f"{phase_icon} **{trial.get('title', 'N/A')[:70]}...** | {trial_status_val} | NCT: {nct_id}",
                expanded=False
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Trial Information:**")
                    st.markdown(f"""
                    - **Trial ID:** {nct_id}
                    - **Phase:** {trial_phase_val}
                    - **Status:** {trial_status_val}
                    - **Start Date:** {trial.get('start_date', trial.get('start_year', 'N/A'))}
                    - **Enrolled Patients:** {trial.get('enrolled', 'N/A')}
                    - **Sponsor:** {trial.get('sponsor', 'N/A')}
                    """)
                    
                    st.markdown("**Study Details:**")
                    st.markdown(f"""
                    - **Condition:** {trial_condition}
                    - **Intervention:** {trial.get('intervention', drug_name)}
                    - **Primary Outcome:** {trial.get('primary_outcome', 'N/A')}
                    """)
                
                with col2:
                    st.markdown(f"""
                    <div style="background-color: {status_color}22; border: 2px solid {status_color}; padding: 1rem; border-radius: 8px; text-align: center;">
                        <h4 style="margin:0; color:{status_color};">{trial_status_val}</h4>
                        <p style="margin:5px 0 0 0;">Study Status</p>
                    </div>
                    <div style="background-color: #f5f5f5; padding: 0.5rem; border-radius: 4px; text-align: center; margin-top: 0.5rem;">
                        <strong>{trial_phase_val}</strong>
                    </div>
                    """, unsafe_allow_html=True)

                trial_url = trial.get('url', '')
                if trial_url:
                    st.markdown(f"[View on ClinicalTrials.gov]({trial_url})")
                else:
                    clinicaltrials_url = _build_clinicaltrials_url(nct_id)
                    if clinicaltrials_url:
                        st.markdown(f"[View on ClinicalTrials.gov]({clinicaltrials_url})")
    
    st.divider()
    # Fetch clinical trials data from ClinicalTrials.gov (verified NCT IDs)
    clinical_trials = []
    if 'api_client' in st.session_state:
        try:
            try:
                clinical_trials = asyncio.run(
                    st.session_state.api_client.fetch_clinical_trials_by_drug(drug_name)
                )
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    clinical_trials = loop.run_until_complete(
                        st.session_state.api_client.fetch_clinical_trials_by_drug(drug_name)
                    )
                finally:
                    loop.close()
        except Exception as e:
            print(f"ClinicalTrials.gov fetch error: {str(e)}", file=sys.stderr)
    else:
        st.warning("API client not available; cannot fetch ClinicalTrials.gov data.")

    valid_trials = []
    invalid_trials = []
    for trial in clinical_trials:
        nct_id = _extract_nct_id(trial)
        if not nct_id:
            invalid_trials.append(trial)
            continue
        trial["nct_id"] = nct_id
        valid_trials.append(trial)

    if invalid_trials:
        print(
            f"ClinicalTrials.gov: filtered {len(invalid_trials)} invalid entries for {drug_name}",
            file=sys.stderr,
        )
        st.caption("Some trial entries were excluded due to missing or invalid NCT IDs.")

    clinical_trials = valid_trials
    
    st.markdown(f"**Found {len(clinical_trials)} clinical trials**")
    if not clinical_trials:
        st.info("No verified ClinicalTrials.gov entries found for this drug.")
        encoded_drug = urllib.parse.quote_plus(drug_name)
        st.markdown(
            f"<a href=\"https://clinicaltrials.gov/search?term={encoded_drug}\" target=\"_blank\" rel=\"noopener noreferrer\">Search on ClinicalTrials.gov</a>",
            unsafe_allow_html=True,
        )
    
    for trial in clinical_trials:
        status_key = str(trial.get("status", "")).upper()
        status_color = "#44ff44" if status_key == "RECRUITING" else "#4444ff" if status_key == "ACTIVE_NOT_RECRUITING" else "#888888"
        nct_id = _extract_nct_id(trial)
        display_nct = nct_id or trial.get("nct_id", "NCT ID unavailable")
        display_status = _format_status(trial.get("status"))
        display_phase = _format_phase(trial.get("phase"))
        
        with st.expander(f"🔬 {display_nct} - {trial.get('title', 'N/A')}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                **Title:** {trial['title']}  
                **Condition:** {', '.join(trial.get('conditions', [])) or trial.get('condition', 'N/A')}  
                **Sponsor:** {trial.get('sponsor', 'N/A')}  
                **Locations:** {trial.get('locations', 'N/A')}
                """)
            
            with col2:
                st.markdown(f"""
                <div style="background-color: {status_color}22; border: 2px solid {status_color}; padding: 0.5rem; border-radius: 4px; margin-bottom: 0.5rem;">
                    <strong>Status:</strong> {display_status}
                </div>
                <div style="background-color: #f0f0f0; padding: 0.5rem; border-radius: 4px; margin-bottom: 0.5rem;">
                    <strong>Phase:</strong> {display_phase}
                </div>
                <div style="background-color: #f0f0f0; padding: 0.5rem; border-radius: 4px;">
                    <strong>Enrollment:</strong> {trial.get('enrollment', 'N/A')} participants
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"**Start Date:** {trial.get('start_date', 'N/A')}")
            clinicaltrials_url = _build_clinicaltrials_url(nct_id)
            if clinicaltrials_url:
                st.markdown(
                    f"<a href=\"{clinicaltrials_url}\" target=\"_blank\" rel=\"noopener noreferrer\">View on ClinicalTrials.gov</a>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("ClinicalTrials.gov link unavailable (missing or invalid NCT ID).")
    
    # Trial statistics
    st.divider()
    st.markdown("### Trial Statistics")
    
    # Create visualization
    status_counts = {}
    phase_counts = {}
    
    for trial in clinical_trials:
        status_counts[trial['status']] = status_counts.get(trial['status'], 0) + 1
        phase_counts[trial['phase']] = phase_counts.get(trial['phase'], 0) + 1
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_status = go.Figure(data=[go.Pie(
            labels=list(status_counts.keys()),
            values=list(status_counts.values()),
            hole=0.4
        )])
        fig_status.update_layout(title="Trials by Status", height=300)
        st.plotly_chart(fig_status, width='stretch')
    
    with col2:
        fig_phase = go.Figure(data=[go.Bar(
            x=list(phase_counts.keys()),
            y=list(phase_counts.values()),
            marker_color='#1f77b4'
        )])
        fig_phase.update_layout(title="Trials by Phase", height=300)
        st.plotly_chart(fig_phase, width='stretch')


def _generate_detailed_text_report(drug_name: str, report_data: dict) -> str:
    """Generate a comprehensive detailed text report for drug repurposing analysis"""
    from datetime import datetime
    
    report = []
    
    # Header
    report.append("=" * 80)
    report.append("COMPREHENSIVE DRUG REPURPOSING ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Report metadata
    report.append("REPORT METADATA")
    report.append("-" * 80)
    report.append(f"Drug Name: {drug_name}")
    report.append(f"Report Generated: {report_data['metadata']['report_date']}")
    report.append(f"Analysis Type: Computational Network Analysis + Clinical Evidence Review")
    report.append("")
    
    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 80)
    approved_count = len(report_data.get('approved_drugs', []))
    trials_count = len(report_data.get('clinical_trials', []))
    repurposing_count = len(report_data.get('repurposing_opportunities', []))
    
    report.append(f"This comprehensive analysis examines {drug_name} for potential therapeutic")
    report.append(f"applications beyond current approved indications.")
    report.append("")
    report.append(f"Analysis Summary:")
    report.append(f"  • Current FDA-Approved Indications: {approved_count}")
    report.append(f"  • Associated Clinical Trials: {trials_count}")
    report.append(f"  • Identified Repurposing Opportunities: {repurposing_count}")
    report.append("")
    
    # Section 1: FDA-Approved Indications
    report.append("SECTION 1: FDA-APPROVED INDICATIONS & CURRENT CLINICAL USE")
    report.append("=" * 80)
    report.append("")
    
    approved_drugs = report_data.get('approved_drugs', [])
    if approved_drugs:
        for i, drug in enumerate(approved_drugs, 1):
            report.append(f"{i}. {drug['indication']}")
            report.append("-" * 80)
            report.append(f"   Drug Name:           {drug['name']}")
            report.append(f"   Approval Date:       {drug['approval_date']}")
            report.append(f"   Status:              {drug['status']}")
            report.append(f"   DrugBank ID:         {drug.get('drugbank_id', 'N/A')}")
            report.append(f"   PubChem ID:          {drug.get('pubchem_id', 'N/A')}")
            report.append(f"   Confidence Score:    {drug['confidence_score']}%")
            report.append("")
            report.append(f"   Mechanism of Action:")
            report.append(f"   {drug['mechanism']}")
            report.append("")
            report.append(f"   Target Proteins:")
            for target in drug.get('target_proteins', []):
                report.append(f"   • {target}")
            report.append("")
            report.append(f"   Evidence Source:     {drug['evidence_source']}")
            report.append("")
    else:
        report.append("No approved indications found.")
        report.append("")
    
    # Section 2: Clinical Trials
    report.append("SECTION 2: ASSOCIATED CLINICAL TRIALS")
    report.append("=" * 80)
    report.append("")
    
    clinical_trials = report_data.get('clinical_trials', [])
    if clinical_trials:
        for i, trial in enumerate(clinical_trials, 1):
            nct_id = _extract_nct_id(trial)
            report.append(f"{i}. {trial['title']}")
            report.append("-" * 80)
            report.append(f"   Trial ID (NCT):      {nct_id or trial.get('trial_id', 'N/A')}")
            report.append(f"   Phase:               {trial['phase']}")
            report.append(f"   Status:              {trial['status']}")
            report.append(f"   Condition:           {trial['condition']}")
            report.append(f"   Start Year:          {trial['start_year']}")
            report.append(f"   Enrolled Patients:   {trial.get('enrolled', 'N/A')}")
            report.append(f"   Sponsor:             {trial.get('sponsor', 'N/A')}")
            report.append(f"   Intervention:        {trial.get('intervention', 'N/A')}")
            report.append(f"   Primary Outcome:     {trial.get('primary_outcome', 'N/A')}")
            clinicaltrials_url = _build_clinicaltrials_url(nct_id)
            report.append(f"   ClinicalTrials URL:  {clinicaltrials_url or 'N/A'}")
            report.append("")
    else:
        report.append("No associated clinical trials found.")
        report.append("")
    
    # Section 3: Repurposing Opportunities
    report.append("SECTION 3: IDENTIFIED REPURPOSING OPPORTUNITIES")
    report.append("=" * 80)
    report.append("")
    
    repurposing_opps = report_data.get('repurposing_opportunities', [])
    if repurposing_opps:
        # Sort by confidence score (descending)
        sorted_opps = sorted(repurposing_opps, key=lambda x: x['confidence'], reverse=True)
        
        for i, opp in enumerate(sorted_opps, 1):
            report.append(f"{i}. {opp['disease']}")
            report.append("-" * 80)
            report.append(f"   Confidence Score:    {opp['confidence']:.1f}%")
            report.append(f"   Priority Level:      {opp['priority']}")
            report.append(f"   Status:              {opp['status']}")
            report.append("")
            
            report.append(f"   PROPOSED MECHANISM OF ACTION:")
            report.append(f"   {opp['mechanism']}")
            report.append("")
            
            report.append(f"   CLINICAL RATIONALE:")
            report.append(f"   {opp['clinical_rationale']}")
            report.append("")
            
            report.append(f"   SUPPORTING EVIDENCE:")
            for j, evidence in enumerate(opp.get('evidence', []), 1):
                report.append(f"   {j}. {evidence}")
            report.append("")
            
            report.append(f"   AFFECTED BIOLOGICAL PATHWAYS:")
            for pathway in opp.get('affected_pathways', []):
                report.append(f"   • {pathway}")
            report.append("")
            
            report.append(f"   NETWORK ANALYSIS:")
            report.append(f"   • Shared Target Proteins: {opp.get('shared_targets', 'N/A')}")
            report.append(f"   • Supporting Publications: {opp.get('supporting_publications', 'N/A')}")
            report.append("")
    else:
        report.append("No repurposing opportunities identified.")
        report.append("")
    
    # Section 4: Analysis Methodology
    report.append("SECTION 4: ANALYSIS METHODOLOGY")
    report.append("=" * 80)
    report.append("")
    report.append("This analysis was conducted using the following approach:")
    report.append("")
    report.append("1. BIOLOGICAL NETWORK ANALYSIS")
    report.append("   • Drug target identification and protein interaction networks")
    report.append("   • Pathway enrichment analysis")
    report.append("   • Disease similarity scoring")
    report.append("")
    report.append("2. CLINICAL TRIAL DATA INTEGRATION")
    report.append("   • Mining of ClinicalTrials.gov for past and ongoing trials")
    report.append("   • Analysis of trial outcomes and conditions")
    report.append("")
    report.append("3. LITERATURE-BASED EVIDENCE SYNTHESIS")
    report.append("   • PubMed literature mining for mechanistic evidence")
    report.append("   • Case reports and observational studies review")
    report.append("   • Preclinical model data integration")
    report.append("")
    report.append("4. CONFIDENCE SCORING")
    report.append("   • Multi-evidence confidence calculation (0-100%)")
    report.append("   • High (>70%): Strong mechanistic and clinical evidence")
    report.append("   • Moderate (50-70%): Reasonable mechanistic basis with some evidence")
    report.append("   • Low (<50%): Preliminary evidence or speculative indication")
    report.append("")
    
    # Section 5: Important Disclaimers
    report.append("SECTION 5: IMPORTANT DISCLAIMERS & LIMITATIONS")
    report.append("=" * 80)
    report.append("")
    report.append("DISCLAIMER:")
    report.append("This analysis is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.")
    report.append("")
    report.append("⚠️  IMPORTANT:")
    report.append("• This computational analysis does NOT constitute medical advice")
    report.append("• Results should NOT be used for clinical decision-making")
    report.append("• All repurposing suggestions are EXPERIMENTAL and require")
    report.append("  rigorous clinical validation")
    report.append("• Consult qualified healthcare providers before any medical decisions")
    report.append("• All proposed uses require appropriate clinical trial design and")
    report.append("  regulatory approval")
    report.append("")
    report.append("LIMITATIONS:")
    report.append("• Analysis based on computational predictions and published literature")
    report.append("• Confidence scores reflect available evidence quality, not efficacy")
    report.append("• Drug safety and pharmacokinetics not fully addressed here")
    report.append("• Patient-specific factors (genetics, comorbidities) not considered")
    report.append("• Dosing recommendations NOT provided in this analysis")
    report.append("• Clinical trial phase-dependent safety concerns may apply")
    report.append("")
    
    # Section 6: Recommendations for Further Investigation
    report.append("SECTION 6: RECOMMENDATIONS FOR FURTHER INVESTIGATION")
    report.append("=" * 80)
    report.append("")
    
    # Identify high-priority opportunities
    high_priority = [opp for opp in repurposing_opps if opp['priority'] == 'High']
    if high_priority:
        report.append("PRIORITY ACTIONS (High Confidence Opportunities):")
        for opp in high_priority:
            report.append(f"• {opp['disease']} ({opp['confidence']:.1f}% confidence)")
            report.append(f"  - Recommended: Systematic literature review + preclinical validation")
            report.append(f"  - Next step: Clinical trial design feasibility assessment")
            report.append("")
    
    report.append("GENERAL RECOMMENDATIONS:")
    report.append("1. Validate findings through independent literature review")
    report.append("2. Conduct rigorous preclinical studies in relevant disease models")
    report.append("3. Assess pharmacokinetic/pharmacodynamic properties for new indications")
    report.append("4. Evaluate potential off-target effects and safety concerns")
    report.append("5. Design properly controlled clinical trials for validation")
    report.append("6. Consult with clinical experts in target disease areas")
    report.append("7. Consider existing regulatory pathways (fast-track, breakthrough therapy)")
    report.append("")
    
    # Footer
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("For more information, contact the research team or visit:")
    report.append("• DrugBank: https://www.drugbank.ca/")
    report.append("• ClinicalTrials.gov: https://clinicaltrials.gov/")
    report.append("• PubMed: https://pubmed.ncbi.nlm.nih.gov/")
    report.append("• FDA Drug Approvals: https://www.fda.gov/drugs/")
    report.append("")
    
    return "\n".join(report)


def _generate_repurposing_report_data(drug_name, api_client=None):
    """Generate comprehensive repurposing report data with drugs and clinical trials - DYNAMIC PER DRUG"""
    # Normalize drug name for consistent lookups
    drug_name_normalized = drug_name.strip().lower()
    
    report_data = {
        'metadata': {
            'drug_name': drug_name,
            'report_date': datetime.now().isoformat(),
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        },
        'approved_drugs': [],
        'clinical_trials': [],
        'repurposing_opportunities': []
    }
    
    # ========== FETCH CLINICAL TRIALS DYNAMICALLY ==========
    report_data['clinical_trials'] = []
    if api_client is not None:
        try:
            try:
                raw_trials = asyncio.run(
                    api_client.fetch_clinical_trials_by_drug(drug_name)
                )
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    raw_trials = loop.run_until_complete(
                        api_client.fetch_clinical_trials_by_drug(drug_name)
                    )
                finally:
                    loop.close()

            for trial in raw_trials:
                nct_id = _extract_nct_id(trial)
                if not nct_id:
                    continue

                start_year = "N/A"
                start_date = trial.get("start_date")
                if start_date and isinstance(start_date, str) and len(start_date) >= 4:
                    start_year = start_date[:4]

                conditions = trial.get('conditions', [])
                condition_value = ", ".join(conditions) if conditions else trial.get('condition', 'N/A')

                report_data['clinical_trials'].append({
                    'trial_id': nct_id,
                    'nct_id': nct_id,
                    'title': trial.get('title', 'N/A'),
                    'phase': trial.get('phase', 'N/A'),
                    'status': trial.get('status', 'Unknown'),
                    'condition': condition_value,
                    'intervention': trial.get('intervention_name', trial.get('drugs', drug_name)),
                    'sponsor': trial.get('sponsor', 'N/A'),
                    'start_year': start_year,
                    'start_date': trial.get('start_date', 'N/A'),
                    'enrolled': trial.get('enrollment', 'N/A'),
                    'primary_outcome': trial.get('primary_outcome', 'N/A'),
                    'url': trial.get('url', '')
                })
        except Exception as e:
            st.warning(f"⚠️ Could not fetch clinical trials for {drug_name}: {str(e)}")
            report_data['clinical_trials'] = []
    
    # ========== GENERATE DYNAMIC APPROVED DRUGS SECTION ==========
    # Create a generic approved drug entry based on the searched drug name
    
    # Fetch drug metadata (DrugBank ID, PubChem ID, status) from database/ChEMBL
    from api_client import get_drug_metadata
    drug_metadata = get_drug_metadata(drug_name)
    
    # Determine confidence score based on data availability
    base_confidence = 0
    if drug_metadata.get('drugbank_id') != 'N/A':
        base_confidence += 25  # Has DrugBank ID
    if drug_metadata.get('pubchem_id') != 'N/A':
        base_confidence += 25  # Has PubChem ID
    if drug_metadata.get('status') != 'Status Unknown - Query FDA Database':
        base_confidence += 25  # Has known status
    if report_data['clinical_trials']:
        base_confidence += 25  # Has clinical trials
    
    approved_entry = {
        'name': drug_name,
        'drug_id': 'N/A',
        'drugbank_id': drug_metadata.get('drugbank_id', 'N/A'),
        'pubchem_id': drug_metadata.get('pubchem_id', 'N/A'),
        'indication': f'Search Results for {drug_name}',
        'approval_date': 'Contact DrugBank for details',
        'mechanism': f'Consult FDA and DrugBank databases for {drug_name} mechanism of action',
        'target_proteins': [],
        'evidence_source': 'ChEMBL + ClinicalTrials.gov + DrugBank',
        'confidence_score': min(100, base_confidence),  # Cap at 100%
        'status': drug_metadata.get('status', 'Status Unknown - Query FDA Database')
    }
    
    # If we found clinical trials, update indication
    if report_data['clinical_trials']:
        approved_entry['indication'] = f'{len(report_data["clinical_trials"])} active clinical trials found'
    
    report_data['approved_drugs'] = [approved_entry]
    
    # ========== GENERATE DYNAMIC REPURPOSING OPPORTUNITIES ==========
    # Create opportunities based on actual data
    if report_data['clinical_trials'] and len(report_data['clinical_trials']) > 0:
        # If we have clinical trials, create opportunities from trial conditions
        trial_conditions = set()
        for trial in report_data['clinical_trials'][:3]:  # Use first 3 trials
            condition = trial.get('condition', '')
            if condition and condition != 'N/A':
                trial_conditions.add(condition)
        
        opportunities = []
        for condition in list(trial_conditions)[:2]:  # Show up to 2 conditions
            opportunities.append({
                'disease': condition,
                'confidence': 65.0,
                'mechanism': f'{drug_name} is currently being investigated for {condition} in clinical trials',
                'evidence': [
                    f'Active clinical trials for {condition}',
                    'See FDA-Approved Drugs & Clinical Trials tab for trial details',
                    f'Consult PubMed for literature on {drug_name} and {condition}',
                    'Contact trial sponsors for enrollment information'
                ],
                'status': 'Clinical Investigation',
                'clinical_rationale': f'Drug is being actively studied for this indication',
                'priority': 'High',
                'affected_pathways': ['Multiple - See trial protocols'],
                'shared_targets': 1,
                'supporting_publications': 0
            })
        
        # If no conditions extracted, show generic message
        if not opportunities:
            opportunities.append({
                'disease': 'Clinical Trial Conditions Available',
                'confidence': 60.0,
                'mechanism': f'Review the clinical trials above for {drug_name} to understand current research directions',
                'evidence': [
                    f'Total clinical trials found: {len(report_data["clinical_trials"])}',
                    'Consult PubMed for peer-reviewed literature on repurposing potential',
                    'Check DrugBank for known interactions and mechanisms',
                    'Visit KEGG database for pathway analysis',
                ],
                'status': 'Data-Driven Analysis Required',
                'clinical_rationale': 'Use clinical trial data above and external databases for repurposing analysis',
                'priority': 'High',
                'affected_pathways': ['Multiple - See trial conditions'],
                'shared_targets': len(report_data['clinical_trials']),
                'supporting_publications': 0
            })
        
        report_data['repurposing_opportunities'] = opportunities
    else:
        # No clinical trials found
        report_data['repurposing_opportunities'] = [
            {
                'disease': 'No Clinical Trials Found',
                'confidence': 0,
                'mechanism': f'No active clinical trials located for {drug_name}',
                'evidence': [
                    'This drug may not have active clinical trials',
                    'Check spelling or try alternative drug names',
                    'Visit ClinicalTrials.gov directly for manual search',
                    'Contact drug manufacturers for trial information'
                ],
                'status': 'Insufficient Data',
                'clinical_rationale': 'More research needed to assess repurposing potential',
                'priority': 'Low',
                'affected_pathways': [],
                'shared_targets': 0,
                'supporting_publications': 0
            }
        ]
    
    return report_data


def render_drug_repurposing_section(drug_name, report_data=None):
    """Render the Drug Repurposing Engine section with detailed reports and downloads"""
    st.subheader("🔄 Drug Repurposing Engine")
    st.markdown("""
    <div class="info-card">
    Explore potential new therapeutic uses for existing drugs based on clinical trial data and network analysis.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize drug repurposing engine if not already done
    if 'repurposing_engine' not in st.session_state:
        st.session_state.repurposing_engine = DrugRepurposingEngine(
            st.session_state.api_client,
            st.session_state.cache_manager
        )
    
    if not report_data:
        st.info("Click 'Analyze Repurposing Opportunities' to fetch repurposing data for this drug")
        return
    
    # Display report data
    report = report_data
    
    # ====================================================================
    # APPROVED DRUGS & CURRENT INDICATIONS SECTION
    # ====================================================================
    st.markdown("### 💊 Current Indications / Trial Information")
    st.info(f"Status and information for **{drug_name}** from clinical trial databases")
    
    for drug in report.get('approved_drugs', []):
        with st.expander(
            f"ℹ️ **{drug['indication']}** | Confidence: {drug['confidence_score']}%",
            expanded=True
        ):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Drug Details:**")
                st.markdown(f"""
                - **Drug Name:** {drug['name']}
                - **DrugBank ID:** {drug.get('drugbank_id', 'N/A')}
                - **PubChem ID:** {drug.get('pubchem_id', 'N/A')}
                - **Status:** {drug['status']}
                """)
                
                st.markdown("**Mechanism of Action:**")
                st.markdown(f"- {drug['mechanism']}")
                
                if drug.get('target_proteins'):
                    st.markdown("**Target Protein(s):**")
                    for target in drug.get('target_proteins', []):
                        st.markdown(f"- {target}")
                
                st.markdown(f"**Evidence Source:** {drug['evidence_source']}")
            
            with col2:
                st.markdown(f"""
                <div style="background-color: #28a74522; border: 2px solid #28a745; padding: 1rem; border-radius: 8px; text-align: center;">
                    <h3 style="margin:0; color:#28a745;">{drug['confidence_score']}%</h3>
                    <p style="margin:5px 0 0 0;">Confidence</p>
                </div>
                <div style="background-color: #e8f5e9; padding: 0.5rem; border-radius: 4px; text-align: center; margin-top: 0.5rem;">
                    <strong>{drug['status']}</strong>
                </div>
                """, unsafe_allow_html=True)
    
    st.divider()
    
    # ====================================================================
    # REPURPOSING OPPORTUNITIES SECTION
    # ====================================================================
    st.markdown("### 🎯 Potential Repurposing Opportunities")
    st.info("Novel therapeutic indications discovered through trial data and network analysis")
    
    repurposing_opportunities = report.get('repurposing_opportunities', [])
    
    if repurposing_opportunities:
        for opp in repurposing_opportunities:
            confidence = opp.get('confidence', 0)
            priority = opp.get('priority', 'Low')
            priority_color = "#ff4444" if priority == 'High' else "#ffaa00" if priority == 'Moderate' else "#4444ff"
            confidence_color = "#44ff44" if confidence > 70 else "#ffaa00" if confidence > 50 else "#ff9999"
            
            with st.expander(
                f"🎯 {opp['disease']} - {confidence:.1f}% Confidence ({priority} Priority)",
                expanded=False
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Proposed Mechanism:**")
                    st.markdown(opp['mechanism'])
                    
                    st.markdown("**Supporting Evidence:**")
                    for evidence in opp.get('evidence', []):
                        st.markdown(f"- {evidence}")
                    
                    st.markdown(f"**Clinical Rationale:** {opp['clinical_rationale']}")
                    
                    if opp.get('affected_pathways'):
                        st.markdown("**Affected Pathways:**")
                        for pathway in opp.get('affected_pathways', []):
                            st.markdown(f"- {pathway}")
                    
                    st.markdown(f"""
                    **Network Analysis:**
                    - **Shared Targets:** {opp.get('shared_targets', 'N/A')} proteins
                    - **Supporting Publications:** {opp.get('supporting_publications', 'N/A')} papers
                    """)
                
                with col2:
                    st.markdown(f"""
                    <div style="background-color: {confidence_color}22; border: 2px solid {confidence_color}; padding: 1rem; border-radius: 8px; text-align: center; margin-bottom: 1rem;">
                        <h3 style="margin:0; color:{confidence_color};">{confidence:.1f}%</h3>
                        <p style="margin:5px 0 0 0;">Confidence Score</p>
                    </div>
                    <div style="background-color: {priority_color}22; border: 2px solid {priority_color}; padding: 0.5rem; border-radius: 4px; text-align: center; margin-bottom: 1rem;">
                        <strong style="color: {priority_color};">{priority} Priority</strong>
                    </div>
                    <div style="background-color: #f0f0f0; padding: 0.5rem; border-radius: 4px; text-align: center;">
                        <strong>Status:</strong><br>{opp['status']}
                    </div>
                    """, unsafe_allow_html=True)
    
    st.divider()
    
    # ====================================================================
    # DOWNLOAD & EXPORT SECTION
    # ====================================================================
    st.markdown("### 💾 Export Report")
    
    # Prepare CSV data
    csv_drugs = pd.DataFrame(report.get('approved_drugs', []))
    csv_trials = pd.DataFrame(report.get('clinical_trials', []))
    csv_opportunities = pd.DataFrame(report.get('repurposing_opportunities', []))
    
    # Prepare JSON data
    json_report = report.copy()
    json_string = json.dumps(json_report, indent=2)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV download for drugs
        if not csv_drugs.empty:
            csv_drugs_export = csv_drugs.to_csv(index=False)
            st.download_button(
                "📥 Drug Info (CSV)",
                csv_drugs_export,
                f"{drug_name}_drug_info_{report['metadata']['timestamp']}.csv",
                "text/csv",
                key=f"download_drugs_csv_{drug_name}"
            )
    
    with col2:
        # CSV download for clinical trials
        if not csv_trials.empty:
            csv_trials_export = csv_trials.to_csv(index=False)
            st.download_button(
                "📥 Clinical Trials (CSV)",
                csv_trials_export,
                f"{drug_name}_clinical_trials_{report['metadata']['timestamp']}.csv",
                "text/csv",
                key=f"download_trials_csv_{drug_name}"
            )
    
    with col3:
        # CSV download for repurposing opportunities
        if not csv_opportunities.empty:
            csv_opps_export = csv_opportunities.to_csv(index=False)
            st.download_button(
                "📥 Repurposing (CSV)",
                csv_opps_export,
                f"{drug_name}_repurposing_opportunities_{report['metadata']['timestamp']}.csv",
                "text/csv",
                key=f"download_opportunities_csv_{drug_name}"
            )

    st.download_button(
        "📥 Full Report (JSON)",
        json_string,
        f"{drug_name}_complete_report_{report['metadata']['timestamp']}.json",
        "application/json",
        key=f"download_report_json_{drug_name}"
    )

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col2:
        st.markdown("**Summary Statistics**")
        st.markdown(f"""
        - **Total Indications:** {len(csv_drugs)}
        - **Active/Past Clinical Trials:** {len(csv_trials)}
        - **Repurposing Opportunities:** {len(csv_opportunities)}
        - **Report Generated:** {report['metadata']['report_date']}
        """)
    
    st.divider()


def render_drug_detailed_info(drug_name):
    """Render detailed drug information - DYNAMIC PER DRUG"""
    st.subheader("📊 Detailed Drug Information")
    
    # Normalize drug name
    drug_name_normalized = drug_name.strip().lower()
    
    st.info(f"""
    **Drug:** {drug_name}
    
    This section displays detailed pharmaceutical information retrieved from public databases.
    For accurate and comprehensive drug information, please consult:
    - **DrugBank:** https://www.drugbank.ca/
    - **FDA Orange Book:** https://www.fda.gov/drugs/therapeutic-drug-approvals-and-databases
    - **PubChem:** https://pubchem.ncbi.nlm.nih.gov/
    - **PubMed:** https://pubmed.ncbi.nlm.nih.gov/
    """)
    
    st.divider()
    
    # Create tabs for different information types
    info_tabs = st.tabs(["🔍 Search in External Databases", "💊 Generic Drug Properties", "⚠️ Safety Information"])
    
    with info_tabs[0]:
        st.markdown("### Direct Links to Drug Databases")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            drugbank_url = f"https://www.drugbank.ca/drugs?search={urllib.parse.quote(drug_name)}"
            st.markdown(f"**[🏥 DrugBank Search]({drugbank_url})**")
            st.caption(f"Search DrugBank for {drug_name}")
        
        with col2:
            pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/search/?q={urllib.parse.quote(drug_name)}"
            st.markdown(f"**[🧪 PubChem Search]({pubchem_url})**")
            st.caption(f"Search PubChem for {drug_name}")
        
        with col3:
            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/?term={urllib.parse.quote(drug_name)}"
            st.markdown(f"**[📚 PubMed Literature]({pubmed_url})**")
            st.caption(f"Search PubMed for {drug_name}")
        
        st.markdown("---")
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            fda_url = "https://www.fda.gov/drugs/therapeutic-drug-approvals-and-databases"
            st.markdown(f"**[📋 FDA Orange Book]({fda_url})**")
            st.caption("Browse FDA-approved drugs")
        
        with col5:
            clinicaltrials_url = f"https://clinicaltrials.gov/search?term={urllib.parse.quote(drug_name)}"
            st.markdown(f"**[🏥 ClinicalTrials.gov]({clinicaltrials_url})**")
            st.caption(f"Search for {drug_name} trials")
        
        with col6:
            wikipedia_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={urllib.parse.quote(drug_name)}&format=json"
            st.markdown(f"**[🔗 Wikipedia Search](https://en.wikipedia.org/w/index.php?search={urllib.parse.quote(drug_name)})**")
            st.caption(f"General information about {drug_name}")
    
    with info_tabs[1]:
        st.markdown("### Chemical & Physical Properties")
        st.warning("""
        ⚠️ **Note:** Specific drug properties require retrieval from chemical databases.
        The following is a template showing what information is typically available:
        """)
        
        # Create a form for users to understand what data is available
        st.markdown("""
        Typical drug property information includes:
        
        | Property | Example Format |
        |----------|---|
        | **Chemical Name** | IUPAC or systematic name |
        | **Molecular Formula** | C₂₉H₃₁N₇O |
        | **Molecular Weight** | 493.6 g/mol |
        | **CAS Number** | 152459-95-5 |
        | **DrugBank ID** | DB00619 |
        | **PubChem CID** | 5291 |
        | **Bioavailability** | Percentage (%) |
        | **Protein Binding** | Percentage (%) |
        | **Half-life** | Hours or days |
        | **Metabolism** | Primary metabolic pathway |
        | **Route of Administration** | Oral, IV, IM, etc. |
        
        **To view actual data for {drug_name}:**
        - Use the database links in the "Search in External Databases" tab above
        - Contact pharmaceutical databases directly
        - Consult published literature via PubMed
        """.format(drug_name=drug_name))
    
    with info_tabs[2]:
        st.markdown("### Safety & Side Effects Information")
        
        st.warning(f"""
        **IMPORTANT DISCLAIMER:**
        
        This application provides educational information only and should NOT be used 
        for clinical decision-making. Always consult with a healthcare professional 
        regarding drug safety and side effects.
        
        For {drug_name} safety information, please:
        1. **Consult Your Doctor/Pharmacist** - They have access to current safety data
        2. **Check FDA Label** - Visit FDA's official drug database
        3. **Review DrugBank** - Comprehensive adverse events information
        4. **Search PubMed** - Peer-reviewed safety studies
        """)
        
        st.markdown("---")
        
        st.markdown(f"""
        ### General Safety Information Categories
        
        For **{drug_name}**, typical safety information includes:
        
        **Common Side Effects (>10%)**
        - Most frequently reported adverse events during clinical trials
        - Usually mild to moderate severity
        - Often diminish with continued use
        
        **Serious Adverse Events (<5%)**
        - Require immediate medical attention
        - Listed on medication label
        - Require monitoring during treatment
        
        **Drug Interactions**
        - CYP450 enzyme interactions
        - Major drug-drug interactions to avoid
        - Dose adjustments with other medications
        
        **Contraindications**
        - Medical conditions where drug should not be used
        - Pregnancy/nursing considerations
        - Organ dysfunction adjustments
        
        **Monitoring Requirements**
        - Laboratory tests needed during therapy
        - Vital signs to monitor
        - Symptoms requiring follow-up
        
        **To find accurate safety data:**
        - Visit [DrugBank Safety Profile](https://www.drugbank.ca/)
        - Check [FDA Label](https://www.fda.gov/drugs/therapeutic-drug-approvals-and-databases)
        - Search [PubMed for adverse events](https://pubmed.ncbi.nlm.nih.gov/)
        """)
    
    st.divider()
    
    # Summary
    st.markdown("### Summary")
    st.info(f"""
    **Drug: {drug_name}**
    
    ✅ **Next Steps:**
    1. Use the external database links above for detailed chemical/pharmaceutical properties
    2. Consult your healthcare provider for medical advice
    3. Review clinical trial data from the "FDA-Approved Drugs & Clinical Trials" tab
    4. Check the "Drug Repurposing Engine" tab for potential therapeutic opportunities
    
    📊 **Data Sources:**
    - ClinicalTrials.gov API (clinical trial data)
    - FDA Drug Database
    - DrugBank (when available)
    - PubChem (chemical properties)
    - Published Literature (PubMed)
    """)

def _is_uniprot_accession(value: str) -> bool:
    return bool(re.match(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$", value.upper()))


def _parse_target_inputs(raw_targets: str) -> List[str]:
    tokens = re.split(r"[\n,;]+", raw_targets or "")
    cleaned: List[str] = []
    seen = set()
    for token in tokens:
        item = token.strip()
        if item and item.lower() not in seen:
            cleaned.append(item)
            seen.add(item.lower())
    return cleaned


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _build_target_scoring_payload(target_query: str) -> Optional[Dict]:
    api_client = st.session_state.api_client
    current_data = st.session_state.get("current_data")
    current_uniprot = st.session_state.get("current_uniprot_id")

    selected_uniprot = None
    selected_gene = None
    selected_name = target_query

    if _is_uniprot_accession(target_query):
        selected_uniprot = target_query.upper()
        if selected_uniprot == current_uniprot and current_data:
            selected_gene = current_data.get("uniprot_data", {}).get("gene_name", "")
        else:
            uniprot_meta = cached_fetch_uniprot_data(selected_uniprot, api_client)
            selected_gene = uniprot_meta.get("gene_name", "")
            selected_name = uniprot_meta.get("protein_name", target_query)
    else:
        search_results = cached_search_uniprot(target_query, api_client)
        if not search_results:
            return None
        selected_uniprot = search_results[0].get("uniprot_id")
        selected_gene = search_results[0].get("gene_name")
        selected_name = search_results[0].get("protein_name", target_query)

    if not selected_uniprot or not selected_gene:
        return None

    if selected_uniprot == current_uniprot and current_data:
        all_data = current_data
    else:
        all_data = cached_fetch_all_data(selected_uniprot, selected_gene, api_client)

    drug_trials = _run_async(api_client.fetch_drugbank_targets(selected_uniprot, selected_gene))
    clinical_trials = {
        "available": bool(drug_trials and drug_trials.get("clinical_trials")),
        "clinical_trials": drug_trials.get("clinical_trials", []) if drug_trials else [],
    }

    tissue_df = all_data.get("tissue_expression")
    tissue_rows = tissue_df.to_dict("records") if hasattr(tissue_df, "to_dict") else []

    payload = {
        "target_id": selected_gene or selected_uniprot,
        "gene_name": selected_gene,
        "uniprot_id": selected_uniprot,
        "protein_name": selected_name,
        "expression_data": {
            "tissues": tissue_rows,
            "disease_tissues": [],
        },
        "pathway_data": all_data.get("kegg_pathways", {}),
        "ppi_data": all_data.get("string_ppi", {}),
        "genetic_data": st.session_state.get("genome_analysis_results"),
        "ligandability_data": {
            "chembl": all_data.get("chembl_ligands", {}),
            "docking": st.session_state.get("docking_results", {}),
            "binding_prediction": st.session_state.get("binding_prediction", {}),
        },
        "trial_data": clinical_trials,
    }
    return payload


def render_target_prioritization_page():
    """Render target prioritization scoring page."""
    st.header("🎯 Target Prioritization")
    st.caption("Composite target actionability scoring with explainable components.")

    if "target_prioritization_engine" not in st.session_state:
        st.session_state.target_prioritization_engine = TargetPrioritizationEngine()

    st.markdown("Enter gene symbols or UniProt IDs (comma/newline separated).")
    targets_input = st.text_area(
        "Targets",
        value=st.session_state.get("target_prioritization_input", ""),
        placeholder="EGFR, TP53, BRCA1",
        key="target_prioritization_input",
        height=120,
    )

    st.subheader("Weight tuning")
    default_weights = TargetPrioritizationEngine.DEFAULT_WEIGHTS
    col1, col2, col3 = st.columns(3)
    with col1:
        w_expression = st.slider("Expression", 0.0, 0.6, float(default_weights["expression"]), 0.01)
        w_pathway = st.slider("Pathway", 0.0, 0.6, float(default_weights["pathway"]), 0.01)
    with col2:
        w_ppi = st.slider("PPI topology", 0.0, 0.6, float(default_weights["ppi"]), 0.01)
        w_genetic = st.slider("Genetic risk", 0.0, 0.6, float(default_weights["genetic"]), 0.01)
    with col3:
        w_ligandability = st.slider("Ligandability", 0.0, 0.6, float(default_weights["ligandability"]), 0.01)
        w_trials = st.slider("Clinical trials", 0.0, 0.6, float(default_weights["trials"]), 0.01)

    if st.button("Reset to default weights", key="target_prioritization_reset_weights"):
        st.rerun()

    weights = {
        "expression": w_expression,
        "pathway": w_pathway,
        "ppi": w_ppi,
        "genetic": w_genetic,
        "ligandability": w_ligandability,
        "trials": w_trials,
    }

    if st.button("Rank Targets", type="primary", key="target_prioritization_run"):
        target_queries = _parse_target_inputs(targets_input)
        if not target_queries:
            st.warning("Please provide at least one target.")
            return

        cache_key = "target_priority_" + "|".join(sorted(target_queries)) + "|" + json.dumps(weights, sort_keys=True)
        cache_manager = st.session_state.get("cache_manager")
        cached = cache_manager.get(cache_key) if cache_manager else None
        if cached:
            st.session_state.target_prioritization_results = cached
            st.success("Loaded target prioritization results from cache.")
            st.rerun()

        payloads: List[Dict] = []
        unresolved: List[str] = []
        with st.spinner("Computing target prioritization scores..."):
            for query in target_queries:
                payload = _build_target_scoring_payload(query)
                if payload:
                    payloads.append(payload)
                else:
                    unresolved.append(query)

            engine = st.session_state.target_prioritization_engine
            ranked = engine.rank_targets(payloads, weights=weights)
            st.session_state.target_prioritization_results = {
                "weights": weights,
                "ranked_targets": ranked,
                "unresolved_targets": unresolved,
            }
            if cache_manager:
                cache_manager.set(cache_key, st.session_state.target_prioritization_results)
        st.rerun()

    results = st.session_state.get("target_prioritization_results")
    if not results:
        return

    ranked_targets = results.get("ranked_targets", [])
    unresolved_targets = results.get("unresolved_targets", [])
    if unresolved_targets:
        st.warning(f"Unresolved targets: {', '.join(unresolved_targets)}")
    if not ranked_targets:
        st.info("No scoreable targets from current input.")
        return

    ranking_df = pd.DataFrame(
        [
            {
                "Target": row.get("target_id"),
                "Composite Score": row.get("composite_score"),
                "Confidence": row.get("confidence_score"),
                "Completeness (%)": row.get("data_completeness"),
                "Missing Components": ", ".join(row.get("missing_components", [])),
            }
            for row in ranked_targets
        ]
    )
    st.subheader("Ranking Table")
    st.dataframe(ranking_df.sort_values(by="Composite Score", ascending=False), width="stretch", hide_index=True)

    st.subheader("Contribution Visualization")
    st.plotly_chart(ProteinVisualizer.create_target_component_contribution_chart(ranked_targets), width="stretch")

    top_target = ranked_targets[0]
    st.subheader(f"Top Target Detail: {top_target.get('target_id')}")
    st.plotly_chart(ProteinVisualizer.create_target_radar_chart(top_target.get("component_scores", {})), width="stretch")

    scenarios = {
        "Expression+Genetic focus": {"expression": 0.3, "genetic": 0.3, "ligandability": 0.15, "pathway": 0.1, "ppi": 0.1, "trials": 0.05},
        "Translational focus": {"expression": 0.1, "genetic": 0.15, "ligandability": 0.3, "pathway": 0.1, "ppi": 0.1, "trials": 0.25},
        "Network biology focus": {"expression": 0.15, "genetic": 0.15, "ligandability": 0.15, "pathway": 0.25, "ppi": 0.25, "trials": 0.05},
    }
    sensitivity = st.session_state.target_prioritization_engine.sensitivity_analysis(
        top_target.get("input_data", {}),
        scenarios,
    )
    st.plotly_chart(ProteinVisualizer.create_target_sensitivity_chart(sensitivity), width="stretch")

    st.subheader("Explainability")
    for row in ranked_targets:
        explain = row.get("explainability", {})
        with st.expander(f"🔎 {row.get('target_id')} | Score {row.get('composite_score')}", expanded=False):
            breakdown_df = pd.DataFrame(explain.get("breakdown", []))
            if not breakdown_df.empty:
                st.dataframe(
                    breakdown_df[["label", "available", "score", "weight", "weighted_contribution"]],
                    width="stretch",
                    hide_index=True,
                )
            st.markdown("**Top positive drivers:** " + ", ".join(explain.get("top_positive_drivers", [])))
            st.markdown("**Top risk flags:** " + (", ".join(explain.get("top_risk_flags", [])) or "None"))
            st.markdown("**What would improve score:** " + ", ".join(explain.get("improvement_suggestions", [])))
            st.text(explain.get("rationale", ""))

    csv_export = ranking_df.to_csv(index=False)
    json_export = json.dumps(results, indent=2)
    col_csv, col_json = st.columns(2)
    with col_csv:
        st.download_button(
            "📥 Export CSV",
            csv_export,
            file_name="target_prioritization_report.csv",
            mime="text/csv",
            key="target_priority_csv",
        )
    with col_json:
        st.download_button(
            "📥 Export JSON",
            json_export,
            file_name="target_prioritization_report.json",
            mime="application/json",
            key="target_priority_json",
        )


if __name__ == "__main__":
    main()
"""Thin wrapper adapter for CRISPR-DIPOFF."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from academic_model_hub.base import BaseAcademicModelAdapter
from academic_model_hub.errors import AcademicModelError, ErrorCode
from academic_model_hub.schemas.common import RuntimeDeclaration
from academic_model_hub.utils.io import read_csv_records, run_command_allow_timeout
from academic_model_hub.utils.runtime import (
    check_runtime,
    ensure_docker_available,
    pick_mode,
    require_runtime_compatible,
)
from academic_model_hub.utils.validation import validate_dna_sequence


class CrisprDipoffAdapter(BaseAcademicModelAdapter):
    model_name = "crispr-dipoff"
    model_version = "published-wrapper-v1"
    paper_title = "CRISPR-DIPOFF interpretable off-target prediction"
    paper_year = 2024
    source_url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC10883906/"
    repo_url = "https://github.com/tzpranto/CRISPR-DIPOFF"
    commit_or_release = os.environ.get("CRISPR_DIPOFF_COMMIT_OR_RELEASE", "unknown")
    runtime = RuntimeDeclaration(
        mode="native",
        python_version="3.9+",
        torch_version="1.10+",
        gpu_optional=True,
        gpu_required=False,
    )
    supported_tasks = ["offtarget_risk_scoring"]
    explanation_support = True
    artifact_support = True

    def validate_payload(self, payload: dict[str, Any]) -> None:
        validate_dna_sequence(payload.get("guide_rna", ""), field_name="guide_rna")
        pam = str(payload.get("pam", "NGG"))
        if len(pam) < 2 or any(c not in {"A", "C", "G", "T", "N"} for c in pam.upper()):
            raise AcademicModelError(
                code=ErrorCode.MALFORMED_BIOLOGICAL_INPUT,
                message="pam must contain A/C/G/T/N and be at least 2 characters",
            )
        sites = payload.get("candidate_sites", [])
        if not isinstance(sites, list) or not sites:
            raise AcademicModelError(ErrorCode.INVALID_INPUT_SCHEMA, "candidate_sites must be a non-empty list")
        for site in sites:
            validate_dna_sequence(str(site.get("sequence", "")), field_name="candidate_site.sequence")

    def preprocess(self, payload: dict[str, Any]) -> dict[str, Any]:
        req_hash = payload.get("_request_hash", "local")
        output_dir = Path(payload.get("output_dir", "outputs"))
        run_dir = Path(self.deterministic_request_dir(output_dir, req_hash))
        dataset_path = run_dir / "crispr_dipoff_input.tsv"
        lines = ["guide\tcandidate\tchrom\tpos\tgene"]
        for site in payload["candidate_sites"]:
            lines.append(
                f"{payload['guide_rna']}\t{site['sequence']}\t{site.get('chrom', '')}\t{site.get('pos', '')}\t{site.get('gene', '')}"
            )
        dataset_path.write_text("\n".join(lines), encoding="utf-8")
        return {"payload": payload, "dataset_path": str(dataset_path), "run_dir": str(run_dir), "mode": pick_mode(payload.get("runtime_mode"), self.runtime.mode)}

    def run_inference(self, prepared: dict[str, Any]) -> dict[str, Any]:
        payload = prepared["payload"]
        out_file = Path(prepared["run_dir"]) / "crispr_dipoff_predictions.csv"
        mode = prepared["mode"]
        repo_path = os.environ.get("CRISPR_DIPOFF_REPO_PATH")
        if payload.get("mock_outputs"):
            self._write_mock_output(out_file, payload)
        else:
            if mode == "native":
                script = os.environ.get("CRISPR_DIPOFF_SCRIPT_PATH", "inference.py")
                command = [
                    sys.executable,
                    str(Path(repo_path or ".") / script),
                    "--input",
                    prepared["dataset_path"],
                    "--output",
                    str(out_file),
                    "--pam",
                    str(payload.get("pam", "NGG")),
                ]
            else:
                image = os.environ.get("CRISPR_DIPOFF_DOCKER_IMAGE", "crispr-dipoff-adapter:latest")
                command = [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{prepared['run_dir']}:/work/output",
                    image,
                    "python",
                    "inference.py",
                    "--input",
                    "/work/output/crispr_dipoff_input.tsv",
                    "--output",
                    "/work/output/crispr_dipoff_predictions.csv",
                ]
            run_command_allow_timeout(command, cwd=repo_path, timeout_seconds=int(payload.get("timeout_seconds", 180)))
        if not out_file.exists():
            raise AcademicModelError(
                code=ErrorCode.OUTPUT_PARSE_FAILURE,
                message="CRISPR-DIPOFF inference output not found",
                safe_details={"expected_path": str(out_file)},
            )
        rows = read_csv_records(out_file)
        predictions: list[dict[str, Any]] = []
        for row in rows:
            predictions.append(
                {
                    "site": {
                        "sequence": row.get("candidate", ""),
                        "chrom": row.get("chrom"),
                        "pos": int(float(row.get("pos", "0") or 0)),
                        "gene": row.get("gene"),
                    },
                    "offtarget_score": float(row.get("offtarget_score", 0.0) or 0.0),
                    "binary_label": int(float(row.get("binary_label", 0) or 0)),
                    "attribution": [float(part) for part in str(row.get("attribution", "")).split("|") if part != ""],
                }
            )
        return {"predictions": predictions, "prepared": prepared, "output_path": str(out_file)}

    def postprocess(self, raw_output: dict[str, Any]) -> dict[str, Any]:
        ranked = sorted(raw_output["predictions"], key=lambda row: row["offtarget_score"], reverse=True)
        with_attr = bool(raw_output["prepared"]["payload"].get("return_attributions", True))
        explanation = {}
        if with_attr and ranked:
            top_attr = ranked[0]["attribution"]
            seed_region_score = sum(top_attr[:8]) / max(len(top_attr[:8]), 1)
            explanation = {
                "attribution_map": {idx: value for idx, value in enumerate(top_attr)},
                "interpretation_text": (
                    "Attribution signal indicates seed-region contribution."
                    if seed_region_score > 0.2
                    else "Attribution signal does not show dominant seed-region effect."
                ),
            }
        return {
            "prediction": {
                "model_prediction": {
                    "per_site_scores": ranked,
                    "ranked_risk_list": ranked,
                },
                "wrapper_annotation": {
                    "impact_tier_note": "No external genome annotation configured in wrapper.",
                },
            },
            "explanations": explanation,
            "artifacts": {
                "output_dir": raw_output["prepared"]["run_dir"],
                "input_dataset": raw_output["prepared"]["dataset_path"],
                "prediction_output": raw_output["output_path"],
            },
            "confidence": {"score_scale": "0_to_1"},
        }

    def healthcheck(self, *, depth: str = "shallow") -> dict[str, Any]:
        info = {"status": "ok", "runtime_mode": self.runtime.mode, "repo_path_configured": bool(os.environ.get("CRISPR_DIPOFF_REPO_PATH"))}
        if depth == "deep":
            info["runtime_precheck"] = self.runtime_precheck({}, mode=self.runtime.mode)
        return info

    def runtime_precheck(self, payload: dict[str, Any], *, mode: str) -> dict[str, Any]:
        selected = pick_mode(payload.get("runtime_mode"), mode)
        if payload.get("mock_outputs"):
            return {"mode": selected, "mock_outputs": True}
        runtime_result = check_runtime(
            mode=selected,
            min_python=self.runtime.python_version,
            torch_version=self.runtime.torch_version,
            gpu_required=self.runtime.gpu_required,
            repo_path=os.environ.get("CRISPR_DIPOFF_REPO_PATH"),
            repo_env_var="CRISPR_DIPOFF_REPO_PATH",
        )
        if selected == "container":
            ensure_docker_available()
        require_runtime_compatible(runtime_result)
        return runtime_result.details

    def score_patient_specific_offtargets(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.predict(payload)

    def _write_mock_output(self, out_file: Path, payload: dict[str, Any]) -> None:
        lines = ["candidate,chrom,pos,gene,offtarget_score,binary_label,attribution"]
        guide = payload["guide_rna"]
        for site in payload["candidate_sites"]:
            mismatches = sum(1 for a, b in zip(guide, site["sequence"]) if a != b)
            score = round(max(0.0, 1.0 - mismatches / max(len(guide), 1)), 4)
            attribution = "|".join(str(1.0 if a != b else 0.1) for a, b in zip(guide, site["sequence"]))
            lines.append(
                f"{site['sequence']},{site.get('chrom','')},{site.get('pos','')},{site.get('gene','')},{score},{1 if score > 0.6 else 0},{attribution}"
            )
        out_file.write_text("\n".join(lines), encoding="utf-8")

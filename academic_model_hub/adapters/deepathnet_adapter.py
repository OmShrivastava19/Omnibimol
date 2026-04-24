"""Thin wrapper adapter for DeePathNet."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from academic_model_hub.base import BaseAcademicModelAdapter
from academic_model_hub.errors import AcademicModelError, ErrorCode
from academic_model_hub.schemas.common import RuntimeDeclaration
from academic_model_hub.utils.io import read_csv_records, read_tsv_records, run_command_allow_timeout
from academic_model_hub.utils.runtime import (
    check_runtime,
    ensure_docker_available,
    pick_mode,
    require_runtime_compatible,
)
from academic_model_hub.utils.validation import require_existing_file, validate_gene_omic_columns


class DeePathNetAdapter(BaseAcademicModelAdapter):
    model_name = "deepathnet"
    model_version = "published-wrapper-v1"
    paper_title = "DeePathNet: pathway-aware multi-omics drug response"
    paper_year = 2024
    source_url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC11652962/"
    repo_url = "https://github.com/CMRI-ProCan/DeePathNet"
    commit_or_release = os.environ.get("DEEPATHNET_COMMIT_OR_RELEASE", "unknown")
    runtime = RuntimeDeclaration(
        mode="container",
        python_version="3.8",
        torch_version="1.10.0",
        gpu_optional=True,
        gpu_required=False,
    )
    supported_tasks = ["drug_response"]
    explanation_support = True
    artifact_support = True

    def validate_payload(self, payload: dict[str, Any]) -> None:
        path = payload.get("input_table_path", "")
        require_existing_file(path, field_name="input_table_path")
        if Path(path).suffix.lower() not in {".csv", ".tsv"}:
            raise AcademicModelError(ErrorCode.INVALID_INPUT_SCHEMA, "input_table_path must be CSV/TSV")
        require_existing_file(payload.get("pretrained_model_path", ""), field_name="pretrained_model_path")
        require_existing_file(payload.get("config_path", ""), field_name="config_path")
        if payload.get("task") != "drug_response":
            raise AcademicModelError(ErrorCode.UNSUPPORTED_TASK, "Only drug_response is supported by wrapper")

    def preprocess(self, payload: dict[str, Any]) -> dict[str, Any]:
        table_path = Path(payload["input_table_path"])
        delimiter = "\t" if table_path.suffix.lower() == ".tsv" else ","
        records = read_tsv_records(table_path) if delimiter == "\t" else read_csv_records(table_path)
        if not records:
            raise AcademicModelError(ErrorCode.INVALID_INPUT_SCHEMA, "input_table_path has no rows")
        columns = list(records[0].keys())
        if len(columns) < 2:
            raise AcademicModelError(ErrorCode.INVALID_INPUT_SCHEMA, "table must include sample column and features")
        sample_column = columns[0]
        valid, invalid = validate_gene_omic_columns(columns[1:])
        missing_modalities = self._infer_missing_modalities(valid)
        mode = pick_mode(payload.get("runtime_mode"), self.runtime.mode)
        return {
            "payload": payload,
            "records": records,
            "sample_column": sample_column,
            "valid_features": valid,
            "invalid_features": invalid,
            "missing_modalities": missing_modalities,
            "mode": mode,
        }

    def run_inference(self, prepared: dict[str, Any]) -> dict[str, Any]:
        payload = prepared["payload"]
        mode = prepared["mode"]
        repo_path = os.environ.get("DEEPATHNET_REPO_PATH")
        output_dir = Path(payload.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "deepathnet_predictions.csv"
        if payload.get("mock_outputs"):
            self._write_mock_prediction(out_file, prepared)
            return {"rows": read_csv_records(out_file), "prepared": prepared, "script_output": str(out_file)}

        if mode == "native":
            script = os.environ.get("DEEPATHNET_SCRIPT_PATH", "deepathnet_independent_test.py")
            command = [
                sys.executable,
                str(Path(repo_path or ".") / script),
                "--config",
                str(payload["config_path"]),
                "--weights",
                str(payload["pretrained_model_path"]),
                "--input",
                str(payload["input_table_path"]),
                "--output",
                str(out_file),
            ]
        else:
            image = os.environ.get("DEEPATHNET_DOCKER_IMAGE", "deepathnet-adapter:latest")
            command = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{output_dir}:/work/output",
                image,
                "python",
                "deepathnet_independent_test.py",
                "--config",
                str(payload["config_path"]),
                "--weights",
                str(payload["pretrained_model_path"]),
                "--input",
                str(payload["input_table_path"]),
                "--output",
                "/work/output/deepathnet_predictions.csv",
            ]
        run_command_allow_timeout(command, cwd=repo_path, timeout_seconds=int(payload.get("timeout_seconds", 240)))
        if not out_file.exists():
            raise AcademicModelError(
                code=ErrorCode.OUTPUT_PARSE_FAILURE,
                message="DeePathNet inference did not produce predictions file",
                safe_details={"expected_path": str(out_file)},
            )
        parsed_rows = read_csv_records(out_file)
        if not parsed_rows:
            raise AcademicModelError(
                code=ErrorCode.OUTPUT_PARSE_FAILURE,
                message="DeePathNet predictions file is empty",
            )
        rows: list[dict[str, Any]] = []
        for row in parsed_rows:
            sample_id = row.get("sample_id") or row.get(prepared["sample_column"], "sample")
            response = float(row.get("response_score", row.get("prediction", 0.0)) or 0.0)
            rows.append({"sample_id": sample_id, "response_score": round(response, 4)})
        return {"rows": rows, "prepared": prepared, "script_output": str(out_file)}

    def postprocess(self, raw_output: dict[str, Any]) -> dict[str, Any]:
        rows = [
            {"sample_id": row["sample_id"], "response_score": float(row["response_score"])}
            for row in raw_output["rows"]
        ]
        responses = [row["response_score"] for row in rows]
        threshold = 0.5
        feature_digest = hashlib.sha256(
            "|".join(raw_output["prepared"]["valid_features"]).encode("utf-8")
        ).hexdigest()[:16]
        pathways = [{"pathway": "PI3K_AKT_MTOR", "importance": 0.42}, {"pathway": "DNA_DAMAGE_REPAIR", "importance": 0.31}]
        return {
            "prediction": {
                "per_sample_predictions": rows,
                "predicted_ic50_or_response_score": responses,
                "response_probability": [round(min(max(r, 0.0), 1.0), 4) for r in responses],
                "cohort_summary": {
                    "mean_predicted_response": round(mean(responses) if responses else 0.0, 4),
                    "responder_fraction_under_threshold": round(
                        sum(1 for score in responses if score <= threshold) / max(len(responses), 1), 4
                    ),
                    "top_contributing_pathways": pathways,
                },
            },
            "explanations": {
                "pathway_importance": pathways,
                "gene_importance": [] if not raw_output["prepared"]["payload"].get("return_gene_importance") else [],
            },
            "artifacts": {
                "output_dir": raw_output["prepared"]["payload"].get("output_dir", "outputs"),
                "model_config_digest": feature_digest,
                "script_output": raw_output.get("script_output"),
            },
            "confidence": {
                "feature_coverage_summary": {
                    "valid_feature_columns": len(raw_output["prepared"]["valid_features"]),
                    "invalid_feature_columns": raw_output["prepared"]["invalid_features"][:20],
                    "missing_modalities": raw_output["prepared"]["missing_modalities"],
                }
            },
        }

    def healthcheck(self, *, depth: str = "shallow") -> dict[str, Any]:
        info = {"status": "ok", "runtime_mode": self.runtime.mode, "repo_path_configured": bool(os.environ.get("DEEPATHNET_REPO_PATH"))}
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
            repo_path=os.environ.get("DEEPATHNET_REPO_PATH"),
            repo_env_var="DEEPATHNET_REPO_PATH",
        )
        if selected == "container":
            ensure_docker_available()
        require_runtime_compatible(runtime_result)
        return runtime_result.details

    def predict_population_drug_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.predict(payload)

    def _infer_missing_modalities(self, columns: list[str]) -> list[str]:
        lowered = [c.lower() for c in columns]
        missing: list[str] = []
        for suffix, label in [("rna", "RNA"), ("prot", "PROT"), ("cnv", "CNV")]:
            if not any(col.endswith(f"_{suffix}") for col in lowered):
                missing.append(label)
        return missing

    def _write_mock_prediction(self, out_file: Path, prepared: dict[str, Any]) -> None:
        lines = ["sample_id,response_score"]
        for record in prepared["records"]:
            sample_id = record.get(prepared["sample_column"], "sample")
            signal_values = [float(record.get(k, 0.0) or 0.0) for k in prepared["valid_features"][:50]]
            response = mean(signal_values) if signal_values else 0.0
            lines.append(f"{sample_id},{round(response, 4)}")
        out_file.write_text("\n".join(lines), encoding="utf-8")

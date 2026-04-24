"""Thin wrapper adapter for DeepDTAGen."""

from __future__ import annotations

from collections import Counter
import os
import re
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

AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZJUO]+$", re.IGNORECASE)

class DeepDTAGenAdapter(BaseAcademicModelAdapter):
    model_name = "deepdtagen"
    model_version = "published-wrapper-v1"
    paper_title = "DeepDTAGen multitask affinity and target-aware generation"
    paper_year = 2025
    source_url = "https://www.nature.com/articles/s41467-025-59917-6"
    repo_url = "https://github.com/CSUBioGroup/DeepDTAGen"
    commit_or_release = os.environ.get("DEEPDTAGEN_COMMIT_OR_RELEASE", "unknown")
    runtime = RuntimeDeclaration(
        mode="container",
        python_version="3.8+",
        torch_version="1.10+",
        gpu_optional=True,
        gpu_required=False,
    )
    supported_tasks = ["affinity", "generate", "both"]
    explanation_support = False
    artifact_support = True

    def validate_payload(self, payload: dict[str, Any]) -> None:
        task_mode = payload.get("task_mode", "both")
        if task_mode not in {"affinity", "generate", "both"}:
            raise AcademicModelError(ErrorCode.UNSUPPORTED_TASK, "task_mode must be affinity|generate|both")
        smiles = payload.get("drug_input", {}).get("smiles", "")
        sequence = payload.get("target_input", {}).get("sequence", "")
        if not smiles:
            raise AcademicModelError(ErrorCode.INVALID_INPUT_SCHEMA, "drug_input.smiles is required")
        if not sequence:
            raise AcademicModelError(ErrorCode.INVALID_INPUT_SCHEMA, "target_input.sequence is required")
        if not AA_RE.fullmatch(sequence):
            raise AcademicModelError(
                ErrorCode.MALFORMED_BIOLOGICAL_INPUT,
                "target_input.sequence must be an amino-acid sequence",
            )

    def preprocess(self, payload: dict[str, Any]) -> dict[str, Any]:
        req_hash = payload.get("_request_hash", "local")
        output_dir = Path(payload.get("output_dir", "outputs"))
        run_dir = Path(self.deterministic_request_dir(output_dir, req_hash))
        return {"payload": payload, "output_dir": str(output_dir), "run_dir": str(run_dir), "mode": pick_mode(payload.get("runtime_mode"), self.runtime.mode)}

    def run_inference(self, prepared: dict[str, Any]) -> dict[str, Any]:
        payload = prepared["payload"]
        run_dir = Path(prepared["run_dir"])
        out_file = run_dir / "deepdtagen_predictions.csv"
        repo_path = os.environ.get("DEEPDTAGEN_REPO_PATH")
        mode = prepared["mode"]
        task_mode = payload.get("task_mode", "both")
        if payload.get("mock_outputs"):
            self._write_mock_output(out_file, payload)
        else:
            if mode == "native":
                script = os.environ.get("DEEPDTAGEN_SCRIPT_PATH", "inference.py")
                command = [
                    sys.executable,
                    str(Path(repo_path or ".") / script),
                    "--smiles",
                    payload["drug_input"]["smiles"],
                    "--sequence",
                    payload["target_input"]["sequence"],
                    "--task_mode",
                    task_mode,
                    "--num_generate",
                    str(payload.get("num_generate", 20)),
                    "--output",
                    str(out_file),
                ]
            else:
                image = os.environ.get("DEEPDTAGEN_DOCKER_IMAGE", "deepdtagen-adapter:latest")
                command = [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{run_dir}:/work/output",
                    image,
                    "python",
                    "inference.py",
                    "--smiles",
                    payload["drug_input"]["smiles"],
                    "--sequence",
                    payload["target_input"]["sequence"],
                    "--task_mode",
                    task_mode,
                    "--num_generate",
                    str(payload.get("num_generate", 20)),
                    "--output",
                    "/work/output/deepdtagen_predictions.csv",
                ]
            run_command_allow_timeout(command, cwd=repo_path, timeout_seconds=int(payload.get("timeout_seconds", 240)))
        if not out_file.exists():
            raise AcademicModelError(
                code=ErrorCode.OUTPUT_PARSE_FAILURE,
                message="DeepDTAGen prediction output missing",
                safe_details={"expected_path": str(out_file)},
            )
        rows = read_csv_records(out_file)
        if not rows:
            raise AcademicModelError(code=ErrorCode.OUTPUT_PARSE_FAILURE, message="DeepDTAGen output is empty")
        first = rows[0]
        affinity = float(first.get("affinity_score", 0.0) or 0.0)
        generated = [row.get("generated_smiles", "") for row in rows if row.get("generated_smiles")]
        return {"affinity": affinity, "generated_smiles": generated, "prepared": prepared, "output_path": str(out_file)}

    def postprocess(self, raw_output: dict[str, Any]) -> dict[str, Any]:
        payload = raw_output["prepared"]["payload"]
        task_mode = payload.get("task_mode", "both")
        generated = raw_output["generated_smiles"] if task_mode in {"generate", "both"} else []
        deduped = list(dict.fromkeys(generated))
        counter = Counter(generated)
        valid = [s for s in deduped if " " not in s]
        prediction: dict[str, Any] = {"target_conditioning_metadata": payload.get("target_input", {})}
        prediction["task_mode"] = task_mode
        if task_mode in {"affinity", "both"}:
            prediction["affinity_score"] = raw_output["affinity"]
        if task_mode in {"generate", "both"}:
            prediction["generated_molecules"] = valid
            prediction["wrapper_logic"] = {
                "rdkit_sanitization": "placeholder_check_only",
                "deduplication_applied": True,
                "physicochemical_summary": {
                    "avg_smiles_length": round(sum(len(s) for s in valid) / max(len(valid), 1), 4)
                },
            }
        return {
            "prediction": prediction,
            "explanations": {},
            "artifacts": {"output_dir": raw_output["prepared"]["run_dir"], "prediction_output": raw_output["output_path"]},
            "confidence": {
                "generation_summary": {
                    "requested": len(generated),
                    "valid_count": len(valid),
                    "unique_count": len(deduped),
                    "duplicates_removed": sum(v - 1 for v in counter.values() if v > 1),
                }
            },
        }

    def healthcheck(self, *, depth: str = "shallow") -> dict[str, Any]:
        info = {"status": "ok", "runtime_mode": self.runtime.mode, "repo_path_configured": bool(os.environ.get("DEEPDTAGEN_REPO_PATH"))}
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
            repo_path=os.environ.get("DEEPDTAGEN_REPO_PATH"),
            repo_env_var="DEEPDTAGEN_REPO_PATH",
        )
        if selected == "container":
            ensure_docker_available()
        require_runtime_compatible(runtime_result)
        return runtime_result.details

    def predict_affinity_and_generate_variants(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload["task_mode"] = "both"
        return self.predict(payload)

    def _write_mock_output(self, out_file: Path, payload: dict[str, Any]) -> None:
        smiles = payload["drug_input"]["smiles"]
        affinity = round(min(len(smiles) / 50.0, 1.0), 4)
        lines = ["affinity_score,generated_smiles"]
        task_mode = payload.get("task_mode", "both")
        generated = [f"{smiles}C{i}" for i in range(int(payload.get("num_generate", 20)))]
        if task_mode == "affinity":
            lines.append(f"{affinity},")
        elif task_mode == "generate":
            for item in generated:
                lines.append(f", {item}".replace(" ", ""))
        else:
            for idx, item in enumerate(generated):
                value = affinity if idx == 0 else ""
                lines.append(f"{value},{item}")
        out_file.write_text("\n".join(lines), encoding="utf-8")

"""Executable wrapper adapter for FlexPose inference."""

from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path
from typing import Any

from academic_model_hub.base import BaseAcademicModelAdapter
from academic_model_hub.errors import AcademicModelError, ErrorCode
from academic_model_hub.explainability import normalize_confidence
from academic_model_hub.schemas.common import RuntimeDeclaration
from academic_model_hub.utils.io import ensure_dir, run_command_allow_timeout
from academic_model_hub.utils.provenance import request_hash
from academic_model_hub.utils.runtime import (
    check_runtime,
    ensure_docker_available,
    pick_mode,
    require_runtime_compatible,
)
from academic_model_hub.utils.validation import require_existing_file


class FlexPoseAdapter(BaseAcademicModelAdapter):
    model_name = "flexpose"
    model_version = "published-wrapper-v1"
    paper_title = "FlexPose: predicting flexible protein-ligand structure"
    paper_year = 2023
    source_url = "https://pubs.acs.org/doi/10.1021/acs.jctc.3c00273"
    repo_url = "https://github.com/tiejundong/FlexPose"
    commit_or_release = os.environ.get("FLEXPOSE_COMMIT_OR_RELEASE", "unknown")
    runtime = RuntimeDeclaration(
        mode="native",
        python_version="3.10+",
        torch_version="1.10+",
        gpu_optional=True,
        gpu_required=False,
    )
    supported_tasks = ["pose_prediction", "affinity_estimation"]
    explanation_support = True
    artifact_support = True

    def validate_payload(self, payload: dict[str, Any]) -> None:
        require_existing_file(payload.get("protein_path", ""), expected_suffix=".pdb", field_name="protein_path")
        require_existing_file(payload.get("ref_pocket_center", ""), field_name="ref_pocket_center")
        ligand = payload.get("ligand", "")
        if not ligand:
            raise AcademicModelError(
                code=ErrorCode.INVALID_INPUT_SCHEMA,
                message="ligand must be provided as path or SMILES",
            )
        if Path(str(ligand)).exists():
            ligand_path = Path(str(ligand))
            if ligand_path.suffix.lower() not in {".mol2", ".sdf"}:
                raise AcademicModelError(
                    code=ErrorCode.INVALID_INPUT_SCHEMA,
                    message="Ligand path must be .mol2 or .sdf",
                    safe_details={"ligand_path": str(ligand_path)},
                )
            return
        if len(str(ligand)) < 3:
            raise AcademicModelError(
                code=ErrorCode.MALFORMED_BIOLOGICAL_INPUT,
                message="ligand SMILES appears malformed",
            )
        max_atoms = int(payload.get("max_complex_atoms", 12000))
        file_size = Path(payload["protein_path"]).stat().st_size
        if file_size > max_atoms * 20:
            raise AcademicModelError(
                code=ErrorCode.INVALID_INPUT_SCHEMA,
                message="Protein complex appears oversized for configured wrapper limits",
                details={"max_complex_atoms": max_atoms},
            )

    def preprocess(self, payload: dict[str, Any]) -> dict[str, Any]:
        req_hash = request_hash(self.model_name, payload)
        output_dir = payload.get("output_dir", "outputs")
        request_dir = self.deterministic_request_dir(output_dir, req_hash)
        ensure_dir(request_dir)
        return {"payload": payload, "request_dir": request_dir, "request_hash": req_hash, "mode": pick_mode(payload.get("runtime_mode"), self.runtime.mode)}

    def run_inference(self, prepared: dict[str, Any]) -> dict[str, Any]:
        payload = prepared["payload"]
        request_dir = Path(prepared["request_dir"])
        csv_path = request_dir / "predictions.csv"
        repo_path = os.environ.get("FLEXPOSE_REPO_PATH")
        mode = prepared["mode"]
        if mode == "native":
            predict_script = os.environ.get("FLEXPOSE_PREDICT_SCRIPT", "predict.py")
            command = [
                sys.executable,
                str(Path(repo_path or ".") / predict_script),
                "--protein",
                str(payload["protein_path"]),
                "--ligand",
                str(payload["ligand"]),
                "--ref",
                str(payload["ref_pocket_center"]),
                "--device",
                str(payload.get("device", "cpu")),
                "--output_csv",
                str(csv_path),
                "--output_dir",
                str(request_dir),
            ]
        else:
            image = os.environ.get("FLEXPOSE_DOCKER_IMAGE", "flexpose-adapter:latest")
            command = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{request_dir}:/work/output",
                image,
                "python",
                "predict.py",
                "--protein",
                str(payload["protein_path"]),
                "--ligand",
                str(payload["ligand"]),
                "--ref",
                str(payload["ref_pocket_center"]),
                "--output_csv",
                "/work/output/predictions.csv",
            ]
        if payload.get("mock_outputs"):
            self._write_mock_csv(csv_path, request_dir, payload)
        else:
            run_command_allow_timeout(command, cwd=repo_path, timeout_seconds=int(payload.get("timeout_seconds", 240)))
        if not csv_path.exists():
            raise AcademicModelError(
                code=ErrorCode.OUTPUT_PARSE_FAILURE,
                message="FlexPose did not produce expected CSV output",
                safe_details={"expected_csv": str(csv_path)},
            )
        return {"csv_path": str(csv_path), "prepared": prepared, "command": command}

    def postprocess(self, raw_output: dict[str, Any]) -> dict[str, Any]:
        csv_path = Path(raw_output["csv_path"])
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except Exception as exc:
            raise AcademicModelError(
                code=ErrorCode.OUTPUT_PARSE_FAILURE,
                message="Unable to parse FlexPose CSV output",
                safe_details={"error_type": type(exc).__name__},
            ) from exc
        top = rows[0] if rows else {}
        raw_conf = float(top.get("confidence", top.get("pose_confidence", 0.0)) or 0.0)
        affinity = float(top.get("affinity", top.get("pred_affinity", 0.0)) or 0.0)
        rank = int(float(top.get("rank", top.get("pose_rank", 1)) or 1))
        pose_file = top.get("pose_file") or top.get("structure_path")
        return {
            "prediction": {
                "top_pose_rank": rank,
                "affinity": affinity,
                "runtime_metadata": {
                    "device": raw_output["prepared"]["payload"].get("device", "cpu"),
                    "energy_minimization": bool(
                        raw_output["prepared"]["payload"].get("energy_minimization", True)
                    ),
                    "ensemble_size": int(raw_output["prepared"]["payload"].get("ensemble_size", 10)),
                },
            },
            "explanations": {"wrapper_note": "Confidence normalization is wrapper logic."},
            "artifacts": {
                "output_dir": raw_output["prepared"]["request_dir"],
                "csv_path": str(csv_path),
                "structure_files": [pose_file] if pose_file else [],
                "invocation_command": raw_output["command"],
            },
            "confidence": {
                "raw_confidence": raw_conf,
                "normalized_confidence": normalize_confidence(raw_conf, 0.0, 1.0),
            },
        }

    def healthcheck(self, *, depth: str = "shallow") -> dict[str, Any]:
        status = {
            "status": "ok",
            "runtime_mode": self.runtime.mode,
            "repo_path_configured": bool(os.environ.get("FLEXPOSE_REPO_PATH")),
        }
        if depth == "deep":
            check = self.runtime_precheck({}, mode=self.runtime.mode)
            status["runtime_precheck"] = check
        return status

    def runtime_precheck(self, payload: dict[str, Any], *, mode: str) -> dict[str, Any]:
        selected = pick_mode(payload.get("runtime_mode"), mode)
        if payload.get("mock_outputs"):
            return {"mode": selected, "mock_outputs": True}
        runtime_result = check_runtime(
            mode=selected,
            min_python=self.runtime.python_version,
            torch_version=self.runtime.torch_version,
            gpu_required=self.runtime.gpu_required,
            repo_path=os.environ.get("FLEXPOSE_REPO_PATH"),
            repo_env_var="FLEXPOSE_REPO_PATH",
        )
        if selected == "container":
            ensure_docker_available()
        require_runtime_compatible(runtime_result)
        return runtime_result.details

    def predict_pose_and_affinity(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.predict(payload)

    def _write_mock_csv(self, csv_path: Path, request_dir: Path, payload: dict[str, Any]) -> None:
        pose_file = request_dir / "pose_rank1.pdb"
        if not pose_file.exists():
            pose_file.write_text("HEADER FLEXPOSE MOCK\nEND\n", encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["rank", "affinity", "confidence", "pose_file", "minimized"])
            writer.writeheader()
            writer.writerow(
                {
                    "rank": 1,
                    "affinity": -8.1,
                    "confidence": 0.78,
                    "pose_file": str(pose_file),
                    "minimized": bool(payload.get("energy_minimization", True)),
                }
            )

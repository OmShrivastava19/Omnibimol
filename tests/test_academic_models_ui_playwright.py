from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Response
import uvicorn

if os.getenv("ACADEMIC_UI_BROWSER_TESTS", "1") in {"0", "false", "False"}:
    pytest.skip("Browser tests disabled by ACADEMIC_UI_BROWSER_TESTS", allow_module_level=True)

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _mock_backend_app() -> FastAPI:
    app = FastAPI()

    @app.get("/api/v1/academic-models/models")
    def models():
        return [
            {"model_name": "flexpose", "paper_title": "FlexPose paper", "runtime_mode": "native", "supported_tasks": ["pose_prediction"], "input_schema_name": "v1", "explanation_support": "basic", "artifact_support": "files", "repo_url": "https://example.org/flexpose"},
            {"model_name": "deepathnet", "paper_title": "DeePathNet paper", "runtime_mode": "native", "supported_tasks": ["drug_response"], "input_schema_name": "v1", "explanation_support": "basic", "artifact_support": "files", "repo_url": "https://example.org/deepathnet"},
            {"model_name": "crispr-dipoff", "paper_title": "CRISPR paper", "runtime_mode": "native", "supported_tasks": ["offtarget_risk_scoring"], "input_schema_name": "v1", "explanation_support": "attribution", "artifact_support": "files", "repo_url": "https://example.org/crispr"},
            {"model_name": "deepdtagen", "paper_title": "DeepDTAGen paper", "runtime_mode": "native", "supported_tasks": ["affinity", "generate", "both"], "input_schema_name": "v1", "explanation_support": "basic", "artifact_support": "files", "repo_url": "https://example.org/deepdtagen"},
        ]

    @app.get("/api/v1/academic-models/health")
    def health(depth: str = "shallow"):
        statuses = {
            "flexpose": {"status": "ok", "depth": depth},
            "deepathnet": {"status": "degraded", "depth": depth},
            "crispr-dipoff": {"status": "ok", "depth": depth},
            "deepdtagen": {"status": "ok", "depth": depth},
        }
        return statuses

    @app.post("/api/v1/academic-models/predict")
    def predict(body: dict, response: Response):
        model_name = body.get("model_name")
        response.headers["x-request-id"] = f"mock-{model_name}"
        if model_name == "flexpose":
            return {
                "status": "success",
                "model": "flexpose",
                "prediction": {"affinity": -8.4, "runtime_metadata": {"duration_seconds": 2.1, "device": "cpu"}},
                "explanations": {},
                "confidence": {"raw_confidence": 0.81, "normalized_confidence": 0.81},
                "artifacts": {"pose_path": "tests/fixtures/academic_model_hub/out/flexpose/92171bb761bd/pose_rank1.pdb", "csv_path": "tests/fixtures/academic_model_hub/out/flexpose/92171bb761bd/predictions.csv"},
                "provenance": {"request_hash": "hash-flex"},
                "errors": [],
            }
        if model_name == "deepathnet":
            return {
                "status": "success",
                "model": "deepathnet",
                "prediction": {
                    "per_sample_predictions": [{"sample_id": "S1", "response_score": 0.5}],
                    "cohort_summary": {"n_samples": 1, "mean_response": 0.5, "std_response": 0.0},
                    "pathway_importance": [{"pathway": "PI3K_AKT", "importance": 0.42}],
                },
                "explanations": {},
                "confidence": {},
                "artifacts": {},
                "provenance": {"request_hash": "hash-deepath"},
                "errors": [],
            }
        if model_name == "crispr-dipoff":
            return {
                "status": "success",
                "model": "crispr-dipoff",
                "prediction": {
                    "model_prediction": {
                        "ranked_candidates": [
                            {"candidate_site": {"sequence": "ACGTACGTACGTACGTACGT", "chrom": "1", "pos": 1234, "gene": "GENE1"}, "offtarget_score": 0.9}
                        ]
                    },
                    "attribution_map": {0: 0.3, 1: 0.1, 2: 0.8},
                },
                "explanations": {},
                "confidence": {},
                "artifacts": {},
                "provenance": {"request_hash": "hash-crispr"},
                "errors": [],
            }
        return {
            "status": "success",
            "model": "deepdtagen",
            "prediction": {"task_mode": "both", "affinity_score": 0.18, "generated_molecules": ["CCC", "CCN"]},
            "explanations": {},
            "confidence": {"generation_summary": {"requested": 2, "valid_count": 2, "unique_count": 2}},
            "artifacts": {},
            "provenance": {"request_hash": "hash-deepdta", "payload": {"task_mode": "both"}},
            "errors": [],
        }

    return app


def _choose_model(page, model_name: str) -> None:
    page.get_by_role("combobox", name=re.compile("Model")).first.click()
    page.get_by_role("option", name=model_name).click()


def _click_run_prediction(page) -> None:
    button = page.get_by_role("button", name="Run Prediction")
    for _ in range(40):
        if button.is_enabled():
            button.click()
            return
        time.sleep(0.25)
    raise AssertionError("Run Prediction button did not become enabled")


@pytest.fixture(scope="module")
def ui_server():
    backend_port = _free_port()
    frontend_port = _free_port()

    config = uvicorn.Config(_mock_backend_app(), host="127.0.0.1", port=backend_port, log_level="error")
    backend_server = uvicorn.Server(config)
    backend_thread = threading.Thread(target=backend_server.run, daemon=True)
    backend_thread.start()
    backend_url = f"http://127.0.0.1:{backend_port}"
    for _ in range(25):
        try:
            ready = httpx.get(f"{backend_url}/api/v1/academic-models/models", timeout=1.5)
            if ready.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.2)

    env = os.environ.copy()
    env["BACKEND_API_URL"] = backend_url
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", str(frontend_port)],
        cwd=str(root),
        env=env,
    )
    for _ in range(30):
        try:
            home = httpx.get(f"http://127.0.0.1:{frontend_port}", timeout=1.5)
            if home.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.3)
    yield f"http://127.0.0.1:{frontend_port}"
    proc.terminate()
    backend_server.should_exit = True
    backend_thread.join(timeout=2)


def test_academic_models_browser_flows(ui_server) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(f"{ui_server}/?page=academic-models", wait_until="domcontentloaded")
        page.get_by_text("Academic Models").first.wait_for(timeout=45000)
        try:
            page.get_by_text("Model Discovery").wait_for(timeout=15000)
        except PlaywrightTimeoutError:
            if page.get_by_label("Open sidebar").count() > 0:
                page.get_by_label("Open sidebar").click()
            page.get_by_role("radio", name="🧠 Academic Models").click()
            page.get_by_text("Model Discovery").wait_for(timeout=30000)
        page.get_by_text("Submit Prediction").wait_for()
        page.get_by_text("Research support only. Outputs are not clinical recommendations.").wait_for()
        page.get_by_role("button", name="Run Deep Health Check").click()
        page.get_by_text("Deep Health Diagnostics").wait_for()

        _choose_model(page, "flexpose")
        page.get_by_label("Protein path").fill("tests/fixtures/academic_model_hub/tiny_protein.pdb")
        page.get_by_label("Ligand SMILES").fill("CCO")
        page.get_by_label("Reference pocket center path").fill("tests/fixtures/academic_model_hub/ref_ligand.mol2")
        _click_run_prediction(page)
        page.get_by_text("Backend request id").wait_for()
        page.get_by_text("Normalized confidence").wait_for()
        page.get_by_text("Quick artifacts").wait_for()
        page.get_by_text("mock-flexpose").first.wait_for()
        page.get_by_role("tab", name="Artifacts").click()
        page.get_by_role("button", name="Artifact: pose_path").wait_for()

        _choose_model(page, "deepathnet")
        page.get_by_label("Input table path").fill("tests/fixtures/academic_model_hub/mini_omics.csv")
        page.get_by_label("Pretrained model path").fill("tests/fixtures/academic_model_hub/deepathnet_weights.pth")
        page.get_by_label("Config path").fill("tests/fixtures/academic_model_hub/deepathnet_config.json")
        _click_run_prediction(page)
        page.get_by_text("Pathway importance").wait_for()
        page.get_by_role("tab", name="Artifacts").click()

        _choose_model(page, "crispr-dipoff")
        page.get_by_label("Guide RNA").fill("ACGTACGTACGTACGTACGT")
        _click_run_prediction(page)
        page.get_by_text("Guide attribution heatmap").wait_for()
        page.get_by_role("tab", name="Errors").click()
        page.get_by_text("No errors reported.").wait_for()

        _choose_model(page, "deepdtagen")
        page.get_by_label("Drug SMILES").fill("CCN(CC)CCC")
        page.get_by_label("Target sequence").fill("MTEITAAMVKELRESTGAGMMDCKNALSETQ")
        _click_run_prediction(page)
        page.get_by_text("Task mode").wait_for()
        page.get_by_text("Run History").wait_for()
        page.get_by_role("combobox", name=re.compile("Reopen run")).click()
        page.get_by_role("option").nth(1).click()
        page.get_by_text("Run Result:").wait_for()
        browser.close()

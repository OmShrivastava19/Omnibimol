from __future__ import annotations

from pathlib import Path

from academic_model_hub.adapters.crispr_dipoff_adapter import CrisprDipoffAdapter
from academic_model_hub.adapters.deepathnet_adapter import DeePathNetAdapter
from academic_model_hub.adapters.deepdtagen_adapter import DeepDTAGenAdapter
from academic_model_hub.adapters.flexpose_adapter import FlexPoseAdapter
from academic_model_hub.errors import ErrorCode
from academic_model_hub.hub import AcademicModelHub
from academic_model_hub.utils.provenance import request_hash


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "academic_model_hub"


def test_registry_and_list_models() -> None:
    hub = AcademicModelHub()
    models = hub.list_models()
    names = {item["model_name"] for item in models}
    assert {"flexpose", "deepathnet", "crispr-dipoff", "deepdtagen"}.issubset(names)


def test_flexpose_predict_normalized_response() -> None:
    adapter = FlexPoseAdapter()
    payload = {
        "protein_path": str(FIXTURE_DIR / "tiny_protein.pdb"),
        "ligand": "CCO",
        "ref_pocket_center": str(FIXTURE_DIR / "ref_ligand.mol2"),
        "device": "cpu",
        "ensemble_size": 2,
        "energy_minimization": True,
        "output_dir": str(FIXTURE_DIR / "out"),
        "mock_outputs": True,
    }
    result = adapter.predict(payload)
    assert result["status"] == "success"
    assert result["confidence"]["normalized_confidence"] <= 1.0
    assert "provenance_manifest" in result["artifacts"]


def test_deepathnet_preprocess_and_prediction() -> None:
    adapter = DeePathNetAdapter()
    payload = {
        "input_table_path": str(FIXTURE_DIR / "mini_omics.csv"),
        "task": "drug_response",
        "pretrained_model_path": str(FIXTURE_DIR / "deepathnet_weights.pth"),
        "config_path": str(FIXTURE_DIR / "deepathnet_config.json"),
        "return_pathway_importance": True,
        "return_gene_importance": False,
        "output_dir": str(FIXTURE_DIR / "out"),
        "mock_outputs": True,
    }
    result = adapter.predict(payload)
    assert result["status"] == "success"
    assert result["prediction"]["cohort_summary"]["mean_predicted_response"] >= 0.0


def test_crispr_dipoff_annotations_and_ranked_output() -> None:
    adapter = CrisprDipoffAdapter()
    payload = {
        "guide_rna": "ACGTACGTACGTACGTACGT",
        "candidate_sites": [
            {"sequence": "ACGTACGTACGTACGTACGT", "chrom": "1", "pos": 1234, "gene": "GENE1"},
            {"sequence": "ACGTACGTACGTACGTTCGT", "chrom": "1", "pos": 2234, "gene": "GENE2"},
        ],
        "pam": "NGG",
        "return_attributions": True,
        "output_dir": str(FIXTURE_DIR / "out"),
        "mock_outputs": True,
    }
    result = adapter.predict(payload)
    assert result["status"] == "success"
    assert "model_prediction" in result["prediction"]
    assert "wrapper_annotation" in result["prediction"]


def test_deepdtagen_both_mode_output() -> None:
    adapter = DeepDTAGenAdapter()
    payload = {
        "drug_input": {"smiles": "CCN(CC)CC"},
        "target_input": {"sequence": "MTEYKLVVVGAGGVGKSAL", "target_id": "EGFR"},
        "task_mode": "both",
        "num_generate": 5,
        "output_dir": str(FIXTURE_DIR / "out"),
        "mock_outputs": True,
    }
    result = adapter.predict(payload)
    assert result["status"] == "success"
    assert result["prediction"]["task_mode"] == "both"
    assert "affinity_score" in result["prediction"]
    assert result["confidence"]["generation_summary"]["requested"] == 5


def test_deepdtagen_task_mode_deterministic_branching() -> None:
    adapter = DeepDTAGenAdapter()
    base_payload = {
        "drug_input": {"smiles": "CCN(CC)CC"},
        "target_input": {"sequence": "MTEYKLVVVGAGGVGKSAL", "target_id": "EGFR"},
        "num_generate": 3,
        "output_dir": str(FIXTURE_DIR / "out"),
        "mock_outputs": True,
    }
    affinity_only = adapter.predict({**base_payload, "task_mode": "affinity"})
    assert affinity_only["status"] == "success"
    assert affinity_only["prediction"]["task_mode"] == "affinity"
    assert "affinity_score" in affinity_only["prediction"]
    assert "generated_molecules" not in affinity_only["prediction"]

    generate_only = adapter.predict({**base_payload, "task_mode": "generate"})
    assert generate_only["status"] == "success"
    assert generate_only["prediction"]["task_mode"] == "generate"
    assert "affinity_score" not in generate_only["prediction"]
    assert "generated_molecules" in generate_only["prediction"]


def test_error_mapping_for_invalid_payload() -> None:
    adapter = CrisprDipoffAdapter()
    result = adapter.predict(
        {
            "guide_rna": "INVALID!",
            "candidate_sites": [{"sequence": "AAAA", "chrom": "1", "pos": 1, "gene": "X"}],
            "pam": "NGG",
        }
    )
    assert result["status"] == "error"
    assert result["errors"][0]["code"] == ErrorCode.MALFORMED_BIOLOGICAL_INPUT.value


def test_request_hash_excludes_output_dir_volatility() -> None:
    payload_one = {"a": 1, "output_dir": "/tmp/one"}
    payload_two = {"a": 1, "output_dir": "/tmp/two"}
    assert request_hash("flexpose", payload_one) == request_hash("flexpose", payload_two)

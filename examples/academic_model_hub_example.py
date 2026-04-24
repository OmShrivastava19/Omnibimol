from __future__ import annotations

from pathlib import Path

from academic_model_hub import AcademicModelHub


def main() -> None:
    hub = AcademicModelHub()
    fixture_dir = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "academic_model_hub"
    payload = {
        "protein_path": str(fixture_dir / "tiny_protein.pdb"),
        "ligand": "CCO",
        "ref_pocket_center": str(fixture_dir / "ref_ligand.mol2"),
        "device": "cpu",
        "ensemble_size": 2,
        "energy_minimization": True,
        "output_dir": str(fixture_dir / "out"),
    }
    print(hub.predict("flexpose", payload))


if __name__ == "__main__":
    main()

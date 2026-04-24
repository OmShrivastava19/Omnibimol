"""Docking utilities for receptor caching and Vina job execution."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import httpx

from backend.core.config import Settings

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    AllChem = None

try:
    from vina import Vina
except Exception:  # pragma: no cover - optional dependency
    Vina = None


class DockingError(RuntimeError):
    """Base class for docking failures."""


class DockingUnavailableError(DockingError):
    """Raised when the worker stack cannot run real docking."""


class DockingConversionError(DockingError):
    """Raised when receptor or ligand conversion fails."""


class DockingExecutionError(DockingError):
    """Raised when Vina execution fails."""


RECEPTOR_PDBQT_FORMAT_VERSION = "pdbqt-v3"
RECEPTOR_CONVERTER_VERSION = "2026-04-24"
ALLOWED_RECEPTOR_ATOM_TYPES = {
    "H",
    "HD",
    "HS",
    "C",
    "A",
    "N",
    "NA",
    "NS",
    "OA",
    "OS",
    "F",
    "Mg",
    "P",
    "SA",
    "S",
    "Cl",
    "Ca",
    "Mn",
    "Fe",
    "Zn",
    "Br",
    "I",
    "Si",
    "B",
    "Cu",
    "Ni",
    "Co",
    "Se",
    "Mo",
    "Sn",
    "Na",
    "K",
}


@dataclass(frozen=True)
class ReceptorCacheRecord:
    cache_key: str
    receptor_pdbqt_path: Path
    metadata_path: Path
    cache_hit: bool
    metadata: dict[str, Any]


@contextmanager
def _exclusive_file_lock(lock_path: Path, timeout_seconds: int = 30) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout_seconds
    fd: int | None = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            break
        except FileExistsError:
            if time.time() >= deadline:
                if lock_path.exists() and time.time() - lock_path.stat().st_mtime > timeout_seconds:
                    try:
                        lock_path.unlink()
                    except OSError:
                        pass
                else:
                    raise DockingExecutionError(f"Timed out waiting for receptor cache lock: {lock_path}")
            time.sleep(0.1)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except OSError:
            pass


class DockingCache:
    """Persistent receptor cache for prepared PDBQT files."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _paths_for_key(self, cache_key: str) -> dict[str, Path]:
        return {
            "receptor_pdbqt": self.cache_dir / f"receptor_{cache_key}.pdbqt",
            "metadata": self.cache_dir / f"receptor_{cache_key}.json",
            "lock": self.cache_dir / f"receptor_{cache_key}.lock",
        }

    def cleanup(self, *, max_age_seconds: int = 30 * 24 * 3600, max_entries: int = 250) -> None:
        now = time.time()
        entries = sorted(self.cache_dir.glob("receptor_*.json"), key=lambda path: path.stat().st_mtime)
        for metadata_path in entries:
            try:
                if now - metadata_path.stat().st_mtime > max_age_seconds:
                    cache_key = metadata_path.stem.replace("receptor_", "")
                    paths = self._paths_for_key(cache_key)
                    for candidate in paths.values():
                        try:
                            candidate.unlink()
                        except OSError:
                            pass
            except OSError:
                continue

        if len(entries) > max_entries:
            for metadata_path in entries[: len(entries) - max_entries]:
                cache_key = metadata_path.stem.replace("receptor_", "")
                paths = self._paths_for_key(cache_key)
                for candidate in paths.values():
                    try:
                        candidate.unlink()
                    except OSError:
                        pass

    def _invalidate_entry(self, cache_key: str) -> None:
        paths = self._paths_for_key(cache_key)
        for candidate in (paths["receptor_pdbqt"], paths["metadata"]):
            try:
                candidate.unlink()
            except OSError:
                pass

    def load(self, cache_key: str) -> ReceptorCacheRecord | None:
        paths = self._paths_for_key(cache_key)
        if not paths["receptor_pdbqt"].exists() or not paths["metadata"].exists():
            return None
        try:
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        except Exception:
            self._invalidate_entry(cache_key)
            return None
        if metadata.get("format_version") != RECEPTOR_PDBQT_FORMAT_VERSION:
            self._invalidate_entry(cache_key)
            return None
        if metadata.get("converter_version") != RECEPTOR_CONVERTER_VERSION:
            self._invalidate_entry(cache_key)
            return None
        try:
            validate_receptor_pdbqt_file(paths["receptor_pdbqt"])
        except DockingConversionError:
            self._invalidate_entry(cache_key)
            return None
        metadata["cache_hit"] = True
        return ReceptorCacheRecord(
            cache_key=cache_key,
            receptor_pdbqt_path=paths["receptor_pdbqt"],
            metadata_path=paths["metadata"],
            cache_hit=True,
            metadata=metadata,
        )

    def store(self, cache_key: str, receptor_pdbqt_text: str, metadata: dict[str, Any]) -> ReceptorCacheRecord:
        paths = self._paths_for_key(cache_key)
        paths["receptor_pdbqt"].parent.mkdir(parents=True, exist_ok=True)

        temp_receptor = paths["receptor_pdbqt"].with_suffix(".pdbqt.tmp")
        temp_metadata = paths["metadata"].with_suffix(".json.tmp")
        temp_receptor.write_text(receptor_pdbqt_text, encoding="utf-8")
        temp_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp_receptor, paths["receptor_pdbqt"])
        os.replace(temp_metadata, paths["metadata"])

        metadata = dict(metadata)
        metadata["cache_hit"] = False
        return ReceptorCacheRecord(
            cache_key=cache_key,
            receptor_pdbqt_path=paths["receptor_pdbqt"],
            metadata_path=paths["metadata"],
            cache_hit=False,
            metadata=metadata,
        )

    def get_or_create(self, *, structure_id: str, source_url: str, pdb_text: str) -> ReceptorCacheRecord:
        content_hash = hashlib.sha256(pdb_text.encode("utf-8")).hexdigest()
        cache_key = hashlib.sha256(
            f"{RECEPTOR_PDBQT_FORMAT_VERSION}|{RECEPTOR_CONVERTER_VERSION}|{structure_id}|{source_url}|{content_hash}".encode("utf-8")
        ).hexdigest()[:24]
        paths = self._paths_for_key(cache_key)

        with _exclusive_file_lock(paths["lock"]):
            cached = self.load(cache_key)
            if cached is not None:
                return cached

            receptor_pdbqt_text = convert_pdb_to_pdbqt(pdb_text, structure_id=structure_id)
            metadata = {
                "format_version": RECEPTOR_PDBQT_FORMAT_VERSION,
                "converter_version": RECEPTOR_CONVERTER_VERSION,
                "structure_id": structure_id,
                "source_url": source_url,
                "content_hash": content_hash,
                "created_at_utc": datetime.now(UTC).isoformat(),
            }
            self.cleanup()
            return self.store(cache_key, receptor_pdbqt_text, metadata)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_pdb_element(atom_name: str, element_field: str) -> str:
    element = (element_field or "").strip()
    if element:
        return element[:2].upper()
    stripped = re.sub(r"[^A-Za-z]", "", atom_name).strip()
    if not stripped:
        return "C"
    if len(stripped) == 1:
        return stripped.upper()
    if stripped[:2].upper() in {"CL", "BR"}:
        return stripped[:2].upper()
    return stripped[0].upper()


def _atom_type_from_element(element: str) -> str:
    mapping = {
        "H": "H",
        "C": "C",
        "N": "N",
        "O": "OA",
        "S": "SA",
        "P": "P",
        "F": "F",
        "CL": "Cl",
        "BR": "Br",
        "I": "I",
        "MG": "Mg",
        "CA": "Ca",
        "MN": "Mn",
        "FE": "Fe",
        "ZN": "Zn",
        "SI": "Si",
        "B": "B",
        "CU": "Cu",
        "NI": "Ni",
        "CO": "Co",
        "SE": "Se",
        "MO": "Mo",
        "SN": "Sn",
        "NA": "Na",
        "K": "K",
    }
    normalized = element.upper().strip()
    if normalized in mapping:
        return mapping[normalized]
    if not normalized:
        return "C"
    return mapping.get(normalized[0], "C")


def _validate_receptor_atom_line(line: str, *, line_number: int) -> None:
    if len(line) < 78:
        raise DockingConversionError(
            f"Invalid receptor PDBQT ATOM line at {line_number}: expected fixed-width columns, got len={len(line)}"
        )

    coordinate_fields = {
        "x": line[30:38],
        "y": line[38:46],
        "z": line[46:54],
        "charge": line[68:76],
    }
    for field_name, field_value in coordinate_fields.items():
        try:
            float(field_value)
        except Exception as exc:
            raise DockingConversionError(
                f"Invalid receptor PDBQT {field_name} field at line {line_number}: {field_value!r}"
            ) from exc

    atom_type = line[77:79].strip()
    if atom_type not in ALLOWED_RECEPTOR_ATOM_TYPES:
        raise DockingConversionError(
            f"Invalid receptor PDBQT atom type at line {line_number}: {atom_type!r}"
        )


def validate_receptor_pdbqt_text(pdbqt_text: str) -> None:
    atom_count = 0
    for line_number, line in enumerate(pdbqt_text.splitlines(), start=1):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        _validate_receptor_atom_line(line, line_number=line_number)
        atom_count += 1

    if atom_count == 0:
        raise DockingConversionError("Receptor PDBQT validation failed: no ATOM/HETATM records found")


def validate_receptor_pdbqt_file(pdbqt_path: Path) -> None:
    try:
        pdbqt_text = pdbqt_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise DockingConversionError(f"Failed to read receptor PDBQT file: {pdbqt_path}") from exc
    validate_receptor_pdbqt_text(pdbqt_text)


def _format_pdbqt_atom_line(
    *,
    record_name: str,
    atom_index: int,
    atom_name: str,
    res_name: str,
    chain_id: str,
    res_seq: int,
    x: float,
    y: float,
    z: float,
    occupancy: float,
    temp_factor: float,
    charge: float,
    atom_type: str,
) -> str:
    # Follow canonical PDB/PDBQT fixed columns so Vina can parse coordinates reliably.
    line = (
        f"{record_name[:6]:<6}"
        f"{atom_index:5d} "
        f"{atom_name[:4]:>4}"
        f" "
        f"{res_name[:3]:>3}"
        f" "
        f"{(chain_id[:1] or 'A'):1}"
        f"{res_seq:4d}"
        f" "
        f"   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy:6.2f}{temp_factor:6.2f}"
        f"  "
        f"{charge:8.3f}"
        f" "
        f"{atom_type[:2]:<2}"
    )
    return line.ljust(80)


def convert_pdb_to_pdbqt(pdb_text: str, *, structure_id: str) -> str:
    """Convert a rigid receptor PDB file into a rigid receptor PDBQT file."""

    output_lines: list[str] = [f"REMARK  RECEPTOR {structure_id}"]
    atom_index = 1
    for line in pdb_text.splitlines():
        record = line[:6].strip().upper()
        if record not in {"ATOM", "HETATM"}:
            continue
        try:
            atom_name = line[12:16].strip() or "C"
            res_name = line[17:20].strip() or "RES"
            chain_id = line[21:22].strip() or "A"
            res_seq = int(line[22:26].strip() or "1")
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            occupancy = float(line[54:60].strip() or "1.00")
            temp_factor = float(line[60:66].strip() or "0.00")
            element = _parse_pdb_element(atom_name, line[76:78] if len(line) >= 78 else "")
        except Exception as exc:
            raise DockingConversionError(f"Failed to parse receptor atom line: {line!r}") from exc

        atom_type = _atom_type_from_element(element)
        output_lines.append(
            _format_pdbqt_atom_line(
                record_name="ATOM",
                atom_index=atom_index,
                atom_name=atom_name,
                res_name=res_name,
                chain_id=chain_id,
                res_seq=res_seq,
                x=x,
                y=y,
                z=z,
                occupancy=occupancy,
                temp_factor=temp_factor,
                charge=0.0,
                atom_type=atom_type,
            )
        )
        atom_index += 1

    if atom_index == 1:
        raise DockingConversionError("No receptor atoms were found in the PDB content")
    receptor_pdbqt = "\n".join(output_lines) + "\n"
    validate_receptor_pdbqt_text(receptor_pdbqt)
    return receptor_pdbqt


def _smiles_to_mol(smiles: str):
    if Chem is None or AllChem is None:
        raise DockingUnavailableError(
            "RDKit is required to convert SMILES to a dockable ligand but is not installed"
        )
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise DockingConversionError("Invalid SMILES string")
    molecule = Chem.AddHs(molecule)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(molecule, params) != 0:
        raise DockingConversionError("Unable to generate 3D conformer for ligand")
    AllChem.UFFOptimizeMolecule(molecule, maxIters=500)
    try:
        AllChem.ComputeGasteigerCharges(molecule)
    except Exception:
        pass
    return molecule


def _mol_to_rigid_pdbqt(molecule, *, molecule_name: str) -> str:
    conformer = molecule.GetConformer()
    output_lines: list[str] = [f"REMARK  LIGAND {molecule_name}", "ROOT"]
    for atom_index, atom in enumerate(molecule.GetAtoms(), start=1):
        position = conformer.GetAtomPosition(atom.GetIdx())
        element = atom.GetSymbol().upper()
        atom_type = _atom_type_from_element(element)
        try:
            charge = float(atom.GetProp("_GasteigerCharge"))
            if charge != charge:
                charge = 0.0
        except Exception:
            charge = 0.0
        output_lines.append(
            _format_pdbqt_atom_line(
                record_name="HETATM",
                atom_index=atom_index,
                atom_name=atom.GetSymbol(),
                res_name="LIG",
                chain_id="A",
                res_seq=1,
                x=position.x,
                y=position.y,
                z=position.z,
                occupancy=1.00,
                temp_factor=0.00,
                charge=charge,
                atom_type=atom_type,
            )
        )
    output_lines.extend(["ENDROOT", "TORSDOF 0"])
    return "\n".join(output_lines) + "\n"


def convert_smiles_to_pdbqt(smiles: str, *, molecule_name: str) -> str:
    molecule = _smiles_to_mol(smiles)
    return _mol_to_rigid_pdbqt(molecule, molecule_name=molecule_name)


def convert_sdf_to_pdbqt(sdf_text: str, *, molecule_name: str) -> str:
    if Chem is None or AllChem is None:
        raise DockingUnavailableError(
            "RDKit is required to convert SDF to a dockable ligand but is not installed"
        )
    molecule = Chem.MolFromMolBlock(sdf_text, sanitize=True, removeHs=False)
    if molecule is None:
        raise DockingConversionError("Unable to parse SDF content")
    try:
        AllChem.ComputeGasteigerCharges(molecule)
    except Exception:
        pass
    if molecule.GetNumConformers() == 0:
        raise DockingConversionError("SDF ligand is missing 3D coordinates")
    return _mol_to_rigid_pdbqt(molecule, molecule_name=molecule_name)


def _extract_vina_modes(vina_pose_text: str) -> list[dict[str, Any]]:
    modes: list[dict[str, Any]] = []
    current_mode: dict[str, Any] | None = None
    current_coords: list[tuple[float, float, float]] = []

    def flush_current_mode() -> None:
        nonlocal current_mode, current_coords
        if current_mode is None:
            return
        if current_coords:
            centroid = {
                "x": round(sum(x for x, _, _ in current_coords) / len(current_coords), 3),
                "y": round(sum(y for _, y, _ in current_coords) / len(current_coords), 3),
                "z": round(sum(z for _, _, z in current_coords) / len(current_coords), 3),
            }
        else:
            centroid = {"x": 0.0, "y": 0.0, "z": 0.0}
        current_mode["center"] = centroid
        current_mode.setdefault("orientation", f"Pose {current_mode['mode']}")
        modes.append(current_mode)
        current_mode = None
        current_coords = []

    for line in vina_pose_text.splitlines():
        if line.startswith("MODEL"):
            flush_current_mode()
            current_mode = {"mode": int(line.split()[1]), "affinity": 0.0, "rmsd_lb": 0.0, "rmsd_ub": 0.0}
            current_coords = []
        elif line.startswith("REMARK VINA RESULT:") and current_mode is not None:
            parts = line.replace("REMARK VINA RESULT:", "").split()
            if len(parts) >= 3:
                current_mode["affinity"] = round(float(parts[0]), 2)
                current_mode["rmsd_lb"] = round(float(parts[1]), 2)
                current_mode["rmsd_ub"] = round(float(parts[2]), 2)
        elif line.startswith(("ATOM", "HETATM")) and current_mode is not None:
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                current_coords.append((x, y, z))
            except Exception:
                continue
        elif line.startswith("ENDMDL") and current_mode is not None:
            flush_current_mode()

    flush_current_mode()
    modes.sort(key=lambda mode: mode.get("affinity", 0.0))
    for index, mode in enumerate(modes, start=1):
        mode["mode"] = index
        mode["orientation"] = mode.get("orientation") or f"Pose {index}"
    return modes


def _compute_box_from_pdb(pdb_text: str, padding: float = 4.0) -> tuple[dict[str, float], dict[str, float]]:
    coordinates: list[tuple[float, float, float]] = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                coordinates.append(
                    (
                        float(line[30:38].strip()),
                        float(line[38:46].strip()),
                        float(line[46:54].strip()),
                    )
                )
            except Exception:
                continue
    if not coordinates:
        raise DockingConversionError("Unable to compute docking box from receptor coordinates")
    xs = [coord[0] for coord in coordinates]
    ys = [coord[1] for coord in coordinates]
    zs = [coord[2] for coord in coordinates]
    center = {
        "x": round((min(xs) + max(xs)) / 2, 3),
        "y": round((min(ys) + max(ys)) / 2, 3),
        "z": round((min(zs) + max(zs)) / 2, 3),
    }
    size = {
        "x": round((max(xs) - min(xs)) + padding * 2, 3),
        "y": round((max(ys) - min(ys)) + padding * 2, 3),
        "z": round((max(zs) - min(zs)) + padding * 2, 3),
    }
    return center, size


class DockingProcessor:
    """Prepare docking inputs and execute AutoDock Vina."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.receptor_cache = DockingCache(settings.docking_cache_dir)

    def fetch_receptor_pdb(self, source_url: str) -> str:
        try:
            response = httpx.get(source_url, timeout=self.settings.docking_timeout_seconds, follow_redirects=True)
            response.raise_for_status()
        except Exception as exc:
            raise DockingExecutionError(f"Failed to fetch receptor structure from {source_url}") from exc
        return response.text

    def prepare_receptor(self, protein_payload: dict[str, Any]) -> dict[str, Any]:
        source_url = str(protein_payload.get("pdb_url") or protein_payload.get("source_url") or "")
        structure_id = str(
            protein_payload.get("structure_id")
            or protein_payload.get("uniprot_id")
            or protein_payload.get("protein_name")
            or "receptor"
        )
        if not source_url:
            raise DockingConversionError("No receptor source URL was provided")

        pdb_text = self.fetch_receptor_pdb(source_url)
        cached = self.receptor_cache.get_or_create(
            structure_id=structure_id,
            source_url=source_url,
            pdb_text=pdb_text,
        )
        return {
            "structure_id": structure_id,
            "source_url": source_url,
            "pdb_text": pdb_text,
            "receptor_cache_key": cached.cache_key,
            "receptor_pdbqt_path": str(cached.receptor_pdbqt_path),
            "cache_hit": cached.cache_hit,
            "cache_metadata": cached.metadata,
        }

    def prepare_ligand(self, ligand_payload: dict[str, Any]) -> dict[str, Any]:
        ligand_name = str(ligand_payload.get("name") or ligand_payload.get("ligand_name") or "Ligand")
        smiles = ligand_payload.get("smiles")
        sdf_data = ligand_payload.get("sdf_data")

        if smiles:
            ligand_pdbqt = convert_smiles_to_pdbqt(str(smiles), molecule_name=ligand_name)
            source_type = "smiles"
        elif sdf_data:
            ligand_pdbqt = convert_sdf_to_pdbqt(str(sdf_data), molecule_name=ligand_name)
            source_type = "sdf"
        else:
            raise DockingConversionError("Ligand input must include SMILES or SDF content")

        return {
            "name": ligand_name,
            "source_type": source_type,
            "smiles": smiles,
            "sdf_data": sdf_data,
            "pdbqt_text": ligand_pdbqt,
        }

    def run_vina(self, *, receptor_pdbqt_path: str, ligand_pdbqt_text: str, receptor_pdb_text: str,
                 exhaustiveness: int, num_modes: int, energy_range: int) -> dict[str, Any]:
        if Vina is None:
            raise DockingUnavailableError(
                "The vina Python package is not installed, so real docking cannot run"
            )

        center, size = _compute_box_from_pdb(receptor_pdb_text)
        validate_receptor_pdbqt_file(Path(receptor_pdbqt_path))
        with tempfile.TemporaryDirectory(prefix="omnibimol-docking-") as temp_dir:
            temp_dir_path = Path(temp_dir)
            ligand_pdbqt_path = temp_dir_path / "ligand.pdbqt"
            output_pdbqt_path = temp_dir_path / "docked.pdbqt"
            ligand_pdbqt_path.write_text(ligand_pdbqt_text, encoding="utf-8")

            vina = Vina(sf_name="vina")
            vina.set_receptor(receptor_pdbqt_path)
            vina.set_ligand_from_file(str(ligand_pdbqt_path))
            vina.compute_vina_maps(center=[center["x"], center["y"], center["z"]], box_size=[size["x"], size["y"], size["z"]])
            vina.dock(exhaustiveness=exhaustiveness, n_poses=num_modes)
            vina.write_poses(str(output_pdbqt_path), n_poses=num_modes, energy_range=energy_range)
            docked_text = output_pdbqt_path.read_text(encoding="utf-8")

        modes = _extract_vina_modes(docked_text)
        if not modes:
            raise DockingExecutionError("Vina completed but no pose information was produced")

        return {
            "binding_affinity": modes[0]["affinity"],
            "modes": modes,
            "best_mode": modes[0],
            "has_coordinates": True,
            "docked_pdbqt": docked_text,
            "docking_box": {"center": center, "size": size},
        }

    def process_job_payload(self, job_payload: dict[str, Any]) -> dict[str, Any]:
        protein_payload = dict(job_payload.get("protein") or job_payload.get("protein_context") or {})
        ligand_payload = dict(job_payload.get("ligand") or job_payload.get("ligand_context") or {})
        parameters = dict(job_payload.get("parameters") or {})

        receptor = self.prepare_receptor(protein_payload)
        ligand = self.prepare_ligand(ligand_payload)

        exhaustiveness = int(parameters.get("exhaustiveness", 8))
        num_modes = int(parameters.get("num_modes", 9))
        energy_range = int(parameters.get("energy_range", 3))

        docking_result = self.run_vina(
            receptor_pdbqt_path=receptor["receptor_pdbqt_path"],
            ligand_pdbqt_text=ligand["pdbqt_text"],
            receptor_pdb_text=receptor["pdb_text"],
            exhaustiveness=exhaustiveness,
            num_modes=num_modes,
            energy_range=energy_range,
        )

        result_payload = {
            "available": True,
            "mode": "real",
            "simulated": False,
            "engine": self.settings.docking_engine,
            "status": "completed",
            "binding_affinity": docking_result["binding_affinity"],
            "modes": docking_result["modes"],
            "best_mode": docking_result["best_mode"],
            "has_coordinates": docking_result["has_coordinates"],
            "docking_box": docking_result["docking_box"],
            "receptor_cache_hit": receptor["cache_hit"],
            "receptor_cache_key": receptor["receptor_cache_key"],
            "receptor_metadata": receptor["cache_metadata"],
            "ligand_name": ligand["name"],
            "ligand_source_type": ligand["source_type"],
            "exhaustiveness": exhaustiveness,
            "num_modes": num_modes,
            "energy_range": energy_range,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        return result_payload

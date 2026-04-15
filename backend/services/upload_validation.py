"""Validation helpers for FASTA/VCF and upload payload constraints."""

from pathlib import Path

from backend.core.errors import InputValidationError

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
FASTA_EXTENSIONS = {".fasta", ".fa", ".fna"}
VCF_EXTENSIONS = {".vcf"}
ALLOWED_CONTENT_TYPES = {
    "text/plain",
    "text/x-fasta",
    "application/octet-stream",
    "text/x-vcard",  # Some clients mislabel VCF payloads
}


def validate_upload_payload(
    *,
    filename: str,
    content: str,
    content_type: str | None = None,
) -> dict[str, object]:
    raw = content.encode("utf-8")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise InputValidationError(
            "Upload exceeds maximum allowed size",
            details={"max_bytes": MAX_UPLOAD_BYTES, "received_bytes": len(raw)},
        )

    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise InputValidationError(
            "Unsupported content type",
            details={"content_type": content_type},
        )

    extension = Path(filename).suffix.lower()
    if extension in FASTA_EXTENSIONS:
        _validate_fasta(content)
        return {"format": "fasta", "bytes": len(raw)}
    if extension in VCF_EXTENSIONS:
        _validate_vcf(content)
        return {"format": "vcf", "bytes": len(raw)}

    raise InputValidationError(
        "Unsupported file extension",
        details={"extension": extension},
    )


def _validate_fasta(content: str) -> None:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines or not lines[0].startswith(">"):
        raise InputValidationError("Invalid FASTA: missing header line starting with '>'")

    allowed_chars = set("ACGTNURYKMSWBDHVX-*")
    for idx, line in enumerate(lines[1:], start=2):
        if line.startswith(">"):
            continue
        if not set(line.upper()).issubset(allowed_chars):
            raise InputValidationError(
                "Invalid FASTA: sequence contains unsupported characters",
                details={"line": idx},
            )


def _validate_vcf(content: str) -> None:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    has_header = any(line.startswith("##fileformat=VCF") for line in lines)
    has_chrom = any(line.startswith("#CHROM") for line in lines)
    if not has_header or not has_chrom:
        raise InputValidationError("Invalid VCF: missing required VCF headers")

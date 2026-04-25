"""Async job endpoints for heavy workloads."""

import re
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.orm import Session

from backend.audit.service import AuditService
from backend.auth.dependencies import get_current_principal
from backend.auth.rbac import get_tenant_context, require_permission
from backend.auth.token_verifier import AuthPrincipal
from backend.db.job_repository import JobRepository
from backend.db.session import get_db
from backend.services.job_service import JobService

router = APIRouter(prefix="/jobs", tags=["jobs"])

# Sequence validation patterns
PROTEIN_ALPHABET = set('ACDEFGHIKLMNPQRSTVWYX*-')
DNA_ALPHABET = set('ACGTN-')
FASTA_HEADER_PATTERN = re.compile(r'^[>][\w\-\s\.()]+$')
SMILES_PATTERN = re.compile(r'^[A-Za-z0-9\[\]\(\)\\=@#+\-]+$')


class ProteinSequenceInput(BaseModel):
    """Protein sequence with validation for length and format."""
    sequence: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="Protein sequence (10-10,000 amino acids). Valid characters: ACDEFGHIKLMNPQRSTVWYX*-"
    )
    format: Literal["fasta", "raw"] = Field(
        "raw",
        description="Sequence format: 'fasta' for FASTA format, 'raw' for plain sequence"
    )

    @field_validator('sequence')
    @classmethod
    def validate_protein_sequence(cls, v: str) -> str:
        """Validate protein sequence contains only valid amino acid characters."""
        v_clean = v.strip().upper()
        # Remove FASTA header if present
        if v_clean.startswith('>'):
            lines = v_clean.split('\n')
            v_clean = ''.join(lines[1:]).replace('\n', '')
        
        if not v_clean:
            raise ValueError("Protein sequence cannot be empty")
        
        invalid_chars = set(v_clean) - PROTEIN_ALPHABET
        if invalid_chars:
            raise ValueError(
                f"Invalid protein sequence characters: {', '.join(sorted(invalid_chars))}. "
                f"Valid amino acids: ACDEFGHIKLMNPQRSTVWYX*-"
            )
        return v_clean


class DNASequenceInput(BaseModel):
    """DNA sequence with validation for length and format."""
    sequence: str = Field(
        ...,
        min_length=20,
        max_length=1000000,
        description="DNA sequence (20-1,000,000 base pairs). Valid characters: ACGTN-"
    )
    format: Literal["fasta", "raw"] = Field(
        "raw",
        description="Sequence format: 'fasta' for FASTA format, 'raw' for plain sequence"
    )

    @field_validator('sequence')
    @classmethod
    def validate_dna_sequence(cls, v: str) -> str:
        """Validate DNA sequence contains only valid nucleotide characters."""
        v_clean = v.strip().upper()
        # Remove FASTA header if present
        if v_clean.startswith('>'):
            lines = v_clean.split('\n')
            v_clean = ''.join(lines[1:]).replace('\n', '')
        
        if not v_clean:
            raise ValueError("DNA sequence cannot be empty")
        
        invalid_chars = set(v_clean) - DNA_ALPHABET
        if invalid_chars:
            raise ValueError(
                f"Invalid DNA sequence characters: {', '.join(sorted(invalid_chars))}. "
                f"Valid nucleotides: ACGTN-"
            )
        return v_clean


class DockingJobPayload(BaseModel):
    """Payload for molecular docking jobs."""
    protein_sequence: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="Protein sequence (10-10,000 amino acids)"
    )
    ligand_smiles: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Ligand SMILES string for docking"
    )
    exhaustiveness: int = Field(
        8,
        ge=1,
        le=32,
        description="Search exhaustiveness (1-32, higher=more thorough but slower)"
    )
    num_modes: int = Field(
        9,
        ge=1,
        le=20,
        description="Number of conformations to generate (1-20)"
    )

    @field_validator('protein_sequence')
    @classmethod
    def validate_protein_seq(cls, v: str) -> str:
        """Validate protein sequence format."""
        v_clean = v.strip().upper().replace('\n', '')
        if v_clean.startswith('>'):
            lines = v_clean.split('\n')
            v_clean = ''.join(lines[1:])
        
        invalid_chars = set(v_clean) - PROTEIN_ALPHABET
        if invalid_chars:
            raise ValueError(
                f"Invalid protein sequence characters: {', '.join(sorted(invalid_chars))}"
            )
        return v_clean

    @field_validator('ligand_smiles')
    @classmethod
    def validate_smiles(cls, v: str) -> str:
        """Validate SMILES format."""
        v = v.strip()
        if not SMILES_PATTERN.match(v):
            raise ValueError(
                "Invalid SMILES string. Must contain only valid chemical notation "
                "(alphanumeric, brackets, parentheses, =, @, #, +, -, backslash)"
            )
        return v


class ProteinAnalysisJobPayload(BaseModel):
    """Payload for protein analysis jobs."""
    protein_sequence: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="Protein sequence (10-10,000 amino acids)"
    )
    analysis_type: Literal["structure", "function", "pathways", "interactions"] = Field(
        "structure",
        description="Type of analysis to perform"
    )
    include_conservation: bool = Field(
        True,
        description="Whether to include sequence conservation analysis"
    )

    @field_validator('protein_sequence')
    @classmethod
    def validate_protein_seq(cls, v: str) -> str:
        """Validate protein sequence format."""
        v_clean = v.strip().upper().replace('\n', '')
        if v_clean.startswith('>'):
            v_clean = '\n'.join(v_clean.split('\n')[1:])
        
        invalid_chars = set(v_clean) - PROTEIN_ALPHABET
        if invalid_chars:
            raise ValueError(
                f"Invalid protein sequence characters: {', '.join(sorted(invalid_chars))}"
            )
        return v_clean


class GenomeAnalysisJobPayload(BaseModel):
    """Payload for genome analysis jobs."""
    dna_sequence: str = Field(
        ...,
        min_length=20,
        max_length=1000000,
        description="DNA sequence (20-1,000,000 base pairs)"
    )
    analysis_type: Literal["variants", "annotation", "pathways"] = Field(
        "variants",
        description="Type of genome analysis to perform"
    )
    include_risk_scoring: bool = Field(
        False,
        description="Whether to include disease risk scoring"
    )

    @field_validator('dna_sequence')
    @classmethod
    def validate_dna_seq(cls, v: str) -> str:
        """Validate DNA sequence format."""
        v_clean = v.strip().upper().replace('\n', '')
        if v_clean.startswith('>'):
            v_clean = '\n'.join(v_clean.split('\n')[1:])
        
        invalid_chars = set(v_clean) - DNA_ALPHABET
        if invalid_chars:
            raise ValueError(
                f"Invalid DNA sequence characters: {', '.join(sorted(invalid_chars))}"
            )
        return v_clean


class DrugRepurposingJobPayload(BaseModel):
    """Payload for drug repurposing jobs."""
    target_sequence: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="Target protein sequence"
    )
    disease_context: str | None = Field(
        None,
        max_length=500,
        description="Optional disease context or indication"
    )
    max_candidates: int = Field(
        10,
        ge=1,
        le=100,
        description="Maximum number of drug candidates to return (1-100)"
    )

    @field_validator('target_sequence')
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        """Validate protein sequence format."""
        v_clean = v.strip().upper().replace('\n', '')
        if v_clean.startswith('>'):
            v_clean = '\n'.join(v_clean.split('\n')[1:])
        
        invalid_chars = set(v_clean) - PROTEIN_ALPHABET
        if invalid_chars:
            raise ValueError(f"Invalid protein sequence characters: {', '.join(sorted(invalid_chars))}")
        return v_clean


class JobEnqueueRequest(BaseModel):
    """Generic job enqueue request with type-specific payload validation."""
    job_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Job type identifier (e.g., 'docking', 'protein_analysis', 'genome_analysis')"
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Job-specific payload (structure depends on job_type)"
    )
    runtime_mode: Literal["native", "simulation"] = Field(
        "native",
        description="Execution mode: 'native' for production, 'simulation' for testing"
    )

    @field_validator('job_type')
    @classmethod
    def validate_job_type(cls, v: str) -> str:
        """Validate job type format."""
        v = v.strip().lower()
        if not re.match(r'^[a-z0-9_]+$', v):
            raise ValueError(
                "Job type must contain only lowercase alphanumeric characters and underscores"
            )
        valid_types = {
            "docking", "protein_analysis", "genome_analysis", 
            "drug_repurposing", "target_prioritization", "portfolio_analysis"
        }
        if v not in valid_types:
            raise ValueError(
                f"Unknown job type '{v}'. Valid types: {', '.join(sorted(valid_types))}"
            )
        return v

    @field_validator('payload')
    @classmethod
    def validate_payload_size(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate payload is not excessively large."""
        # Rough check for payload size (serialized)
        import json
        try:
            serialized = json.dumps(v)
            if len(serialized) > 10 * 1024 * 1024:  # 10MB limit
                raise ValueError("Payload exceeds maximum size of 10MB")
        except TypeError as e:
            raise ValueError(f"Payload contains non-serializable objects: {e}")
        return v

    @model_validator(mode='after')
    def validate_payload_against_job_type(self) -> 'JobEnqueueRequest':
        """Validate payload structure matches job_type."""
        if self.job_type == 'docking':
            try:
                DockingJobPayload(**self.payload)
            except ValueError as e:
                raise ValueError(f"Invalid docking job payload: {e}")
        elif self.job_type == 'protein_analysis':
            try:
                ProteinAnalysisJobPayload(**self.payload)
            except ValueError as e:
                raise ValueError(f"Invalid protein analysis job payload: {e}")
        elif self.job_type == 'genome_analysis':
            try:
                GenomeAnalysisJobPayload(**self.payload)
            except ValueError as e:
                raise ValueError(f"Invalid genome analysis job payload: {e}")
        elif self.job_type == 'drug_repurposing':
            try:
                DrugRepurposingJobPayload(**self.payload)
            except ValueError as e:
                raise ValueError(f"Invalid drug repurposing job payload: {e}")
        # Other job types can have flexible payload structures
        return self


def _serialize_job(job) -> dict[str, object]:
    return {
        "id": job.id,
        "tenant_id": job.tenant_id,
        "requested_by_user_id": job.requested_by_user_id,
        "job_type": job.job_type,
        "status": job.status,
        "input_payload": job.input_payload,
        "result_payload": job.result_payload,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


@router.post("")
def enqueue_job(
    body: JobEnqueueRequest,
    _: AuthPrincipal = Depends(require_permission("project.write")),
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Enqueue a new async job for processing.
    
    Supports job types:
    - docking: Molecular docking simulations
    - protein_analysis: Protein structure and function analysis
    - genome_analysis: Genomic variant and annotation analysis
    - drug_repurposing: Drug repositioning screening
    - target_prioritization: Therapeutic target ranking
    - portfolio_analysis: Multi-target portfolio optimization
    
    Args:
        body: JobEnqueueRequest with job_type, payload, and runtime_mode
        
    Returns:
        Job metadata with ID, status, and timestamps
    """
    tenant_context = get_tenant_context(db, principal)
    repo = JobRepository(db)
    
    # Include runtime_mode in payload metadata
    payload_with_mode = {
        **body.payload,
        "_runtime_mode": body.runtime_mode
    }
    
    job = JobService(repo).enqueue(
        tenant_id=tenant_context.tenant_id,
        requested_by_user_id=tenant_context.user_id,
        job_type=body.job_type,
        input_payload=payload_with_mode,
    )
    AuditService(db).log_event(
        tenant_id=tenant_context.tenant_id,
        actor_id=tenant_context.user_id,
        action="job.enqueued",
        resource_type="job",
        resource_id=str(job.id),
        ip_address=None,
        user_agent=None,
        details={"job_type": body.job_type, "runtime_mode": body.runtime_mode},
        commit=True,
    )
    return _serialize_job(job)


@router.get("/{job_id}")
def get_job_status(
    job_id: int,
    _: AuthPrincipal = Depends(require_permission("project.read")),
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Get the current status and results of an async job.
    
    Args:
        job_id: Numeric ID of the job
        
    Returns:
        Job metadata including status, results, and error messages if applicable
    """
    tenant_context = get_tenant_context(db, principal)
    repo = JobRepository(db)
    job = repo.get_job_for_tenant(tenant_id=tenant_context.tenant_id, job_id=job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    previous_status = job.status
    job = JobService(repo).advance(job)
    if previous_status != job.status and job.status in {"completed", "failed"}:
        action = "job.completed" if job.status == "completed" else "job.failed"
        AuditService(db).log_event(
            tenant_id=tenant_context.tenant_id,
            actor_id=tenant_context.user_id,
            action=action,
            resource_type="job",
            resource_id=str(job.id),
            ip_address=None,
            user_agent=None,
            details={"job_type": job.job_type, "status": job.status},
            commit=True,
        )
    return _serialize_job(job)

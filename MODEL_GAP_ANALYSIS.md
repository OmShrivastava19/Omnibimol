# Model Gap Analysis

## FlexPose
- Paper claims: flexible protein-ligand pose prediction with confidence and affinity.
- Repo clearly supports: scripted prediction workflow with structure artifacts.
- Wrapper exposes: native/container invocation paths, deterministic request directories, real CSV parsing, provenance/run summary manifests.
- Now supported end-to-end: payload validation -> runtime precheck -> upstream command execution -> normalized output parsing.
- Remaining blockers: local upstream clone, script path compatibility, and checkpoint availability are still environment-dependent.
- Needs lab collaboration: canonical confidence calibration across benchmark sets and official output field stability across releases.

## DeePathNet
- Paper claims: pathway-aware transformer for multi-omics drug response and importance.
- Repo clearly supports: config/script-driven inference and pretrained-model evaluation.
- Wrapper exposes: CSV/TSV ingestion, feature-coverage diagnostics, native/container script invocation, normalized cohort outputs.
- Now supported end-to-end: schema checks -> script execution -> prediction file parse -> pathway-centric summaries.
- Remaining blockers: gene-importance outputs vary by upstream script/options; optional calibration mappings require lab conventions.
- Needs lab collaboration: authoritative task/config compatibility matrix and calibrated probability mapping standards.

## CRISPR-DIPOFF
- Paper claims: interpretable off-target prediction with attribution-centric reasoning.
- Repo clearly supports: sequence-based risk inference with explainability hooks.
- Wrapper exposes: temporary dataset generation, native/container invocation, attribution parsing, wrapper annotation separation.
- Now supported end-to-end: DNA/PAM validation -> upstream inference path -> ranked risk normalization with attribution-aware interpretation.
- Remaining blockers: exact attribution format can differ by upstream version; external genomic impact tiers remain optional wrapper extensions.
- Needs lab collaboration: validated interpretation thresholds and recommended annotation joins for patient-context ranking.

## DeepDTAGen
- Paper claims: multitask affinity prediction and target-aware molecule generation.
- Repo clearly supports: dual-task modeling workflow with generation path.
- Wrapper exposes: native/container dual-task routing (`affinity|generate|both`), sequence validation, output parsing, wrapper-labeled filters.
- Now supported end-to-end: runtime precheck -> upstream multitask command -> normalized affinity/generation contract.
- Remaining blockers: full upstream conditioning diagnostics and richer chemistry validation exports depend on exact repo capabilities.
- Needs lab collaboration: deployment-grade checkpoints and model-approved generation quality/novelty metrics.

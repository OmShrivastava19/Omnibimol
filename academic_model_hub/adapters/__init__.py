"""Academic model adapters."""

from academic_model_hub.adapters.crispr_dipoff_adapter import CrisprDipoffAdapter
from academic_model_hub.adapters.deepathnet_adapter import DeePathNetAdapter
from academic_model_hub.adapters.deepdtagen_adapter import DeepDTAGenAdapter
from academic_model_hub.adapters.flexpose_adapter import FlexPoseAdapter

__all__ = [
    "FlexPoseAdapter",
    "DeePathNetAdapter",
    "CrisprDipoffAdapter",
    "DeepDTAGenAdapter",
]

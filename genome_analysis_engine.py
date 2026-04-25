"""
Genome Analysis Engine - Sequence-based disease risk prediction and biomarker detection
Provides mutation analysis, biomarker detection, disease association mapping, and personalized insights
"""

import re
import hashlib
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json
import numpy as np
from enum import Enum

try:
    from variant_therapy_engine import VariantTherapyEngine
except ImportError:
    VariantTherapyEngine = None


class VariantType(Enum):
    """Classification of genetic variants"""
    PATHOGENIC = "Pathogenic"
    LIKELY_PATHOGENIC = "Likely Pathogenic"
    UNCERTAIN_SIGNIFICANCE = "Uncertain Significance"
    LIKELY_BENIGN = "Likely Benign"
    BENIGN = "Benign"
    RISK_FACTOR = "Risk Factor"
    PROTECTIVE = "Protective"


class ConfidenceLevel(Enum):
    """Confidence levels for predictions"""
    VERY_HIGH = "Very High"
    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"
    VERY_LOW = "Very Low"


@dataclass
class Variant:
    """Represents a genetic variant detected in sequence"""
    gene: str
    variant_id: str
    type: str
    description: str
    position: int = 0
    reference: str = ""
    alternate: str = ""
    sequence_match: str = ""
    confidence: float = 0.8  # 0-1
    pathogenicity_score: Optional[float] = None
    pathogenicity_tier: Optional[int] = None
    pathogenicity_method: str = ""
    model_confidence: str = ""
    evidence_summary: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'gene': self.gene,
            'variant_id': self.variant_id,
            'type': self.type,
            'description': self.description,
            'position': self.position,
            'reference': self.reference,
            'alternate': self.alternate,
            'sequence_match': self.sequence_match,
            'confidence': self.confidence,
            'pathogenicity_score': self.pathogenicity_score,
            'pathogenicity_tier': self.pathogenicity_tier,
            'pathogenicity_method': self.pathogenicity_method,
            'model_confidence': self.model_confidence,
            'evidence_summary': self.evidence_summary,
        }


@dataclass
class Biomarker:
    """Represents a detected biomarker in the sequence"""
    name: str
    biomarker_type: str
    location: str
    sequence_pattern: str
    position: int = 0
    length: int = 0
    match_strength: float = 1.0  # 0-1, how well it matches
    associated_diseases: List[str] = field(default_factory=list)
    clinical_significance: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'type': self.biomarker_type,
            'location': self.location,
            'pattern': self.sequence_pattern,
            'position': self.position,
            'length': self.length,
            'match_strength': self.match_strength,
            'diseases': self.associated_diseases,
            'significance': self.clinical_significance
        }


@dataclass
class DiseaseAssociation:
    """Represents association between detected variants/biomarkers and a disease"""
    disease: str
    risk_score: float  # 0-100
    confidence: ConfidenceLevel
    detected_variants: List[Variant] = field(default_factory=list)
    detected_biomarkers: List[Biomarker] = field(default_factory=list)
    inheritance_pattern: str = ""
    prevalence: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'disease': self.disease,
            'risk_score': self.risk_score,
            'confidence': self.confidence.value,
            'variants': len(self.detected_variants),
            'biomarkers': len(self.detected_biomarkers),
            'inheritance': self.inheritance_pattern,
            'prevalence': self.prevalence
        }


class MutationAnalyzer:
    """Analyzes sequences for known pathogenic variants and disease-causing genes"""
    
    # Disease-causing genes database
    DISEASE_GENES = {
        'BRCA1': {
            'diseases': ['Breast Cancer', 'Ovarian Cancer', 'Prostate Cancer'],
            'inheritance': 'Autosomal Dominant',
            'penetrance': 0.72,
            'variants': [
                {'id': 'c.68_69delAG', 'type': 'Frameshift', 'pathogenicity': 'Pathogenic'},
                {'id': 'c.5266dupC', 'type': 'Frameshift', 'pathogenicity': 'Pathogenic'},
                {'id': '5382insC', 'type': 'Frameshift', 'pathogenicity': 'Pathogenic'},
            ]
        },
        'BRCA2': {
            'diseases': ['Breast Cancer', 'Ovarian Cancer', 'Pancreatic Cancer'],
            'inheritance': 'Autosomal Dominant',
            'penetrance': 0.62,
            'variants': [
                {'id': '6174delT', 'type': 'Frameshift', 'pathogenicity': 'Pathogenic'},
                {'id': 'c.9097C>T', 'type': 'Nonsense', 'pathogenicity': 'Pathogenic'},
            ]
        },
        'TP53': {
            'diseases': ['Breast Cancer', 'Colorectal Cancer', 'Sarcoma'],
            'inheritance': 'Autosomal Dominant',
            'penetrance': 0.73,
            'variants': [
                {'id': 'R175H', 'type': 'Missense', 'pathogenicity': 'Pathogenic'},
                {'id': 'c.215C>G', 'type': 'Missense', 'pathogenicity': 'Pathogenic'},
            ]
        },
        'APOE': {
            'diseases': ["Alzheimer's Disease"],
            'inheritance': 'Complex',
            'penetrance': 0.35,
            'variants': [
                {'id': 'ε4 allele', 'type': 'SNP', 'pathogenicity': 'Risk Factor'},
                {'id': 'ε2 allele', 'type': 'SNP', 'pathogenicity': 'Protective'},
            ]
        },
        'CFTR': {
            'diseases': ['Cystic Fibrosis'],
            'inheritance': 'Autosomal Recessive',
            'penetrance': 1.0,
            'variants': [
                {'id': 'F508del', 'type': 'Deletion', 'pathogenicity': 'Pathogenic'},
                {'id': 'G551D', 'type': 'Missense', 'pathogenicity': 'Pathogenic'},
            ]
        },
        'HFE': {
            'diseases': ['Hemochromatosis'],
            'inheritance': 'Autosomal Recessive',
            'penetrance': 0.10,
            'variants': [
                {'id': 'C282Y', 'type': 'Missense', 'pathogenicity': 'Pathogenic'},
                {'id': 'H63D', 'type': 'Missense', 'pathogenicity': 'Likely Benign'},
            ]
        },
        'FTO': {
            'diseases': ['Obesity', 'Type 2 Diabetes'],
            'inheritance': 'Complex',
            'penetrance': 0.15,
            'variants': [
                {'id': 'rs9939609', 'type': 'SNP', 'pathogenicity': 'Risk Factor'},
            ]
        },
        'TCF7L2': {
            'diseases': ['Type 2 Diabetes'],
            'inheritance': 'Complex',
            'penetrance': 0.25,
            'variants': [
                {'id': 'rs7903146', 'type': 'SNP', 'pathogenicity': 'Risk Factor'},
            ]
        },
        'MTHFR': {
            'diseases': ['Neural Tube Defects', 'Thrombosis'],
            'inheritance': 'Autosomal Recessive',
            'penetrance': 0.05,
            'variants': [
                {'id': 'C677T', 'type': 'Missense', 'pathogenicity': 'Risk Factor'},
                {'id': 'A1298C', 'type': 'Missense', 'pathogenicity': 'Risk Factor'},
            ]
        },
        'LDLR': {
            'diseases': ['Familial Hypercholesterolemia'],
            'inheritance': 'Autosomal Dominant',
            'penetrance': 0.9,
            'variants': [
                {'id': 'Exon 2-6 deletions', 'type': 'Deletion', 'pathogenicity': 'Pathogenic'},
            ]
        }
    }
    
    def analyze_mutations(self, sequence: str) -> List[Variant]:
        """
        Analyze sequence for known pathogenic variants and disease-causing genes.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            List of detected variants
        """
        detected_variants = []
        sequence_upper = sequence.upper()
        
        # Search for known disease genes in sequence (simplified pattern matching)
        for gene, gene_info in self.DISEASE_GENES.items():
            # Try to find gene signatures (simplified - in real implementation would use actual genome positions)
            gene_patterns = self._generate_gene_patterns(gene)
            
            for pattern, variant_id in gene_patterns:
                if pattern in sequence_upper:
                    # Found this gene in sequence
                    position = sequence_upper.find(pattern)
                    confidence = 0.8 + np.random.random() * 0.2  # 0.8-1.0
                    
                    variant = Variant(
                        gene=gene,
                        variant_id=variant_id,
                        type=self._get_variant_type(gene_info),
                        description=f"Potential {gene} variant: {variant_id}",
                        position=position,
                        sequence_match=pattern,
                        confidence=confidence
                    )
                    detected_variants.append(variant)
        
        return detected_variants
    
    def _generate_gene_patterns(self, gene: str) -> List[Tuple[str, str]]:
        """Generate DNA patterns for a gene (simplified)"""
        patterns = {
            'BRCA1': [('BRCA', 'BRCA1_detected'), ('ATG', 'BRCA1_start_codon')],
            'BRCA2': [('BRCA', 'BRCA2_detected'), ('ATG', 'BRCA2_start_codon')],
            'TP53': [('TP53', 'TP53_detected'), ('AACG', 'TP53_core')],
            'APOE': [('APOE', 'APOE_detected'), ('CGC', 'APOE_core')],
            'CFTR': [('CFTR', 'CFTR_detected'), ('ATG', 'CFTR_start')],
            'HFE': [('HFE', 'HFE_detected'), ('TGT', 'HFE_cys282')],
            'FTO': [('FTO', 'FTO_detected'), ('ATG', 'FTO_start')],
            'TCF7L2': [('TCF7L2', 'TCF7L2_detected'), ('ATG', 'TCF7L2_start')],
            'MTHFR': [('MTHFR', 'MTHFR_detected'), ('ATG', 'MTHFR_start')],
            'LDLR': [('LDLR', 'LDLR_detected'), ('ATG', 'LDLR_start')]
        }
        return patterns.get(gene, [(gene, f"{gene}_detected")])
    
    def _get_variant_type(self, gene_info: Dict) -> str:
        """Get variant type from gene info"""
        if 'inheritance' in gene_info:
            if 'Dominant' in gene_info['inheritance']:
                return 'Dominant'
            elif 'Recessive' in gene_info['inheritance']:
                return 'Recessive'
        return 'Complex'
    
    def calculate_risk_score(self, variants: List[Variant], user_metadata: Optional[Dict] = None) -> float:
        """
        Calculate disease risk score based on detected variants.
        
        Args:
            variants: List of detected variants
            user_metadata: Optional user data (age, gender, etc.)
            
        Returns:
            Risk score 0-100
        """
        if not variants:
            return 10.0  # Baseline population risk
        
        risk_score = 10.0  # Start with baseline
        
        for variant in variants:
            gene_info = self.DISEASE_GENES.get(variant.gene, {})
            penetrance = gene_info.get('penetrance', 0.5)
            confidence = max(
                variant.confidence,
                variant.pathogenicity_score if variant.pathogenicity_score is not None else 0.0,
            )
            
            # Contribution increases with penetrance and confidence
            risk_contribution = penetrance * confidence * 40  # Scale to 0-40
            risk_score += risk_contribution
        
        # Age adjustment (higher risk with age for late-onset diseases)
        if user_metadata and 'age' in user_metadata:
            age = user_metadata['age']
            age_factor = 1.0 + (age - 40) * 0.01 if age > 40 else 1.0
            risk_score *= min(age_factor, 2.0)  # Cap at 2x
        
        return min(risk_score, 100.0)


class BiomarkerDetector:
    """Detects disease-associated biomarkers in sequences"""
    
    # Known biomarker database
    BIOMARKERS = {
        'HER2': {
            'type': 'Protein-coding',
            'patterns': ['ERBB2', 'HER2_amplification', 'GRB7'],
            'diseases': ['Breast Cancer'],
            'significance': 'Therapeutic Target',
            'clinical_use': 'Trastuzumab (Herceptin) eligibility'
        },
        'EGFR': {
            'type': 'Protein-coding',
            'patterns': ['EGFR', 'EGF_receptor'],
            'diseases': ['Lung Cancer', 'Glioblastoma'],
            'significance': 'Therapeutic Target',
            'clinical_use': 'EGFR inhibitor therapy'
        },
        'KRAS': {
            'type': 'Oncogene',
            'patterns': ['KRAS', 'G12C', 'G12V'],
            'diseases': ['Colorectal Cancer', 'Pancreatic Cancer', 'Lung Cancer'],
            'significance': 'Prognostic Marker',
            'clinical_use': 'Prognosis and treatment selection'
        },
        'BRAF': {
            'type': 'Oncogene',
            'patterns': ['BRAF', 'V600E'],
            'diseases': ['Melanoma', 'Colorectal Cancer'],
            'significance': 'Therapeutic Target',
            'clinical_use': 'BRAF inhibitor therapy'
        },
        'ER': {
            'type': 'Receptor',
            'patterns': ['ESR1', 'ERalpha'],
            'diseases': ['Breast Cancer'],
            'significance': 'Treatment Indicator',
            'clinical_use': 'Hormone therapy eligibility'
        },
        'PR': {
            'type': 'Receptor',
            'patterns': ['PGR', 'PRG'],
            'diseases': ['Breast Cancer'],
            'significance': 'Treatment Indicator',
            'clinical_use': 'Hormone therapy eligibility'
        },
        'PD-L1': {
            'type': 'Immune Checkpoint',
            'patterns': ['CD274', 'PD-L1'],
            'diseases': ['Lung Cancer', 'Melanoma', 'Colorectal Cancer'],
            'significance': 'Therapeutic Target',
            'clinical_use': 'Immunotherapy eligibility'
        },
        'MSI': {
            'type': 'Genomic Signature',
            'patterns': ['microsatellite_instability', 'MSI-H'],
            'diseases': ['Colorectal Cancer', 'Gastric Cancer'],
            'significance': 'Prognostic Marker',
            'clinical_use': 'Immunotherapy response prediction'
        },
        'TMPRSS2-ERG': {
            'type': 'Gene Fusion',
            'patterns': ['TMPRSS2_ERG_fusion', 'ERG_overexpression'],
            'diseases': ['Prostate Cancer'],
            'significance': 'Prognostic Marker',
            'clinical_use': 'Risk stratification'
        },
        'ABL1': {
            'type': 'Oncogene',
            'patterns': ['BCR_ABL', 'BCR_ABL1'],
            'diseases': ['Chronic Myeloid Leukemia'],
            'significance': 'Diagnostic Marker',
            'clinical_use': 'TKI therapy target'
        }
    }
    
    def detect_biomarkers(self, sequence: str) -> List[Biomarker]:
        """
        Scan sequence for disease-associated biomarkers.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            List of detected biomarkers
        """
        detected_biomarkers = []
        sequence_upper = sequence.upper()
        
        for biomarker_name, biomarker_info in self.BIOMARKERS.items():
            for pattern in biomarker_info['patterns']:
                pattern_upper = pattern.upper()
                
                # Look for pattern in sequence
                if pattern_upper in sequence_upper:
                    position = sequence_upper.find(pattern_upper)
                    match_strength = 0.85 + np.random.random() * 0.15  # 0.85-1.0
                    
                    biomarker = Biomarker(
                        name=biomarker_name,
                        biomarker_type=biomarker_info['type'],
                        location=f"Position {position}",
                        sequence_pattern=pattern_upper,
                        position=position,
                        length=len(pattern_upper),
                        match_strength=match_strength,
                        associated_diseases=biomarker_info['diseases'],
                        clinical_significance=biomarker_info['significance']
                    )
                    detected_biomarkers.append(biomarker)
                    break  # Count biomarker once per sequence
        
        return detected_biomarkers


class DiseaseAssociationMapper:
    """Maps detected variants and biomarkers to disease associations"""
    
    # Disease-variant/biomarker associations
    DISEASE_ASSOCIATIONS = {
        'Breast Cancer': {
            'variants': ['BRCA1', 'BRCA2', 'TP53'],
            'biomarkers': ['HER2', 'ER', 'PR'],
            'inheritance': 'Autosomal Dominant (hereditary)',
            'prevalence': 0.121,  # ~12% lifetime risk
            'baseline_risk': 12.0
        },
        'Ovarian Cancer': {
            'variants': ['BRCA1', 'BRCA2'],
            'biomarkers': ['HER2'],
            'inheritance': 'Autosomal Dominant',
            'prevalence': 0.014,
            'baseline_risk': 1.4
        },
        'Colorectal Cancer': {
            'variants': ['TP53', 'KRAS'],
            'biomarkers': ['KRAS', 'MSI'],
            'inheritance': 'Complex',
            'prevalence': 0.046,
            'baseline_risk': 4.6
        },
        'Lung Cancer': {
            'variants': ['TP53'],
            'biomarkers': ['EGFR', 'KRAS', 'PD-L1'],
            'inheritance': 'Complex (mostly sporadic)',
            'prevalence': 0.065,
            'baseline_risk': 6.5
        },
        "Alzheimer's Disease": {
            'variants': ['APOE'],
            'biomarkers': [],
            'inheritance': 'Complex (late-onset)',
            'prevalence': 0.065,
            'baseline_risk': 6.5
        },
        'Type 2 Diabetes': {
            'variants': ['FTO', 'TCF7L2', 'MTHFR'],
            'biomarkers': [],
            'inheritance': 'Complex (multifactorial)',
            'prevalence': 0.097,
            'baseline_risk': 9.7
        },
        'Cystic Fibrosis': {
            'variants': ['CFTR'],
            'biomarkers': [],
            'inheritance': 'Autosomal Recessive',
            'prevalence': 0.0003,
            'baseline_risk': 0.03
        },
        'Hemochromatosis': {
            'variants': ['HFE'],
            'biomarkers': [],
            'inheritance': 'Autosomal Recessive',
            'prevalence': 0.001,
            'baseline_risk': 0.1
        },
        'Prostate Cancer': {
            'variants': ['BRCA1', 'BRCA2', 'TP53'],
            'biomarkers': ['TMPRSS2-ERG'],
            'inheritance': 'Complex',
            'prevalence': 0.121,
            'baseline_risk': 12.1
        },
        'Melanoma': {
            'variants': ['TP53'],
            'biomarkers': ['BRAF', 'PD-L1'],
            'inheritance': 'Complex',
            'prevalence': 0.024,
            'baseline_risk': 2.4
        },
        'Chronic Myeloid Leukemia': {
            'variants': [],
            'biomarkers': ['ABL1'],
            'inheritance': 'Somatic (acquired)',
            'prevalence': 0.0002,
            'baseline_risk': 0.02
        }
    }
    
    def map_disease_associations(self, variants: List[Variant], biomarkers: List[Biomarker],
                                 user_metadata: Optional[Dict] = None) -> List[DiseaseAssociation]:
        """
        Map detected variants and biomarkers to disease associations.
        
        Args:
            variants: List of detected variants
            biomarkers: List of detected biomarkers
            user_metadata: Optional user data
            
        Returns:
            List of disease associations sorted by risk
        """
        associations = []
        
        variant_genes = {v.gene for v in variants}
        biomarker_names = {b.name for b in biomarkers}
        
        for disease, disease_info in self.DISEASE_ASSOCIATIONS.items():
            # Check for variant matches
            matching_variants = [v for v in variants if v.gene in disease_info['variants']]
            matching_biomarkers = [b for b in biomarkers if b.name in disease_info['biomarkers']]
            
            if not matching_variants and not matching_biomarkers:
                continue  # Skip diseases with no matches
            
            # Calculate risk score
            base_risk = disease_info['baseline_risk']
            
            # Variant contribution
            variant_risk = 0
            for variant in matching_variants:
                variant_risk += variant.confidence * 35  # Up to 35% per variant
            
            # Biomarker contribution
            biomarker_risk = 0
            for biomarker in matching_biomarkers:
                biomarker_risk += biomarker.match_strength * 25  # Up to 25% per biomarker
            
            # Combine risks
            total_risk = min(base_risk + variant_risk + biomarker_risk, 100.0)
            
            # Determine confidence
            num_matches = len(matching_variants) + len(matching_biomarkers)
            if num_matches >= 3:
                confidence = ConfidenceLevel.VERY_HIGH
            elif num_matches == 2:
                confidence = ConfidenceLevel.HIGH
            elif num_matches == 1 and matching_variants:
                confidence = ConfidenceLevel.MODERATE
            else:
                confidence = ConfidenceLevel.LOW
            
            association = DiseaseAssociation(
                disease=disease,
                risk_score=total_risk,
                confidence=confidence,
                detected_variants=matching_variants,
                detected_biomarkers=matching_biomarkers,
                inheritance_pattern=disease_info['inheritance'],
                prevalence=disease_info['prevalence']
            )
            
            associations.append(association)
        
        # Sort by risk score (descending)
        associations.sort(key=lambda x: x.risk_score, reverse=True)
        
        return associations


class PersonalizedRecommendationEngine:
    """Generates personalized recommendations based on genomic profile"""
    
    # Therapeutic recommendations database
    THERAPEUTIC_DATABASE = {
        'Breast Cancer': {
            'first_line': [
                {'drug': 'Tamoxifen', 'indication': 'ER+ tumors', 'biomarker': 'ER+', 'notes': 'Monitor for side effects'},
                {'drug': 'Aromatase Inhibitors (AI)', 'indication': 'Postmenopausal ER+ tumors', 'biomarker': 'ER+', 'notes': 'Bone health monitoring'},
            ],
            'targeted': [
                {'drug': 'Trastuzumab (Herceptin)', 'indication': 'HER2+ tumors', 'biomarker': 'HER2+', 'notes': 'Requires cardiac monitoring'},
                {'drug': 'Pertuzumab', 'indication': 'HER2+ advanced disease', 'biomarker': 'HER2+', 'notes': 'Used with Trastuzumab'},
                {'drug': 'PARP Inhibitors (Olaparib)', 'indication': 'BRCA1/2 mutations', 'biomarker': 'BRCA+', 'notes': 'Maintenance therapy'},
            ],
            'lifestyle': ['Regular exercise', 'Mediterranean diet', 'Stress management', 'Weight management'],
            'monitoring': ['Regular mammography', 'Clinical breast exams', 'Tumor markers']
        },
        "Alzheimer's Disease": {
            'first_line': [
                {'drug': 'Donepezil', 'indication': 'Mild to moderate AD', 'biomarker': 'APOE-ε4', 'notes': 'Cholinesterase inhibitor'},
                {'drug': 'Memantine', 'indication': 'Moderate to severe AD', 'biomarker': 'General', 'notes': 'NMDA antagonist'},
            ],
            'targeted': [
                {'drug': 'Lecanemab', 'indication': 'Early cognitive decline', 'biomarker': 'Amyloid-β+', 'notes': 'Anti-amyloid monoclonal'},
            ],
            'lifestyle': ['Cognitive training', 'Mediterranean diet', 'Physical activity', 'Social engagement'],
            'monitoring': ['Cognitive testing', 'MRI surveillance', 'Caregiver support']
        },
        'Type 2 Diabetes': {
            'first_line': [
                {'drug': 'Metformin', 'indication': 'First-line agent', 'biomarker': 'General', 'notes': 'Gastrointestinal side effects'},
                {'drug': 'Lifestyle modification', 'indication': 'Diet and exercise', 'biomarker': 'General', 'notes': 'Most important intervention'},
            ],
            'targeted': [
                {'drug': 'GLP-1 Agonists', 'indication': 'Additional glucose control needed', 'biomarker': 'FTO+', 'notes': 'Weight loss benefit'},
                {'drug': 'SGLT2 Inhibitors', 'indication': 'Cardiovascular/renal protection', 'biomarker': 'General', 'notes': 'Additional benefits beyond glucose'},
            ],
            'lifestyle': ['Low glycemic diet', 'Regular exercise (150 min/week)', 'Weight loss', 'Stress management'],
            'monitoring': ['HbA1c testing', 'Lipid panel', 'Kidney function', 'Blood pressure']
        },
        'Colorectal Cancer': {
            'first_line': [
                {'drug': '5-Fluorouracil (5-FU)', 'indication': 'Standard chemotherapy', 'biomarker': 'General', 'notes': 'Often combined with Leucovorin'},
            ],
            'targeted': [
                {'drug': 'Cetuximab', 'indication': 'KRAS wild-type tumors', 'biomarker': 'KRAS-WT', 'notes': 'EGFR inhibitor'},
                {'drug': 'Pembrolizumab', 'indication': 'MSI-H tumors', 'biomarker': 'MSI-H', 'notes': 'Checkpoint inhibitor'},
            ],
            'lifestyle': ['High-fiber diet', 'Regular exercise', 'Limited alcohol', 'No smoking'],
            'monitoring': ['CEA tumor marker', 'Colonoscopy surveillance', 'Imaging studies']
        },
        'Hemochromatosis': {
            'first_line': [
                {'drug': 'Phlebotomy', 'indication': 'Iron removal', 'biomarker': 'HFE+', 'notes': 'Induction phase: weekly'},
                {'drug': 'Deferasirox', 'indication': 'Iron chelation if phlebotomy not tolerated', 'biomarker': 'HFE+', 'notes': 'Oral agent'},
            ],
            'targeted': [
                {'drug': 'Dietary iron restriction', 'indication': 'Maintenance therapy', 'biomarker': 'HFE+', 'notes': 'Avoid iron supplements'},
            ],
            'lifestyle': ['Low iron diet', 'Avoid alcohol', 'Avoid raw shellfish', 'Regular monitoring'],
            'monitoring': ['Serum ferritin', 'Transferrin saturation', 'Liver function', 'Cardiac assessment']
        }
    }
    
    # Pharmacogenomic guidance
    PHARMACOGENOMIC_GUIDANCE = {
        'CYP2D6': {
            'enzyme': 'Cytochrome P450 2D6',
            'substrates': ['Codeine', 'Tramadol', 'Tamoxifen', 'Fluoxetine', 'Risperidone'],
            'phenotypes': {
                'Ultra-rapid metabolizer': {'action': 'May require higher doses or alternative drugs', 'risk': 'Therapeutic failure'},
                'Rapid metabolizer': {'action': 'Standard dosing usually appropriate', 'risk': 'Slight therapeutic benefit reduction'},
                'Normal metabolizer': {'action': 'Standard dosing', 'risk': 'No special concerns'},
                'Intermediate metabolizer': {'action': 'Monitor closely; may need dose adjustment', 'risk': 'Reduced efficacy or increased side effects'},
                'Poor metabolizer': {'action': 'Use alternative drug or significantly reduce dose', 'risk': 'Severe side effects'},
            }
        },
        'CYP2C19': {
            'enzyme': 'Cytochrome P450 2C19',
            'substrates': ['Clopidogrel', 'Omeprazole', 'Escitalopram', 'Pantoprazole', 'Voriconazole'],
            'phenotypes': {
                'Rapid metabolizer': {'action': 'Higher doses needed for therapeutic effect', 'risk': 'Reduced efficacy'},
                'Normal metabolizer': {'action': 'Standard dosing', 'risk': 'No special concerns'},
                'Intermediate metabolizer': {'action': 'May need dose adjustment', 'risk': 'Monitor for efficacy'},
                'Poor metabolizer': {'action': 'Use alternative or reduce dose significantly', 'risk': 'Increased side effects'},
            }
        },
        'TPMT': {
            'enzyme': 'Thiopurine S-methyltransferase',
            'substrates': ['Azathioprine', '6-Mercaptopurine', '6-Thioguanine'],
            'phenotypes': {
                'High activity': {'action': 'Standard dosing', 'risk': 'No special concerns'},
                'Intermediate activity': {'action': 'Reduce dose by 30-50%', 'risk': 'Bone marrow suppression risk'},
                'Low activity': {'action': 'Consider alternative; if used, significantly reduce dose', 'risk': 'Severe toxicity'},
            }
        },
        'VKORC1': {
            'enzyme': 'Vitamin K Epoxide Reductase',
            'substrates': ['Warfarin'],
            'phenotypes': {
                'High activity': {'action': 'Higher warfarin doses usually needed', 'risk': 'Subtherapeutic INR'},
                'Normal activity': {'action': 'Standard dosing', 'risk': 'No special concerns'},
                'Low activity': {'action': 'Lower warfarin doses required', 'risk': 'Bleeding risk'},
            }
        }
    }
    
    def generate_recommendations(self, associations: List[DiseaseAssociation],
                                user_metadata: Dict) -> Dict:
        """
        Generate personalized recommendations based on disease associations.
        
        Args:
            associations: List of disease associations
            user_metadata: User demographics and clinical data
            
        Returns:
            Dictionary with comprehensive recommendations
        """
        recommendations = {
            'high_priority': [],
            'moderate_priority': [],
            'lifestyle': [],
            'monitoring': [],
            'pharmacogenomics': [],
            'disclaimers': []
        }
        
        # Add standard disclaimer
        recommendations['disclaimers'].append(
            "These are predicted genetic risk indicators based on sequence analysis for research/educational purposes only. "
            "NOT a medical diagnosis. Consult healthcare providers for medical decisions."
        )
        
        # Process high-confidence disease associations
        for assoc in associations[:3]:  # Top 3 associations
            if assoc.confidence in [ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH]:
                disease = assoc.disease
                
                if disease in self.THERAPEUTIC_DATABASE:
                    therapy_info = self.THERAPEUTIC_DATABASE[disease]
                    
                    # Add first-line options
                    for drug_info in therapy_info['first_line']:
                        recommendations['high_priority'].append({
                            'disease': disease,
                            'category': 'First-Line',
                            'treatment': drug_info['drug'],
                            'indication': drug_info['indication'],
                            'confidence': assoc.confidence.value,
                            'notes': drug_info['notes']
                        })
                    
                    # Add targeted therapies if biomarkers present
                    for biomarker in assoc.detected_biomarkers:
                        for drug_info in therapy_info['targeted']:
                            if biomarker.name in drug_info.get('biomarker', ''):
                                recommendations['moderate_priority'].append({
                                    'disease': disease,
                                    'category': 'Targeted Therapy',
                                    'treatment': drug_info['drug'],
                                    'biomarker': biomarker.name,
                                    'indication': drug_info['indication'],
                                    'notes': drug_info['notes']
                                })
                    
                    # Add lifestyle recommendations
                    if 'lifestyle' in therapy_info:
                        recommendations['lifestyle'].extend(therapy_info['lifestyle'])
                    
                    # Add monitoring recommendations
                    if 'monitoring' in therapy_info:
                        recommendations['monitoring'].extend(therapy_info['monitoring'])
        
        # Add age-based recommendations
        age = user_metadata.get('age', 50)
        if age > 50:
            recommendations['monitoring'].extend(['Annual health screening', 'Age-appropriate cancer screening'])
        
        # Remove duplicates
        recommendations['lifestyle'] = list(set(recommendations['lifestyle']))
        recommendations['monitoring'] = list(set(recommendations['monitoring']))
        
        # Add pharmacogenomic guidance
        recommendations['pharmacogenomics'] = self._get_pharmacogenomic_guidance(user_metadata)
        
        return recommendations
    
    def _get_pharmacogenomic_guidance(self, user_metadata: Dict) -> List[Dict]:
        """Get pharmacogenomic guidance based on common drug metabolizer genes"""
        guidance = []
        
        # Simulate detected metabolizer phenotypes
        metabolizer_phenotypes = {
            'CYP2D6': 'Intermediate metabolizer',
            'CYP2C19': 'Normal metabolizer',
            'TPMT': 'Normal metabolizer'
        }
        
        for gene, phenotype in metabolizer_phenotypes.items():
            if gene in self.PHARMACOGENOMIC_GUIDANCE:
                gene_info = self.PHARMACOGENOMIC_GUIDANCE[gene]
                phenotype_info = gene_info['phenotypes'].get(phenotype, {})
                
                guidance.append({
                    'gene': gene,
                    'enzyme': gene_info['enzyme'],
                    'phenotype': phenotype,
                    'affected_drugs': gene_info['substrates'],
                    'action': phenotype_info.get('action', 'Monitor'),
                    'risk': phenotype_info.get('risk', 'None')
                })
        
        return guidance


class GenomeAnalysisEngine:
    """
    Comprehensive genome analysis engine combining all components.
    Provides sequence-driven mutation analysis, biomarker detection, and personalized recommendations.
    Supports caching for performance optimization.
    """
    
    def __init__(self, cache_manager=None, variant_therapy_engine=None):
        self.mutation_analyzer = MutationAnalyzer()
        self.biomarker_detector = BiomarkerDetector()
        self.disease_mapper = DiseaseAssociationMapper()
        self.recommendation_engine = PersonalizedRecommendationEngine()
        self.cache_manager = cache_manager  # Optional cache manager for persistent storage
        self.variant_therapy_engine = variant_therapy_engine
        if self.variant_therapy_engine is None and VariantTherapyEngine is not None:
            try:
                self.variant_therapy_engine = VariantTherapyEngine()
            except Exception:
                self.variant_therapy_engine = None
    
    def _generate_sequence_hash(self, sequence: str) -> str:
        """Generate a hash of the sequence for caching purposes"""
        return hashlib.md5(sequence.upper().encode()).hexdigest()
    
    def analyze_genome(self, sequence: str, user_metadata: Optional[Dict] = None, annotated_variants: Optional[List[Dict]] = None) -> Dict:
        """
        Comprehensive genome analysis pipeline with caching support.
        
        Args:
            sequence: DNA sequence string
            user_metadata: User data (age, gender, weight, etc.)
            
        Returns:
            Comprehensive analysis results
        """
        if not user_metadata:
            user_metadata = {'age': 50, 'gender': 'Unknown', 'weight': 70}
        
        # Check cache if available
        if self.cache_manager:
            sequence_hash = self._generate_sequence_hash(sequence)
            cache_key = f"genome_analysis_{sequence_hash}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
        
        # Step 1: Mutation analysis
        variants = self.mutation_analyzer.analyze_mutations(sequence)
        
        # Step 2: Biomarker detection
        biomarkers = self.biomarker_detector.detect_biomarkers(sequence)
        
        # Step 3: Disease association mapping
        disease_associations = self.disease_mapper.map_disease_associations(
            variants, biomarkers, user_metadata
        )

        variant_prioritization = None
        if annotated_variants:
            scored_variants = self._score_annotated_variants(annotated_variants)
            variant_prioritization = {
                'scored_variants': scored_variants,
                'total_variants': len(scored_variants),
                'high_confidence_variants': len([
                    row for row in scored_variants
                    if float(row.get('pathogenicity_score', 0.0) or 0.0) >= 0.75
                ]),
            }
        
        # Step 4: Generate recommendations
        recommendations = self.recommendation_engine.generate_recommendations(
            disease_associations, user_metadata
        )
        
        # Compile results
        results = {
            'sequence_analysis': {
                'length': len(sequence),
                'gc_content': self._calculate_gc_content(sequence),
                'valid_nucleotides': sum(1 for c in sequence if c.upper() in 'ATCG')
            },
            'mutation_analysis': {
                'detected_variants': [v.to_dict() for v in variants],
                'total_variants': len(variants),
                'high_risk_variants': len([v for v in variants if v.confidence > 0.85])
            },
            'biomarker_detection': {
                'detected_biomarkers': [b.to_dict() for b in biomarkers],
                'total_biomarkers': len(biomarkers),
                'therapeutic_targets': len([b for b in biomarkers if 'Therapeutic' in b.clinical_significance])
            },
            'disease_associations': {
                'associations': [assoc.to_dict() for assoc in disease_associations],
                'high_confidence': len([a for a in disease_associations if a.confidence == ConfidenceLevel.VERY_HIGH]),
                'moderate_confidence': len([a for a in disease_associations if a.confidence == ConfidenceLevel.HIGH])
            },
            'variant_prioritization': variant_prioritization,
            'recommendations': recommendations,
            'analysis_metadata': {
                'user_age': user_metadata.get('age'),
                'user_gender': user_metadata.get('gender'),
                'user_weight': user_metadata.get('weight'),
                'analysis_type': 'Research/Educational'
            }
        }
        
        # Cache results if cache manager available
        if self.cache_manager:
            try:
                self.cache_manager.set(cache_key, json.dumps(results))
            except Exception as e:
                # Caching failure shouldn't break analysis
                pass
        
        return results

    def _score_annotated_variants(self, annotated_variants: List[Dict]) -> List[Dict]:
        """Score annotated variant payloads when a prioritizer is available."""
        if not annotated_variants:
            return []

        if self.variant_therapy_engine is None:
            return [dict(variant) for variant in annotated_variants]

        try:
            return self.variant_therapy_engine.score_variant_pathogenicity(
                [dict(variant) for variant in annotated_variants],
                use_prioritization=True,
            )
        except Exception:
            return self.variant_therapy_engine.score_variant_pathogenicity(
                [dict(variant) for variant in annotated_variants],
                use_prioritization=False,
            )
    
    def _calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content percentage"""
        seq_upper = sequence.upper()
        gc_count = seq_upper.count('G') + seq_upper.count('C')
        total = len([c for c in seq_upper if c in 'ATGC'])
        return (gc_count / total * 100) if total > 0 else 0

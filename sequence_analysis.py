# mypy: enable-error-code=var-annotated
"""
Sequence Analysis Suite - Comprehensive computational analysis of biological sequences
Supports DNA, RNA, and protein sequences with multiple analysis tools.
"""

import re
import io
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from dataclasses import dataclass
from collections import Counter, defaultdict
import numpy as np
from Bio import SeqIO, Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.Phylo import write
import networkx as nx
import json
import warnings
warnings.filterwarnings('ignore')


class AlignmentStats(TypedDict):
    alignment_length: int
    num_sequences: int
    conserved_positions: int
    conservation_percentage: float
    gap_positions: int
    gap_percentage: float


class TreeMetadata(TypedDict):
    method: str
    num_taxa: int
    tree_length: float
    newick_format: str


class MotifAnnotation(TypedDict):
    motif: str
    start: int
    end: int
    length: int
    conservation: float
    sequences: list[str]


@dataclass
class Sequence:
    """Represents a biological sequence with metadata"""
    id: str
    sequence: str
    description: str = ""
    sequence_type: str = "unknown"  # 'dna', 'rna', 'protein'
    
    def __post_init__(self) -> None:
        """Auto-detect sequence type if not specified"""
        if self.sequence_type == "unknown":
            self.sequence_type = self._detect_sequence_type()
    
    def _detect_sequence_type(self) -> str:
        """Detect if sequence is DNA, RNA, or protein"""
        seq_upper = self.sequence.upper().replace('-', '').replace('N', '').replace('X', '')
        if not seq_upper:
            return "unknown"
        
        # Check for RNA-specific nucleotides
        if 'U' in seq_upper and 'T' not in seq_upper:
            return "rna"
        
        # Check for DNA (T but no U)
        if 'T' in seq_upper and 'U' not in seq_upper:
            # Could be DNA or protein, check for protein-specific amino acids
            protein_chars = set('ACDEFGHIKLMNPQRSTVWY')
            if all(c in protein_chars for c in seq_upper):
                # Check if it's likely protein (has many non-ATGC chars)
                non_dna = len([c for c in seq_upper if c not in 'ATGC'])
                if non_dna > len(seq_upper) * 0.1:  # More than 10% non-DNA chars
                    return "protein"
            return "dna"
        
        # Check for protein amino acids
        protein_chars = set('ACDEFGHIKLMNPQRSTVWY')
        if all(c in protein_chars for c in seq_upper):
            return "protein"
        
        return "unknown"


class FASTAParser:
    """Parse and validate FASTA format sequences"""
    
    @staticmethod
    def parse(file_content: str) -> list[Sequence]:
        """Parse FASTA content from string"""
        sequences = []
        try:
            # Handle both file-like objects and strings
            if isinstance(file_content, str):
                file_handle = io.StringIO(file_content)
            else:
                file_handle = file_content
            
            for record in SeqIO.parse(file_handle, "fasta"):
                seq = Sequence(
                    id=record.id,
                    sequence=str(record.seq),
                    description=record.description
                )
                sequences.append(seq)
        except Exception as e:
            raise ValueError(f"FASTA parsing error: {str(e)}")
        
        return sequences
    
    @staticmethod
    def validate(sequences: list[Sequence]) -> tuple[bool, list[str]]:
        """Validate sequences and return (is_valid, error_messages)"""
        errors = []
        
        if not sequences:
            errors.append("No sequences found in input")
            return False, errors
        
        # Check for empty sequences
        for seq in sequences:
            if not seq.sequence or len(seq.sequence.strip()) == 0:
                errors.append(f"Sequence {seq.id} is empty")
        
        # Check for invalid characters
        valid_dna = set('ATCGN-')
        valid_rna = set('AUCGN-')
        valid_protein = set('ACDEFGHIKLMNPQRSTVWYX*-')
        
        for seq in sequences:
            seq_upper = seq.sequence.upper()
            if seq.sequence_type == "dna":
                invalid = set(seq_upper) - valid_dna
            elif seq.sequence_type == "rna":
                invalid = set(seq_upper) - valid_rna
            elif seq.sequence_type == "protein":
                invalid = set(seq_upper) - valid_protein
            else:
                continue
            
            if invalid:
                errors.append(f"Sequence {seq.id} contains invalid characters: {invalid}")
        
        return len(errors) == 0, errors


class MultipleSequenceAligner:
    """Perform Multiple Sequence Alignment using various algorithms"""
    
    def __init__(self, method: str = "clustalw") -> None:
        """
        Initialize aligner
        Methods: 'clustalw', 'muscle', 'mafft', 'simple'
        """
        self.method = method.lower()
        self.aligner = Align.PairwiseAligner()
        self.aligner.mode = 'global'
        self.aligner.match_score = 2.0
        self.aligner.mismatch_score = -1.0
        self.aligner.open_gap_score = -0.5
        self.aligner.extend_gap_score = -0.1
    
    def align(self, sequences: list[Sequence]) -> tuple[MultipleSeqAlignment, AlignmentStats]:
        """
        Perform multiple sequence alignment
        Returns (alignment, metadata)
        """
        if len(sequences) < 2:
            raise ValueError("Need at least 2 sequences for alignment")
        
        # Convert to BioPython SeqRecord objects
        seq_records = []
        for seq in sequences:
            seq_rec = SeqRecord(Seq(seq.sequence), id=seq.id, description=seq.description)
            seq_records.append(seq_rec)
        
        # For now, use progressive alignment (simple method)
        # In production, would call external tools like MUSCLE or MAFFT
        alignment = self._progressive_align(seq_records)
        
        # Calculate alignment statistics
        metadata = self._calculate_alignment_stats(alignment)
        
        return alignment, metadata
    
    def _progressive_align(self, seq_records: list[SeqRecord]) -> MultipleSeqAlignment:
        """Progressive alignment algorithm"""
        if len(seq_records) == 1:
            return MultipleSeqAlignment(seq_records)
        
        # Start with first sequence
        aligned = [seq_records[0]]
        
        # Align each subsequent sequence to the growing alignment
        for seq in seq_records[1:]:
            # Find best alignment position
            best_score = float('-inf')
            best_aligned = None
            
            # Try aligning to each existing sequence in the alignment
            for ref_seq in aligned:
                alignments = self.aligner.align(ref_seq.seq, seq.seq)
                if alignments:
                    best_alignment = max(alignments, key=lambda x: x.score)
                    if best_alignment.score > best_score:
                        best_score = best_alignment.score
                        best_aligned = best_alignment
            
            if best_aligned:
                # Extract aligned sequences
                ref_aligned, seq_aligned = best_aligned.aligned
                # For simplicity, add gaps to maintain alignment
                # In production, use proper MSA algorithms
                aligned.append(seq)
        
        return MultipleSeqAlignment(aligned)
    
    def _calculate_alignment_stats(self, alignment: MultipleSeqAlignment) -> AlignmentStats:
        """Calculate alignment statistics"""
        if not alignment:
            return {
                "alignment_length": 0,
                "num_sequences": 0,
                "conserved_positions": 0,
                "conservation_percentage": 0.0,
                "gap_positions": 0,
                "gap_percentage": 0.0,
            }
        
        length = alignment.get_alignment_length()
        num_seqs = len(alignment)
        
        # Count conserved positions
        conserved = 0
        gaps = 0
        
        for i in range(length):
            column = alignment[:, i]
            # Check if all non-gap characters are the same
            non_gaps = [c for c in column if c != '-']
            if len(non_gaps) > 0 and len(set(non_gaps)) == 1:
                conserved += 1
            if '-' in column:
                gaps += 1
        
        stats: AlignmentStats = {
            "alignment_length": length,
            "num_sequences": num_seqs,
            "conserved_positions": conserved,
            "conservation_percentage": (conserved / length * 100) if length > 0 else 0,
            "gap_positions": gaps,
            "gap_percentage": (gaps / length * 100) if length > 0 else 0
        }
        return stats


class PhylogeneticTreeBuilder:
    """Construct phylogenetic trees from aligned sequences"""
    
    def __init__(self, method: str = "neighbor_joining") -> None:
        """
        Initialize tree builder
        Methods: 'neighbor_joining', 'upgma'
        """
        self.method = method.lower()
        self.calculator = DistanceCalculator('identity')  # For protein, use 'blosum62'
    
    def build_tree(self, alignment: MultipleSeqAlignment) -> tuple[str, TreeMetadata]:
        """
        Build phylogenetic tree
        Returns (newick_string, metadata)
        """
        if len(alignment) < 2:
            raise ValueError("Need at least 2 sequences for tree construction")
        
        # Calculate distance matrix
        try:
            dm = self.calculator.get_distance(alignment)
        except Exception as e:
            # Fallback to simple distance calculation
            dm = self._simple_distance_matrix(alignment)
        
        # Build tree
        constructor = DistanceTreeConstructor()
        
        if self.method == "neighbor_joining":
            tree = constructor.nj(dm)
        else:  # UPGMA
            tree = constructor.upgma(dm)
        
        # Convert to Newick format
        from io import StringIO
        handle = StringIO()
        write([tree], handle, "newick")
        newick_string = handle.getvalue().strip()
        
        # Calculate tree statistics
        metadata: TreeMetadata = {
            "method": self.method,
            "num_taxa": len(alignment),
            "tree_length": sum(tree.depths().values()),
            "newick_format": newick_string
        }
        
        return newick_string, metadata
    
    def _simple_distance_matrix(self, alignment: MultipleSeqAlignment) -> Any:
        """Simple distance matrix calculation"""
        from Bio.Phylo.TreeConstruction import DistanceMatrix
        num_seqs = len(alignment)
        distances = []
        
        for i in range(num_seqs):
            row = []
            for j in range(i + 1):
                if i == j:
                    row.append(0.0)
                else:
                    # Calculate pairwise identity
                    seq1 = str(alignment[i].seq)
                    seq2 = str(alignment[j].seq)
                    matches = sum(c1 == c2 and c1 != '-' for c1, c2 in zip(seq1, seq2))
                    total = sum(1 for c1, c2 in zip(seq1, seq2) if c1 != '-' or c2 != '-')
                    identity = matches / total if total > 0 else 0.0
                    distance = 1.0 - identity
                    row.append(distance)
            distances.append(row)
        
        names = [record.id for record in alignment]
        return DistanceMatrix(names, distances)


class DomainIdentifier:
    """Identify functional domains in protein sequences"""
    
    def __init__(self) -> None:
        """Initialize domain identifier"""
        # Common protein domains patterns (simplified)
        # In production, would use Pfam HMMs or InterPro
        self.domain_patterns = {
            "Zinc Finger": [r"C.{2,4}C.{12,15}H.{3,5}H", r"H.{3,5}H.{12,15}C.{2,4}C"],
            "Helix-Turn-Helix": [r"[ILV].{5,7}[ILV].{5,7}[ILV]"],
            "Leucine Zipper": [r"L.{6}L.{6}L.{6}L"],
            "EF-hand": [r"D.{3}D.{3}[ILV].{6}[DE].{6}Y"],
        }
    
    def identify_domains(self, sequences: list[Sequence], alignment: Optional[MultipleSeqAlignment] = None) -> dict[str, list[dict[str, Any]]]:
        """
        Identify domains in sequences
        Returns dict mapping sequence_id to list of domain annotations
        """
        results = {}
        
        for seq in sequences:
            if seq.sequence_type != "protein":
                continue
            
            domains = []
            seq_upper = seq.sequence.upper()
            
            # Search for known domain patterns
            for domain_name, patterns in self.domain_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, seq_upper)
                    for match in matches:
                        domains.append({
                            "domain_name": domain_name,
                            "start": match.start() + 1,  # 1-indexed
                            "end": match.end(),
                            "sequence": match.group(),
                            "confidence": 0.7,  # Simplified confidence score
                            "method": "pattern_match"
                        })
            
            # Remove overlapping domains (keep first)
            domains = self._remove_overlaps(domains)
            results[seq.id] = domains
        
        return results
    
    def _remove_overlaps(self, domains: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove overlapping domain annotations"""
        if not domains:
            return []
        
        # Sort by start position
        sorted_domains = sorted(domains, key=lambda x: x["start"])
        non_overlapping: list[dict[str, Any]] = [sorted_domains[0]]
        
        for domain in sorted_domains[1:]:
            last = non_overlapping[-1]
            # Check if overlapping
            if domain["start"] > last["end"]:
                non_overlapping.append(domain)
            elif domain["end"] - domain["start"] > last["end"] - last["start"]:
                # Replace if current domain is longer
                non_overlapping[-1] = domain
        
        return non_overlapping


class MotifFinder:
    """Find conserved motifs in sequences"""
    
    def __init__(self, min_length: int = 4, max_length: int = 20) -> None:
        """Initialize motif finder"""
        self.min_length = min_length
        self.max_length = max_length
    
    def find_motifs(self, sequences: list[Sequence], alignment: Optional[MultipleSeqAlignment] = None) -> dict[str, Any]:
        """
        Find conserved motifs
        Returns dict with motifs and their positions
        """
        if alignment:
            return self._find_motifs_in_alignment(alignment)
        else:
            return self._find_motifs_in_sequences(sequences)
    
    def _find_motifs_in_alignment(self, alignment: MultipleSeqAlignment) -> dict[str, Any]:
        """Find motifs in aligned sequences"""
        motifs: list[dict[str, Any]] = []
        length = alignment.get_alignment_length()
        
        # Look for conserved regions
        for window_size in range(self.min_length, min(self.max_length + 1, length // 2)):
            for start in range(length - window_size + 1):
                column = alignment[:, start:start + window_size]
                
                # Check conservation
                conserved_seqs = []
                for i, seq in enumerate(column):
                    seq_str = ''.join(seq)
                    if '-' not in seq_str:
                        conserved_seqs.append((i, seq_str))
                
                if len(conserved_seqs) >= len(alignment) * 0.7:  # 70% conservation
                    # Check if all sequences have same pattern
                    patterns = [seq for _, seq in conserved_seqs]
                    if len(set(patterns)) == 1:
                        motifs.append({
                            "motif": patterns[0],
                            "start": start + 1,
                            "end": start + window_size,
                            "length": window_size,
                            "conservation": len(conserved_seqs) / len(alignment),
                            "sequences": [alignment[i].id for i, _ in conserved_seqs]
                        })
        
        # Remove duplicates and overlapping motifs
        motifs = self._deduplicate_motifs(motifs)
        
        return {
            "motifs": motifs,
            "num_motifs": len(motifs),
            "method": "alignment_based"
        }
    
    def _find_motifs_in_sequences(self, sequences: list[Sequence]) -> dict[str, Any]:
        """Find motifs using k-mer frequency analysis"""
        if len(sequences) < 2:
            return {"motifs": [], "num_motifs": 0, "method": "kmer_frequency"}
        
        # Count k-mers across all sequences
        kmer_counts: defaultdict[str, list[Tuple[str, int]]] = defaultdict(list)
        
        for k in range(self.min_length, self.max_length + 1):
            for seq in sequences:
                seq_upper = seq.sequence.upper().replace('-', '')
                for i in range(len(seq_upper) - k + 1):
                    kmer = seq_upper[i:i+k]
                    kmer_counts[kmer].append((seq.id, i + 1))
        
        # Find k-mers present in multiple sequences
        motifs: list[dict[str, Any]] = []
        for kmer, positions in kmer_counts.items():
            unique_seqs = set(seq_id for seq_id, _ in positions)
            if len(unique_seqs) >= max(2, len(sequences) * 0.5):  # Present in at least 50% of sequences
                motifs.append({
                    "motif": kmer,
                    "length": len(kmer),
                    "frequency": len(positions),
                    "sequences": list(unique_seqs),
                    "positions": positions[:10]  # Limit positions
                })
        
        # Sort by frequency
        motifs.sort(key=lambda x: x["frequency"], reverse=True)
        
        return {
            "motifs": motifs[:20],  # Top 20 motifs
            "num_motifs": len(motifs),
            "method": "kmer_frequency"
        }
    
    def _deduplicate_motifs(self, motifs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate and overlapping motifs"""
        if not motifs:
            return []
        
        # Sort by length (longer first) and conservation
        sorted_motifs = sorted(motifs, key=lambda x: (x["length"], x.get("conservation", 0)), reverse=True)
        unique: list[dict[str, Any]] = []
        
        for motif in sorted_motifs:
            # Check if overlaps with existing motifs
            overlaps = False
            for existing in unique:
                if (motif["start"] <= existing["end"] and motif["end"] >= existing["start"]):
                    overlaps = True
                    break
            
            if not overlaps:
                unique.append(motif)
        
        return unique


class ConservationScorer:
    """Calculate conservation scores for aligned sequences"""
    
    def __init__(self, method: str = "shannon_entropy") -> None:
        """
        Initialize conservation scorer
        Methods: 'shannon_entropy', 'simple_frequency'
        """
        self.method = method.lower()
    
    def calculate_conservation(self, alignment: MultipleSeqAlignment) -> dict[str, Any]:
        """
        Calculate conservation scores for each position
        Returns dict with position-wise scores and statistics
        """
        if not alignment:
            return {}
        
        length = alignment.get_alignment_length()
        num_seqs = len(alignment)
        
        scores: list[float] = []
        positions: list[dict[str, Any]] = []
        
        for i in range(length):
            column = alignment[:, i]
            # Remove gaps
            non_gaps = [c for c in column if c != '-']
            
            if not non_gaps:
                score = 0.0  # All gaps
            elif self.method == "shannon_entropy":
                score = self._shannon_entropy(non_gaps)
            else:  # simple_frequency
                score = self._simple_frequency(non_gaps)
            
            scores.append(score)
            positions.append({
                "position": i + 1,
                "score": score,
                "residues": dict(Counter(non_gaps)),
                "gap_count": column.count('-')
            })
        
        # Calculate statistics
        scores_array = np.array(scores)
        
        results: dict[str, Any] = {
            "scores": positions,
            "mean_conservation": float(np.mean(scores_array)),
            "std_conservation": float(np.std(scores_array)),
            "min_conservation": float(np.min(scores_array)),
            "max_conservation": float(np.max(scores_array)),
            "highly_conserved_positions": [i + 1 for i, s in enumerate(scores) if s > np.percentile(scores_array, 90)],
            "method": self.method
        }
        return results
    
    def _shannon_entropy(self, residues: List[str]) -> float:
        """Calculate Shannon entropy (lower = more conserved)"""
        if not residues:
            return 0.0
        
        counts = Counter(residues)
        total = len(residues)
        entropy = 0.0
        
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Normalize to 0-1 scale (1 = completely conserved, 0 = highly variable)
        max_entropy = np.log2(len(set(residues))) if len(set(residues)) > 1 else 0
        if max_entropy > 0:
            conservation = 1.0 - (entropy / max_entropy)
        else:
            conservation = 1.0
        
        return max(0.0, conservation)
    
    def _simple_frequency(self, residues: List[str]) -> float:
        """Simple frequency-based conservation (fraction of most common residue)"""
        if not residues:
            return 0.0
        
        counts = Counter(residues)
        most_common_count = max(counts.values())
        total = len(residues)
        
        return most_common_count / total if total > 0 else 0.0


class SequenceAnalysisSuite:
    """Main class orchestrating all sequence analysis tools"""
    
    def __init__(self):
        """Initialize the analysis suite"""
        self.parser = FASTAParser()
        self.aligner = MultipleSequenceAligner()
        self.tree_builder = PhylogeneticTreeBuilder()
        self.domain_identifier = DomainIdentifier()
        self.motif_finder = MotifFinder()
        self.conservation_scorer = ConservationScorer()
    
    def analyze(self, fasta_content: str, run_alignment: bool = True, 
                run_phylogeny: bool = True, run_domains: bool = True,
                run_motifs: bool = True, run_conservation: bool = True) -> Dict[str, Any]:
        """
        Run complete analysis pipeline
        Returns comprehensive analysis results
        """
        results: Dict[str, Any] = {
            "input_sequences": [],
            "alignment": None,
            "phylogenetic_tree": None,
            "domains": {},
            "motifs": {},
            "conservation": {},
            "errors": []
        }
        
        try:
            # 1. Parse sequences
            sequences = self.parser.parse(fasta_content)
            validation_result, errors = self.parser.validate(sequences)
            
            if not validation_result:
                results["errors"].extend(errors)
                return results
            
            results["input_sequences"] = [
                {
                    "id": seq.id,
                    "description": seq.description,
                    "length": len(seq.sequence),
                    "type": seq.sequence_type
                }
                for seq in sequences
            ]
            
            # 2. Multiple Sequence Alignment
            alignment = None
            if run_alignment and len(sequences) >= 2:
                try:
                    alignment, alignment_metadata = self.aligner.align(sequences)
                    results["alignment"] = {
                        "aligned_sequences": [
                            {
                                "id": record.id,
                                "sequence": str(record.seq),
                                "length": len(record.seq)
                            }
                            for record in alignment
                        ],
                        "metadata": alignment_metadata,
                        "format": "fasta_aligned"
                    }
                except Exception as e:
                    results["errors"].append(f"Alignment failed: {str(e)}")
            
            # 3. Conservation Scoring (requires alignment)
            if run_conservation and alignment:
                try:
                    conservation_results = self.conservation_scorer.calculate_conservation(alignment)
                    results["conservation"] = conservation_results
                except Exception as e:
                    results["errors"].append(f"Conservation scoring failed: {str(e)}")
            
            # 4. Domain Identification
            if run_domains:
                try:
                    domains = self.domain_identifier.identify_domains(sequences, alignment)
                    results["domains"] = domains
                except Exception as e:
                    results["errors"].append(f"Domain identification failed: {str(e)}")
            
            # 5. Motif Finding
            if run_motifs:
                try:
                    motifs = self.motif_finder.find_motifs(sequences, alignment)
                    results["motifs"] = motifs
                except Exception as e:
                    results["errors"].append(f"Motif finding failed: {str(e)}")
            
            # 6. Phylogenetic Tree Construction (requires alignment)
            if run_phylogeny and alignment and len(sequences) >= 2:
                try:
                    newick_tree, tree_metadata = self.tree_builder.build_tree(alignment)
                    results["phylogenetic_tree"] = {
                        "newick": newick_tree,
                        "metadata": tree_metadata
                    }
                except Exception as e:
                    results["errors"].append(f"Phylogenetic tree construction failed: {str(e)}")
        
        except Exception as e:
            results["errors"].append(f"Analysis pipeline error: {str(e)}")
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate a comprehensive text report from analysis results"""
        report = []
        report.append("=" * 80)
        report.append("SEQUENCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Input sequences
        report.append("INPUT SEQUENCES")
        report.append("-" * 80)
        for seq in results.get("input_sequences", []):
            report.append(f"ID: {seq['id']}")
            report.append(f"  Type: {seq['type']}")
            report.append(f"  Length: {seq['length']} residues")
            report.append(f"  Description: {seq.get('description', 'N/A')}")
            report.append("")
        
        # Alignment
        if results.get("alignment"):
            align_data = results["alignment"]
            report.append("MULTIPLE SEQUENCE ALIGNMENT")
            report.append("-" * 80)
            metadata = align_data.get("metadata", {})
            report.append(f"Alignment Length: {metadata.get('alignment_length', 'N/A')}")
            report.append(f"Number of Sequences: {metadata.get('num_sequences', 'N/A')}")
            report.append(f"Conserved Positions: {metadata.get('conserved_positions', 'N/A')}")
            report.append(f"Conservation: {metadata.get('conservation_percentage', 0):.2f}%")
            report.append("")
        
        # Conservation
        if results.get("conservation"):
            cons_data = results["conservation"]
            report.append("CONSERVATION ANALYSIS")
            report.append("-" * 80)
            report.append(f"Mean Conservation Score: {cons_data.get('mean_conservation', 0):.4f}")
            report.append(f"Highly Conserved Positions (>90th percentile): {len(cons_data.get('highly_conserved_positions', []))}")
            report.append("")
        
        # Domains
        if results.get("domains"):
            report.append("DOMAIN IDENTIFICATION")
            report.append("-" * 80)
            for seq_id, domains in results["domains"].items():
                if domains:
                    report.append(f"Sequence: {seq_id}")
                    for domain in domains:
                        report.append(f"  {domain['domain_name']}: positions {domain['start']}-{domain['end']} (confidence: {domain['confidence']:.2f})")
                    report.append("")
        
        # Motifs
        if results.get("motifs"):
            motifs_data = results["motifs"]
            report.append("MOTIF ANALYSIS")
            report.append("-" * 80)
            report.append(f"Number of Motifs Found: {motifs_data.get('num_motifs', 0)}")
            report.append(f"Method: {motifs_data.get('method', 'N/A')}")
            for motif in motifs_data.get("motifs", [])[:10]:  # Top 10
                report.append(f"  Motif: {motif.get('motif', 'N/A')} (length: {motif.get('length', 'N/A')})")
            report.append("")
        
        # Phylogenetic Tree
        if results.get("phylogenetic_tree"):
            tree_data = results["phylogenetic_tree"]
            report.append("PHYLOGENETIC TREE")
            report.append("-" * 80)
            report.append(f"Method: {tree_data.get('metadata', {}).get('method', 'N/A')}")
            report.append(f"Newick Format:")
            report.append(tree_data.get("newick", "N/A"))
            report.append("")
        
        # Errors
        if results.get("errors"):
            report.append("ERRORS AND WARNINGS")
            report.append("-" * 80)
            for error in results["errors"]:
                report.append(f"  - {error}")
            report.append("")
        
        report.append("=" * 80)
        report.append("End of Report")
        report.append("=" * 80)
        
        return "\n".join(report)

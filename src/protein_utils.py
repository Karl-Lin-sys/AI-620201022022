"""
ProteinAI - Utility Functions
=============================
"""

import numpy as np
from typing import List, Dict, Tuple

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

# Amino acid properties
AA_PROPERTIES = {
    'A': {'name': 'Alanine', 'hydro': 1.8, 'mass': 89.1, 'polar': False},
    'C': {'name': 'Cysteine', 'hydro': 2.5, 'mass': 121.2, 'polar': False},
    'D': {'name': 'Aspartic acid', 'hydro': -3.5, 'mass': 133.1, 'polar': True},
    'E': {'name': 'Glutamic acid', 'hydro': -3.5, 'mass': 147.1, 'polar': True},
    'F': {'name': 'Phenylalanine', 'hydro': 2.8, 'mass': 165.2, 'polar': False},
    'G': {'name': 'Glycine', 'hydro': -0.4, 'mass': 75.1, 'polar': False},
    'H': {'name': 'Histidine', 'hydro': -3.2, 'mass': 155.2, 'polar': True},
    'I': {'name': 'Isoleucine', 'hydro': 4.5, 'mass': 131.2, 'polar': False},
    'K': {'name': 'Lysine', 'hydro': -3.9, 'mass': 146.2, 'polar': True},
    'L': {'name': 'Leucine', 'hydro': 3.8, 'mass': 131.2, 'polar': False},
    'M': {'name': 'Methionine', 'hydro': 1.9, 'mass': 149.2, 'polar': False},
    'N': {'name': 'Asparagine', 'hydro': -3.5, 'mass': 132.1, 'polar': True},
    'P': {'name': 'Proline', 'hydro': -1.6, 'mass': 115.1, 'polar': False},
    'Q': {'name': 'Glutamine', 'hydro': -3.5, 'mass': 146.2, 'polar': True},
    'R': {'name': 'Arginine', 'hydro': -4.5, 'mass': 174.2, 'polar': True},
    'S': {'name': 'Serine', 'hydro': -0.8, 'mass': 105.1, 'polar': True},
    'T': {'name': 'Threonine', 'hydro': -0.7, 'mass': 119.1, 'polar': True},
    'V': {'name': 'Valine', 'hydro': 4.2, 'mass': 117.1, 'polar': False},
    'W': {'name': 'Tryptophan', 'hydro': -0.9, 'mass': 204.2, 'polar': False},
    'Y': {'name': 'Tyrosine', 'hydro': -1.3, 'mass': 181.2, 'polar': True},
}


def analyze_sequence(sequence: str) -> Dict:
    """Analyze properties of a protein sequence."""
    seq = sequence.upper()
    n = len(seq)
    
    if n == 0:
        return {'error': 'Empty sequence'}
    
    # Composition
    composition = {aa: seq.count(aa) / n * 100 for aa in AMINO_ACIDS}
    
    # Properties
    hydro = sum(AA_PROPERTIES.get(aa, {}).get('hydro', 0) for aa in seq) / n
    mass = sum(AA_PROPERTIES.get(aa, {}).get('mass', 110) for aa in seq)
    polar_ratio = sum(1 for aa in seq if AA_PROPERTIES.get(aa, {}).get('polar', False)) / n
    
    return {
        'length': n,
        'molecular_weight': mass,
        'hydropathy': hydro,
        'polarity': polar_ratio,
        'composition': composition
    }


def sequence_similarity(seq1: str, seq2: str) -> float:
    """Calculate sequence identity between two sequences."""
    if len(seq1) != len(seq2):
        return 0.0
    matches = sum(1 for a, b in zip(seq1.upper(), seq2.upper()) if a == b)
    return matches / len(seq1)


def validate_sequence(sequence: str) -> Tuple[bool, str]:
    """Validate protein sequence."""
    invalid = [c for c in sequence.upper() if c not in AMINO_ACIDS]
    if invalid:
        return False, f"Invalid amino acids: {set(invalid)}"
    return True, "Valid sequence"


def format_fasta(name: str, sequence: str, width: int = 60) -> str:
    """Format sequence as FASTA."""
    lines = [f">{name}"]
    for i in range(0, len(sequence), width):
        lines.append(sequence[i:i+width])
    return '\n'.join(lines)

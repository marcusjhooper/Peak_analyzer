import re
import logging
from pyfaidx import Fasta

logger = logging.getLogger(__name__)

def parse_coordinates(coords):
    """Parse genomic coordinates in format chr:start-end"""
    match = re.match(r'([^:]+):(\d+)-(\d+)', coords)
    if not match:
        raise ValueError("Invalid coordinate format. Use chr:start-end (e.g., chr13:113379626-113380127)")
    return match.group(1), int(match.group(2)), int(match.group(3))

def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 
                  'a': 't', 'c': 'g', 'g': 'c', 't': 'a',
                  'N': 'N', 'n': 'n'}
    return ''.join(complement.get(base, base) for base in reversed(seq))

def calculate_gc_content(seq):
    """Calculate the GC content of a DNA sequence."""
    seq = seq.upper()
    gc_count = seq.count('G') + seq.count('C')
    total = len(seq) - seq.count('N')
    if total == 0:
        return 0
    return (gc_count / total) * 100

def highlight_motif(sequence, motif):
    """Highlight occurrences of a motif in a sequence with HTML."""
    if not motif:
        return sequence
    
    # Case insensitive search
    motif_upper = motif.upper()
    sequence_upper = sequence.upper()
    
    # Find all occurrences of the motif
    highlighted_seq = ""
    last_end = 0
    
    for match in re.finditer(motif_upper, sequence_upper):
        start, end = match.span()
        # Add the part before the match
        highlighted_seq += sequence[last_end:start]
        # Add the highlighted match
        highlighted_seq += f'<span style="background-color: yellow; font-weight: bold;">{sequence[start:end]}</span>'
        last_end = end
    
    # Add the remaining part after the last match
    highlighted_seq += sequence[last_end:]
    
    return highlighted_seq

def format_sequence_with_line_numbers(seq, width=50):
    """Format a sequence with line numbers and fixed width."""
    lines = []
    for i in range(0, len(seq), width):
        line_num = i + 1
        chunk = seq[i:i+width]
        lines.append(f"{line_num:8d} {chunk}")
    return '\n'.join(lines)

def get_sequence(genome_path, chrom, start, end):
    """Get a sequence from a genome file."""
    try:
        logger.info(f"Loading genome from {genome_path}")
        genome = Fasta(genome_path)
        
        # Check if chromosome exists
        if chrom not in genome:
            raise ValueError(f"Chromosome {chrom} not found in genome file")
        
        # Get chromosome length
        chrom_length = len(genome[chrom])
        logger.info(f"Chromosome {chrom} length: {chrom_length}")
        
        # Validate coordinates
        if start < 0 or end > chrom_length:
            raise ValueError(f"Coordinates {start}-{end} are out of range for chromosome {chrom} (length: {chrom_length})")
        
        # Get the sequence
        logger.info(f"Getting sequence for {chrom}:{start}-{end}")
        sequence = genome[chrom][start:end]
        
        # Convert to string and ensure it's uppercase
        seq_str = str(sequence.seq).upper()
        
        # Validate sequence
        if not seq_str:
            raise ValueError("Empty sequence returned")
            
        # Log sequence details
        logger.info(f"Retrieved sequence of length {len(seq_str)}")
        logger.debug(f"First 10 bases: {seq_str[:10]}")
        
        return seq_str
    except Exception as e:
        logger.error(f"Error getting sequence: {str(e)}")
        raise ValueError(f"Failed to get sequence: {str(e)}") 
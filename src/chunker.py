"""
Chunk Meditations text by book + passage for coherent extraction.

Strategy: Split by book + numbered sections to preserve context and reasoning chains.
"""

import json
import re
from typing import Optional


def chunk_meditations(text: str, min_chunk_size: int = 200) -> list[dict]:
    """
    Chunk Meditations by book and section.
    
    Assumes text structure: "BOOK X." followed by numbered sections.
    Each chunk = one or more complete passages (BOOK + section number).
    
    Args:
        text: Raw Meditations text
        min_chunk_size: Minimum chars per chunk (skip very short passages)
        
    Returns:
        List of {passage_id, text, book, section, metadata}
    """
    chunks = []
    chunk_id = 0
    
    # Split by "BOOK X" markers
    book_pattern = r'BOOK\s+([IVX]+)\.'
    books = re.split(book_pattern, text)
    
    # books = ['', 'I', 'content', 'II', 'content', ...]
    # Reconstruct: pairs of (book_num, book_content)
    
    i = 1
    while i < len(books):
        book_num = books[i]
        book_content = books[i + 1] if i + 1 < len(books) else ""
        
        # Split book content by numbered sections (1, 2, 3, ...)
        # Pattern: "1." or "1. " at line start
        section_pattern = r'(?:^|\n)(\d+)\.\s+'
        sections = re.split(section_pattern, book_content)
        
        # sections = ['', '1', 'content', '2', 'content', ...]
        j = 1
        while j < len(sections):
            section_num = sections[j]
            section_text = sections[j + 1] if j + 1 < len(sections) else ""
            
            # Clean up section text
            section_text = section_text.strip()
            
            # Skip very short sections
            if len(section_text) < min_chunk_size:
                j += 2
                continue
            
            # Create chunk
            chunk = {
                "passage_id": f"book{book_num}_{section_num}",
                "text": section_text,
                "metadata": {
                    "book": book_num,
                    "section": int(section_num),
                    "book_int": roman_to_int(book_num),
                }
            }
            
            chunks.append(chunk)
            chunk_id += 1
            j += 2
        
        i += 2
    
    print(f"Chunked Meditations into {len(chunks)} passages")
    return chunks


def roman_to_int(roman: str) -> int:
    """Convert Roman numeral to integer."""
    val = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    int_val = 0
    for i in range(len(roman)):
        if i + 1 < len(roman) and val[roman[i]] < val[roman[i + 1]]:
            int_val -= val[roman[i]]
        else:
            int_val += val[roman[i]]
    return int_val


def save_chunks(chunks: list[dict], output_file: str):
    """Save chunks to JSON."""
    with open(output_file, 'w') as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved {len(chunks)} chunks to {output_file}")


def load_chunks(input_file: str) -> list[dict]:
    """Load chunks from JSON."""
    with open(input_file) as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {input_file}")
    return chunks


if __name__ == "__main__":
    import sys
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/meditations_raw.txt"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "data/meditations_chunks.json"
    
    # Read raw text
    with open(input_file, 'r') as f:
        text = f.read()
    
    # Chunk
    chunks = chunk_meditations(text)
    
    # Save
    save_chunks(chunks, output_file)
    
    # Print sample
    print("\nSample chunks:")
    for chunk in chunks[:3]:
        preview = chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"]
        print(f"  {chunk['passage_id']}: {preview}")
import json
import re
from typing import Optional


def chunk_meditations(text: str, min_chunk_size: int = 200) -> list[dict]:
    chunks = []
    chunk_id = 0
    
    book_pattern = r'BOOK\s+([IVX]+)\.'
    books = re.split(book_pattern, text)
    
    i = 1
    while i < len(books):
        book_num = books[i]
        book_content = books[i + 1] if i + 1 < len(books) else ""
        
        section_pattern = r'(?:^|\n)(\d+)\.\s+'
        sections = re.split(section_pattern, book_content)
        
        j = 1
        while j < len(sections):
            section_num = sections[j]
            section_text = sections[j + 1] if j + 1 < len(sections) else ""
            
            section_text = section_text.strip()
            
            if len(section_text) < min_chunk_size:
                j += 2
                continue
            
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
    with open(output_file, 'w') as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved {len(chunks)} chunks to {output_file}")


def load_chunks(input_file: str) -> list[dict]:
    with open(input_file) as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {input_file}")
    return chunks


if __name__ == "__main__":
    import sys
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/meditations_raw.txt"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "data/meditations_chunks.json"
    
    with open(input_file, 'r') as f:
        text = f.read()
    
    chunks = chunk_meditations(text)
    
    save_chunks(chunks, output_file)
    
    print("\nSample chunks:")
    for chunk in chunks[:3]:
        preview = chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"]
        print(f"  {chunk['passage_id']}: {preview}")
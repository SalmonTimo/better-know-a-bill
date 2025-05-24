#!/usr/bin/env python3
import argparse
import re
import json
import os # For document_id
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

# —— Configuration ——————————————————————————————————————————————————————
DEFAULT_OVERLAP_SIZE = 150  # Characters
DEFAULT_FALLBACK_CHUNK_SIZE = 750 # Characters for splitting text without sub-headings

# —— Task 1.1: Enhance Page-Level Text and Metadata —————————————————————

def extract_pages_text(pdf_path):
    """
    Extracts text from a PDF and returns a list of dictionaries.
    Each dictionary contains the page number (1-indexed) and its text content.
    """
    pages_data = []
    for i, page_layout in enumerate(extract_pages(pdf_path)):
        text_chunks = []
        for elem in page_layout:
            if isinstance(elem, LTTextContainer):
                text_chunks.append(elem.get_text())
        pages_data.append({
            "page_num": i + 1,
            "text": "".join(text_chunks)
        })
    return pages_data

def get_page_char_offsets(pages_data):
    """
    Calculates the starting character offset for each page in the full text.
    Assumes pages are joined by a single newline character.
    """
    page_char_offsets = []
    current_offset = 0
    for page in pages_data:
        page_char_offsets.append({"page_num": page["page_num"], "start_offset": current_offset})
        current_offset += len(page["text"]) + 1 # +1 for the newline joiner
    return page_char_offsets

def get_page_from_offset(char_offset, page_char_offsets_map):
    """
    Determines the page number for a given character offset.
    """
    for i, page_info in enumerate(page_char_offsets_map):
        if i + 1 < len(page_char_offsets_map):
            if page_info["start_offset"] <= char_offset < page_char_offsets_map[i+1]["start_offset"]:
                return page_info["page_num"]
        else: # Last page
            if page_info["start_offset"] <= char_offset:
                return page_info["page_num"]
    return page_char_offsets_map[-1]["page_num"] if page_char_offsets_map else 1 # Fallback

# —— Task 2.1: Define Sub-Section Heading Patterns ——————————————————————

# Matches lines like "DIVISION E—LEGISLATIVE BRANCH…", "TITLE I—MILITARY…", or "SEC. 3. REFERENCES."
# Level 1 and 2 headings
SECTION_HEADING_L1_L2 = re.compile(
    r"^(DIVISION\s+[A-Z0-9]+(?:[—-][^\n]+)?|TITLE\s+[IVXLCDM]+(?:[—-][^\n]+)?|SEC\.\s*\d+[A-Za-z]?\.(?:\s+[^\n]+)?)$",
    re.MULTILINE | re.IGNORECASE
)

# Sub-section patterns for levels 3+
SUB_HEADING_PATTERNS = {
    # r"^\s*\([a-z]\)\s+" for (a), (b)
    3: re.compile(r"^\s*(\([a-z]\))(?=\s|\n)"),
    # r"^\s*\(\d+\)\s+" for (1), (2)
    4: re.compile(r"^\s*(\(\d+\))(?=\s|\n)"),
    # r"^\s*\([A-Z]\)\s+" for (A), (B)
    5: re.compile(r"^\s*(\([A-Z]\))(?=\s|\n)"),
    # r"^\s*\(i+\)\s+" for (i), (ii) - extended to cover iv, ix etc.
    6: re.compile(r"^\s*(\([ivxlcdm]+\))(?=\s|\n)", re.IGNORECASE),
}
# Combined pattern for finding any sub-heading for splitting purposes
ALL_SUB_HEADING_REGEX = re.compile(
    "|".join(p.pattern for p in SUB_HEADING_PATTERNS.values()),
    re.MULTILINE
)

# Note: Center-aligned headers like "WEAPONS PROCUREMENT, NAVY" are not explicitly
# handled here as sub-sections of SEC. blocks unless they follow a clear pattern
# like the ones above. They might be part of the SEC. title itself or require
# more complex layout analysis not covered by regex on text.

# —— Task 2.2, 2.3, 3.1: Hierarchical Sub-chunking, Overlap, Metadata ———

def generate_chunk_id(doc_id, parent_id, title_like, index):
    """Generates a unique chunk ID."""
    title_slug = re.sub(r'\W+', '_', title_like.lower().strip()).strip('_')
    if parent_id:
        return f"{parent_id}_{title_slug}_{index}"
    return f"{doc_id}_{title_slug}_{index}"

def split_text_into_fallback_chunks(text, parent_chunk_global_start, doc_id, parent_id, current_level, ancestors, page_char_map, overlap_size, max_size):
    """Splits text into smaller, fixed-size chunks with overlap when no subheadings are found."""
    fallback_chunks = []
    text_len = len(text)
    start = 0
    chunk_idx = 0
    while start < text_len:
        end = min(start + max_size, text_len)
        
        # Try to end at a sentence boundary
        if end < text_len:
            sentence_end = text.rfind('.', start, end)
            if sentence_end != -1 and sentence_end > start + max_size // 2 : # Ensure not too small
                 end = sentence_end + 1

        actual_start_char_in_doc = parent_chunk_global_start + start
        actual_end_char_in_doc = parent_chunk_global_start + end

        # Add overlap from previous chunk's content
        # The overlap is taken from the *original* text stream, not from previous fallback chunk's text
        overlap_start_offset_in_doc = max(parent_chunk_global_start, actual_start_char_in_doc - overlap_size)
        
        # Ensure overlap doesn't cross parent boundaries significantly if this is the first fallback chunk.
        # For subsequent fallback chunks, overlap comes from the *current* parent's text.
        if chunk_idx > 0: # Overlap from previous text *within this fallback sequence*
            # The "start" for text slicing is relative to the input `text` block
            effective_overlap_start = max(0, start - overlap_size)
            chunk_text_with_overlap = text[effective_overlap_start:end]
            # Adjust start_char to reflect the beginning of the overlap
            start_char_for_chunk = parent_chunk_global_start + effective_overlap_start
        else: # First fallback chunk, overlap can come from text *before* this `text` block.
            # This scenario is trickier if `text` is not the absolute beginning of parent.
            # For simplicity, we'll assume `text` starts where it should and overlap is handled by parent caller,
            # or we only add forward overlap. The current implementation adds "backward" overlap into current chunk.
            chunk_text_with_overlap = text[start:end] # Initial chunk text
            start_char_for_chunk = actual_start_char_in_doc
            # If we want to add prefix from parent (if `start > 0` for `text` relative to parent)
            # this needs to be handled carefully. The current request is about overlap between *consecutive fine-grained chunks*.

            # Let's refine `start_char_for_chunk` and `chunk_text_with_overlap` for the first chunk.
            # It should include `overlap_size` characters before its *original* `start` if available within the parent.
            # This implies `text` itself should be the full parent text.
            # This function is called with segments of text. We need `full_text` or smarter offset management.
            # For now, let's assume `parent_chunk_global_start` is the reference for the `text` block passed.
            
            # Corrected overlap logic for *this function's scope*
            current_chunk_original_text = text[start:end]
            if fallback_chunks: # If there's a previous fallback chunk, get overlap from it
                prev_chunk_text = fallback_chunks[-1]['text_without_overlap'] # Need to store this temporarily
                overlap_text = prev_chunk_text[-overlap_size:]
                chunk_text_with_overlap = overlap_text + current_chunk_original_text
                start_char_for_chunk = actual_start_char_in_doc - len(overlap_text)

            else: # No previous fallback chunk, no internal overlap to add from prior segment.
                  # Overlap from text *before* this segment would be handled by the caller if needed.
                chunk_text_with_overlap = current_chunk_original_text
                start_char_for_chunk = actual_start_char_in_doc


        chunk_id = generate_chunk_id(doc_id, parent_id, f"paragraph_{chunk_idx}", chunk_idx)

        fallback_chunks.append({
            "chunk_id": chunk_id,
            "title": f"Part {chunk_idx + 1}", # Generic title
            "text": chunk_text_with_overlap,
            "text_without_overlap": text[start:end], # Store for next iteration's overlap
            "start_char": start_char_for_chunk,
            "end_char": start_char_for_chunk + len(chunk_text_with_overlap),
            "start_page": get_page_from_offset(start_char_for_chunk, page_char_map),
            "end_page": get_page_from_offset(start_char_for_chunk + len(chunk_text_with_overlap) -1, page_char_map),
            "parent_id": parent_id,
            "ancestor_ids": list(ancestors),
            "level": current_level, # Fallback chunks are at the same level as the text they are part of
            "document_id": doc_id
        })
        start = end
        chunk_idx += 1
    
    # Clean up temporary field
    for chunk in fallback_chunks:
        if "text_without_overlap" in chunk:
            del chunk["text_without_overlap"]
            
    return fallback_chunks


def make_hierarchical_chunks_recursive(
    full_doc_text, current_text_segment, current_level, base_offset,
    parent_id, current_ancestors, doc_id, page_char_map, overlap_size, fallback_chunk_size
):
    """
    Recursively finds headings and sub-headings to create hierarchical chunks.
    Manages character offsets relative to the full document text.
    """
    chunks = []
    
    # Determine which regex set to use based on level
    if current_level <= 2: # DIVISION, TITLE, SEC.
        primary_heading_regex = SECTION_HEADING_L1_L2
        # For level 1 and 2, we don't search for sub-headings within them directly in this pass.
        # The initial call to `make_coarse_chunks` or its equivalent handles L1/L2 boundaries.
        # This function, when called for an L1/L2 chunk's content, will then look for L3.
        # So, if current_level is already 1 or 2, we should be looking for the next level down.
        if current_level == 1: # Expecting SEC. or L3 within a Division/Title's direct text
            # This logic is tricky. The current `SECTION_HEADING_L1_L2` finds SEC.
            # If a TITLE has direct text before a SEC, that needs handling.
            # Let's assume coarse chunks (L1/L2) are already made, and this func refines them.
            # So, if parent_id suggests we are *inside* an L1/L2, we look for L3.
             pass # Handled by initial sectioning; this function refines *within* sections.
        
        # If this function is called to process the *content* of a SEC. chunk (level 2),
        # then the relevant_heading_regex should be for level 3.
        relevant_heading_regex = SUB_HEADING_PATTERNS.get(current_level + 1)

    else: # Sub-headings (a), (1), etc.
        relevant_heading_regex = SUB_HEADING_PATTERNS.get(current_level + 1) # Look for next deeper level

    # If no more specific sub-heading patterns for deeper levels, use the combined one for splitting.
    # This part is more about identifying *any* known sub-heading to segment the text.
    search_regex = relevant_heading_regex if relevant_heading_regex else ALL_SUB_HEADING_REGEX
    if not relevant_heading_regex and current_level >= min(SUB_HEADING_PATTERNS.keys()): # If we are deep, use ALL
        search_regex = ALL_SUB_HEADING_REGEX
    elif current_level < min(SUB_HEADING_PATTERNS.keys()): # e.g. processing a SEC. body for (a)
         search_regex = SUB_HEADING_PATTERNS.get(min(SUB_HEADING_PATTERNS.keys()))


    matches = []
    if search_regex:
        matches = list(search_regex.finditer(current_text_segment))

    last_segment_end = 0
    segment_idx = 0

    for i, match in enumerate(matches):
        match_start_in_segment = match.start()
        match_end_in_segment = match.end()
        heading_text = match.group(1).strip() if match.groups() else match.group(0).strip() # group(1) for capturing groups in SUB_HEADING

        # Determine level of this found heading
        found_level = current_level + 1 # Default assumption
        for lvl, pattern in SUB_HEADING_PATTERNS.items():
            if pattern.match(current_text_segment[match_start_in_segment:]):
                found_level = lvl
                heading_text = pattern.match(current_text_segment[match_start_in_segment:]).group(1).strip()
                break
        
        # Text before this sub-heading (if any)
        if match_start_in_segment > last_segment_end:
            plain_text_segment = current_text_segment[last_segment_end:match_start_in_segment]
            # This plain text is part of the parent chunk (at current_level)
            # Split it into smaller, overlapping chunks
            chunks.extend(split_text_into_fallback_chunks(
                plain_text_segment,
                base_offset + last_segment_end,
                doc_id,
                parent_id,
                current_level + 1, # Content chunks are deeper
                current_ancestors,
                page_char_map,
                overlap_size,
                fallback_chunk_size
            ))

        # Create chunk for the sub-heading itself
        sub_chunk_text_start_in_segment = match_start_in_segment
        sub_chunk_text_end_in_segment = matches[i+1].start() if i+1 < len(matches) else len(current_text_segment)
        
        sub_chunk_content = current_text_segment[sub_chunk_text_start_in_segment : sub_chunk_text_end_in_segment]

        # Apply overlap for sub-chunk text
        # Overlap comes from text *before* this sub-chunk's original start
        actual_start_char = base_offset + sub_chunk_text_start_in_segment
        
        # Determine start with overlap
        # Overlap should not cross the boundary of the `current_text_segment`'s beginning.
        overlap_adjusted_start_in_segment = max(0, sub_chunk_text_start_in_segment - overlap_size)
        
        # Ensure we don't grab from previous sibling's *content* due to simple offset math.
        # The text for overlap should come from `current_text_segment` up to `sub_chunk_text_start_in_segment`.
        if chunks and sub_chunk_text_start_in_segment > 0: # If there were preceding fallback chunks or this isn't the first sub_chunk
            # Get overlap text from the `current_text_segment` before this heading.
            # The text *before* this heading, within the current parent's scope
            prefix_text_candidate_start = max(last_segment_end, sub_chunk_text_start_in_segment - overlap_size)
            overlap_text = current_text_segment[prefix_text_candidate_start : sub_chunk_text_start_in_segment]
        else: # First chunk in this segment, or no prior text
            overlap_text = ""

        final_sub_chunk_text = overlap_text + sub_chunk_content
        final_sub_chunk_start_char = actual_start_char - len(overlap_text)
        final_sub_chunk_end_char = final_sub_chunk_start_char + len(final_sub_chunk_text)


        chunk_id = generate_chunk_id(doc_id, parent_id, heading_text, segment_idx)
        new_ancestors = [chunk_id] + list(current_ancestors) # Current chunk is an ancestor to its children

        # The text for this chunk is from its heading to the next, or end of parent.
        # This text *includes* its own heading.
        # The recursive call should process the content *after* the heading.
        
        # Content for recursion is text *after* the current sub_heading's line, up to the next one.
        # The `sub_chunk_content` contains the current heading. We need to find where its text body starts.
        heading_line_end_match = re.match(r"^[^\n]*\n?", sub_chunk_content)
        content_offset_within_sub_chunk = len(heading_line_end_match.group(0)) if heading_line_end_match else 0
        
        recursive_content = sub_chunk_content[content_offset_within_sub_chunk:]
        recursive_base_offset = base_offset + sub_chunk_text_start_in_segment + content_offset_within_sub_chunk
        
        # Add the current sub-section chunk
        chunks.append({
            "chunk_id": chunk_id,
            "title": heading_text,
            "text": final_sub_chunk_text, # Text includes heading and its content, plus overlap
            "start_char": final_sub_chunk_start_char,
            "end_char": final_sub_chunk_end_char,
            "start_page": get_page_from_offset(final_sub_chunk_start_char, page_char_map),
            "end_page": get_page_from_offset(final_sub_chunk_end_char -1, page_char_map),
            "parent_id": parent_id,
            "ancestor_ids": list(current_ancestors), # Ancestors up to *this* chunk's parent
            "level": found_level,
            "document_id": doc_id
        })
        segment_idx += 1
        
        # Recursively process the content of this sub-section
        if recursive_content.strip(): # Only recurse if there's actual content
            chunks.extend(make_hierarchical_chunks_recursive(
                full_doc_text, recursive_content, found_level, recursive_base_offset,
                chunk_id, new_ancestors, doc_id, page_char_map, overlap_size, fallback_chunk_size
            ))
        
        last_segment_end = sub_chunk_text_end_in_segment # End of the current sub-section's full text span

    # Remaining text after the last sub-heading (if any)
    if last_segment_end < len(current_text_segment):
        plain_text_segment = current_text_segment[last_segment_end:]
        # This plain text is part of the parent chunk (at current_level)
        chunks.extend(split_text_into_fallback_chunks(
            plain_text_segment,
            base_offset + last_segment_end,
            doc_id,
            parent_id, # Parent is the chunk this text was found in
            current_level + 1, # Content chunks are deeper than their structural parent
            current_ancestors,
            page_char_map,
            overlap_size,
            fallback_chunk_size
        ))
        
    return chunks


def find_top_level_boundaries(full_text):
    """ Finds DIVISION, TITLE, SEC. boundaries. """
    return [(m.start(), m.group(0).strip(), m) for m in SECTION_HEADING_L1_L2.finditer(full_text)]


def determine_level_from_title(title):
    if title.upper().startswith("DIVISION"): return 1
    if title.upper().startswith("TITLE"): return 1 # Can be L1, or L2 if under a Division
    if title.upper().startswith("SEC."): return 2
    return 0 # Unknown

def process_pdf_to_chunks(pdf_path, doc_id, overlap_size, fallback_chunk_size):
    """
    Main processing pipeline to extract and structure chunks from a PDF.
    """
    # 1. Extract page data (Task 1.1)
    pages_data = extract_pages_text(pdf_path)
    if not pages_data:
        print(f"Warning: No text extracted from {pdf_path}")
        return []

    full_text = "\n".join(p["text"] for p in pages_data)
    page_char_map = get_page_char_offsets(pages_data)

    # 2. Find initial top-level boundaries (Task related to 2.2's start)
    top_boundaries = find_top_level_boundaries(full_text)
    
    all_chunks = []
    doc_ancestors = [] # For top-level chunks, ancestors are empty

    for i, (start_char, title, match_obj) in enumerate(top_boundaries):
        end_char = top_boundaries[i+1][0] if i+1 < len(top_boundaries) else len(full_text)
        
        current_level = determine_level_from_title(title)
        # Adjust level if SEC. appears under a TITLE that was already found (needs context of previous chunk)
        # For simplicity here, we'll use the direct determination. A more complex parent tracking could refine this.

        chunk_id = generate_chunk_id(doc_id, None, title, i)
        
        # Text for this coarse chunk (includes its own heading)
        coarse_chunk_text_content = full_text[start_char:end_char]
        
        # Add overlap for the coarse chunk itself (from text *before* its start_char if not the first chunk)
        final_coarse_chunk_text = coarse_chunk_text_content
        final_coarse_chunk_start_char = start_char

        if i > 0 and start_char > 0 : # Not the first chunk in the document
            # Find text from `full_text` before this chunk for overlap
            overlap_actual_start = max(0, start_char - overlap_size)
            overlap_text_content = full_text[overlap_actual_start:start_char]
            final_coarse_chunk_text = overlap_text_content + coarse_chunk_text_content
            final_coarse_chunk_start_char = overlap_actual_start
        
        final_coarse_chunk_end_char = final_coarse_chunk_start_char + len(final_coarse_chunk_text)


        # Add the top-level chunk
        top_level_chunk_data = {
            "chunk_id": chunk_id,
            "title": title,
            "text": final_coarse_chunk_text,
            "start_char": final_coarse_chunk_start_char,
            "end_char": final_coarse_chunk_end_char,
            "start_page": get_page_from_offset(final_coarse_chunk_start_char, page_char_map),
            "end_page": get_page_from_offset(final_coarse_chunk_end_char -1, page_char_map),
            "parent_id": None,
            "ancestor_ids": [],
            "level": current_level,
            "document_id": doc_id
        }
        all_chunks.append(top_level_chunk_data)

        # Now, process the *content* of this coarse chunk for sub-headings
        # Content for recursion starts *after* the heading line of the coarse chunk
        heading_line_match = re.match(r"^[^\n]*\n?", coarse_chunk_text_content) # Match the first line (heading)
        content_offset_within_coarse_chunk = len(heading_line_match.group(0)) if heading_line_match else 0
        
        text_for_sub_chunking = coarse_chunk_text_content[content_offset_within_coarse_chunk:]
        base_offset_for_sub_chunking = start_char + content_offset_within_coarse_chunk

        if text_for_sub_chunking.strip():
            sub_chunks = make_hierarchical_chunks_recursive(
                full_text, # Pass full_text for context if needed by advanced overlap
                text_for_sub_chunking,
                current_level, # The level of the parent (this coarse_chunk)
                base_offset_for_sub_chunking, # Global start offset for the text_for_sub_chunking
                chunk_id, # Parent ID for sub_chunks
                [chunk_id], # Ancestors for sub_chunks
                doc_id,
                page_char_map,
                overlap_size,
                fallback_chunk_size
            )
            all_chunks.extend(sub_chunks)
            
    # Handle text before the first major heading, if any
    if not top_boundaries or top_boundaries[0][0] > 0:
        start_char_preamble = 0
        end_char_preamble = top_boundaries[0][0] if top_boundaries else len(full_text)
        preamble_text = full_text[start_char_preamble:end_char_preamble]
        if preamble_text.strip():
            preamble_ancestors = []
            preamble_chunks = split_text_into_fallback_chunks(
                preamble_text,
                start_char_preamble,
                doc_id,
                None, # No parent for preamble sections
                1, # Level 1 for preamble text
                preamble_ancestors,
                page_char_map,
                overlap_size,
                fallback_chunk_size
            )
            # Prepend preamble chunks if they exist
            all_chunks = preamble_chunks + all_chunks


    return all_chunks


# —— Main CLI ——————————————————————————————————————————————————————

def main():
    parser = argparse.ArgumentParser(
        description="Extract hierarchical, overlapping, and metadata-rich chunks from a PDF."
    )
    parser.add_argument("pdf_path", help="Path to the input PDF file")
    parser.add_argument(
        "--output", "-o",
        default="structured_chunks.json",
        help="Where to write the JSON array of structured chunks"
    )
    parser.add_argument(
        "--overlap", type=int, default=DEFAULT_OVERLAP_SIZE,
        help=f"Number of characters for overlap between chunks (default: {DEFAULT_OVERLAP_SIZE})"
    )
    parser.add_argument(
        "--fallback_size", type=int, default=DEFAULT_FALLBACK_CHUNK_SIZE,
        help=f"Target size for fallback chunks when no subheadings are found (default: {DEFAULT_FALLBACK_CHUNK_SIZE})"
    )
    args = parser.parse_args()

    doc_id = os.path.splitext(os.path.basename(args.pdf_path))[0]
    
    print(f"Processing {args.pdf_path} with doc_id: {doc_id}...")

    all_structured_chunks = process_pdf_to_chunks(
        args.pdf_path,
        doc_id,
        args.overlap,
        args.fallback_size
    )

    # Task 3.2: Refine Output Structure (flat JSON array)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_structured_chunks, f, ensure_ascii=False, indent=2)

    print(f"💾  Wrote {len(all_structured_chunks)} structured chunks to '{args.output}'.")

if __name__ == "__main__":
    main()
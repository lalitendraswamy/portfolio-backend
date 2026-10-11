import os
import re
import glob
# pyrefly: ignore [missing-import]
from pinecone import Pinecone
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configuration
KNOWLEDGE_BASE_DIR = "knowledge_base"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "portfolio-chatbot")
MAX_CHUNK_WORDS = 300  # Sections longer than this are split on paragraph boundaries
CHUNK_OVERLAP_WORDS = 50  # Overlap used only when a single paragraph exceeds MAX_CHUNK_WORDS
EXCLUDED_FILES = {"README.md"}  # Authoring instructions, not portfolio knowledge

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

# Initialize Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=api_key)

# Initialize Pinecone
pc_api_key = os.getenv("PINECONE_API_KEY")
if not pc_api_key:
    raise ValueError("PINECONE_API_KEY not found in environment variables")
pc = Pinecone(api_key=pc_api_key)

def get_embedding(text: str) -> list[float]:
    result = genai.embed_content(
        model="models/gemini-embedding-2",
        content=text,
        task_type="retrieval_document",
        output_dimensionality=1024
    )
    return result['embedding']

def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading_path, body) pairs, one per heading.

    heading_path is the breadcrumb of enclosing headings, e.g.
    "Frequently Asked Questions > HR and Availability > What are his salary expectations?".
    Headings with no body of their own (pure group headings) produce no section.
    """
    sections = []
    headings: list[str] = []
    body: list[str] = []
    in_code_block = False

    def flush():
        content = "\n".join(body).strip()
        if content:
            sections.append((" > ".join(headings), content))
        body.clear()

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code_block = not in_code_block
        match = None if in_code_block else HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            headings[level - 1:] = [match.group(2)]
        else:
            body.append(line)
    flush()

    return sections

def split_long_section(body: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    """Pack paragraphs into pieces of at most max_words; window-split any single oversized paragraph."""
    pieces = []
    current: list[str] = []
    current_len = 0

    for paragraph in re.split(r"\n\s*\n", body):
        word_count = len(paragraph.split())
        if current and current_len + word_count > max_words:
            pieces.append("\n\n".join(current))
            current, current_len = [], 0

        if word_count > max_words:
            words = paragraph.split()
            step = max_words - CHUNK_OVERLAP_WORDS
            for start in range(0, len(words) - CHUNK_OVERLAP_WORDS, step):
                pieces.append(" ".join(words[start:start + max_words]))
            continue

        current.append(paragraph)
        current_len += word_count

    if current:
        pieces.append("\n\n".join(current))

    return pieces

def split_into_chunks(text: str) -> list[tuple[str, str]]:
    """Heading-aware chunking: one chunk per markdown section, prefixed with its heading path
    so each chunk is self-describing for both the embedding and the LLM."""
    chunks = []
    for heading_path, body in split_into_sections(text):
        for piece in split_long_section(body):
            chunk_text = f"{heading_path}\n\n{piece}" if heading_path else piece
            chunks.append((heading_path, chunk_text))
    return chunks

def ingest_data():
    # Check if index exists, if not create it (gemini-embedding-2 is requested at 1024 dimensions)
    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        from pinecone import ServerlessSpec
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024, 
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    
    index = pc.Index(INDEX_NAME)
    
    # Read all markdown files
    md_files = [
        path for path in glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "**/*.md"), recursive=True)
        if os.path.basename(path) not in EXCLUDED_FILES
    ]
    if not md_files:
        print(f"No markdown files found in {KNOWLEDGE_BASE_DIR}/")
        return
        
    vectors = []
    chunk_id = 0
    upserted_ids = set()
    
    for file_path in md_files:
        print(f"Processing {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        filename = os.path.basename(file_path)
        chunks = split_into_chunks(content)
        
        for i, (section, chunk) in enumerate(chunks):
            embedding = get_embedding(chunk)

            # Prepare vector for Pinecone
            vector_id = f"{filename}_chunk_{i}"
            metadata = {
                "source": filename,
                "section": section,
                "chunk_index": i,
                "text": chunk
            }

            vectors.append({"id": vector_id, "values": embedding, "metadata": metadata})
            upserted_ids.add(vector_id)
            chunk_id += 1
            
            # Upsert in batches of 100
            if len(vectors) >= 100:
                print(f"Upserting batch of {len(vectors)} vectors...")
                index.upsert(vectors=vectors)
                vectors = []
                
    # Upsert remaining vectors
    if vectors:
        print(f"Upserting final batch of {len(vectors)} vectors...")
        index.upsert(vectors=vectors)

    # Remove vectors left over from earlier runs (e.g. a file shrank or the chunking changed)
    stale_ids = [vid for page in index.list() for vid in page if vid not in upserted_ids]
    if stale_ids:
        print(f"Deleting {len(stale_ids)} stale vectors...")
        for start in range(0, len(stale_ids), 1000):
            index.delete(ids=stale_ids[start:start + 1000])

    print(f"Successfully ingested {chunk_id} chunks into Pinecone index '{INDEX_NAME}'.")

if __name__ == "__main__":
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"Directory '{KNOWLEDGE_BASE_DIR}' not found. Please create it and add your markdown files.")
    else:
        ingest_data()

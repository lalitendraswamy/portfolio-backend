import os
import glob
# pyrefly: ignore [missing-import]
from pinecone import Pinecone
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configuration
KNOWLEDGE_BASE_DIR = "knowledge_base"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "portfolio-chatbot")
CHUNK_SIZE = 500  # Number of words per chunk (approximate)

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

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def ingest_data():
    # Check if index exists, if not create it (dimension for text-embedding-004 is 768)
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
    md_files = glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "**/*.md"), recursive=True)
    if not md_files:
        print(f"No markdown files found in {KNOWLEDGE_BASE_DIR}/")
        return
        
    vectors = []
    chunk_id = 0
    
    for file_path in md_files:
        print(f"Processing {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        filename = os.path.basename(file_path)
        chunks = split_into_chunks(content)
        
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            
            # Prepare vector for Pinecone
            vector_id = f"{filename}_chunk_{i}"
            metadata = {
                "source": filename,
                "chunk_index": i,
                "text": chunk
            }
            
            vectors.append({"id": vector_id, "values": embedding, "metadata": metadata})
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
        
    print(f"Successfully ingested {chunk_id} chunks into Pinecone index '{INDEX_NAME}'.")

if __name__ == "__main__":
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"Directory '{KNOWLEDGE_BASE_DIR}' not found. Please create it and add your markdown files.")
    else:
        ingest_data()

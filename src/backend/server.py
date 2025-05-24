import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from functools import lru_cache
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional



# --- Configuration ---
load_dotenv()
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_directory_not_found")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "chroma_collection_not_found") # Match with store_in_chroma.py
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "gemini_api_key_not_found")

# --- Initialize ChromaDB Client & Embedding Function ---
# This should ideally be managed with lifespan events for production
# For simplicity in this script, we initialize it globally.
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
    sentence_transformer_ef = embedding_functions.DefaultEmbeddingFunction()
    collection = chroma_client.get_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=sentence_transformer_ef
    )
    print(f"Successfully connected to ChromaDB collection: '{CHROMA_COLLECTION_NAME}' with {collection.count()} items.")
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    print(f"Ensure ChromaDB is running and the collection '{CHROMA_COLLECTION_NAME}' exists in '{CHROMA_PERSIST_DIRECTORY}'.")
    collection = None # Set to None if connection fails

# --- Initialize Gemini Client ---
gemini_model = None # Placeholder until key is set
if GEMINI_API_KEY != "gemini_api_key_not_found":
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        # Choose model here
        gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
        print("Gemini model initialized.")
    except ImportError:
        print("google.generativeai not installed. pip install google-generativeai")
        gemini_model = None
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
        gemini_model = None
else:
    print("Warning: GEMINI_API_KEY not set. LLM functionality will be placeholder.")


app = FastAPI()

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"], # Add frontend dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Pydantic Models ---
class Section(BaseModel):
    id: str
    title: str
    start_page: int = Field(..., alias="page") # Keep alias for compatibility if old data exists
    subsections: List['Section'] = []

Section.model_rebuild() # For recursive model

class QueryRequest(BaseModel):
    question: str

class Citation(BaseModel):
    chunk_id: str
    title: Optional[str] = None
    text: str # Full text of the chunk
    start_page: int
    end_page: int
    # TODO: Add other metadata fields from ChromaDB if needed
    # e.g., document_id: str, level: int

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]

class SearchResultItem(BaseModel):
    chunk_id: str
    title: Optional[str] = None
    text: str # Full chunk text
    snippet: Optional[str] = None # A short relevant snippet
    start_page: int
    end_page: int
    # TODO: Add other metadata relevant to search
    # e.g., document_id: str, level: int

class SearchResponse(BaseModel):
    results: List[SearchResultItem]

class HighlightCoordinatesRequest(BaseModel):
    chunk_ids: List[str]

class HighlightItem(BaseModel):
    page: int
    x: float
    y: float
    width: float
    height: float

class HighlightCoordinatesResponse(BaseModel):
    items: List[HighlightItem]


# --- Utility / Placeholder Functions ---

# TODO: Double check this is sorting sections correctly
@lru_cache(maxsize=1)
def load_sections_from_data() -> List[Section]:
    """
    Dynamically load top-level (level=1) and second-level (level=2) sections
    from ChromaDB, build a nested Section tree, and cache the result.
    """
    # 1) fetch level-2 chunks to group under their parents
    lvl2 = collection.get(
        where={"level": 2},
        include=["metadatas"]
    )
    children_map: Dict[str, List[Section]] = {}
    for cid, meta in zip(lvl2["ids"], lvl2["metadatas"]):
        parent = meta.get("parent_id")
        if not parent:
            continue
        children_map.setdefault(parent, []).append(
            Section(
                id=cid,
                title=meta.get("title", ""),
                page=meta.get("start_page", 0),
                subsections=[]
            )
        )

    # 2) fetch level-1 chunks and attach level-2 children
    lvl1 = collection.get(
        where={"level": 1},
        include=["metadatas"],
        # sort=[("metadatas.start_page", "asc")]
    )

    sections: List[Section] = []
    for cid, meta in zip(lvl1["ids"], lvl1["metadatas"]):
        sections.append(
            Section(
                id=cid,
                title=meta.get("title", ""),
                page=meta.get("start_page", 0),
                subsections=children_map.get(cid, [])
            )
        )

    return sections


async def generate_llm_answer_with_rag(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Generates an answer using an LLM with RAG context.
    """
    if not gemini_model:
        # Placeholder if Gemini is not configured
        context_texts = "\n\n".join([chunk.get('text', '') for chunk in context_chunks])
        return f"Placeholder LLM Answer for '{question}'. \n\nRelevant context provided:\n---\n{context_texts[:1000]}...\n---"

    context_str = "\n\n---\n\n".join([
        f"Source (Page {chunk.get('start_page', 'N/A')}, Title: {chunk.get('title', 'N/A')} - ID: ...{chunk.get('chunk_id', '')[-6:]}):\n{chunk.get('text', '')}"
        for chunk in context_chunks
    ])

    prompt = f"""You are a helpful assistant analyzing a legislative bill.
Answer the following question based *only* on the provided context sections from the bill.
If the answer cannot be found in the provided context, state that clearly.
Do not make up information. Be concise and refer to specific page numbers or section titles if helpful.

Question: {question}

Context from the bill:
{context_str}

Answer:
"""
    try:
        response = await gemini_model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Sorry, I encountered an error trying to generate an answer with the LLM."


def get_highlight_coordinates_for_chunks(chunk_ids: List[str]) -> List[HighlightItem]:
    """
    Placeholder: Convert chunk_ids into highlight items with page & bounding boxes.
    In a real implementation, this would involve:
    1. Fetching chunk details (start_char, end_char, page_num) from ChromaDB or another source.
    2. Using a PDF library (like pdfminer.six with more advanced layout analysis, or PyMuPDF)
       to find the bounding box of the text range on the specified page.
    This is a complex task.
    """
    items = []
    # TODO: iterate through chunk_ids, fetch their page, and calculate coordinates.
    if not collection:
        return []
        
    for chunk_id in chunk_ids:
        try:
            result = collection.get(ids=[chunk_id], include=["metadatas"])
            if result and result['metadatas']:
                page_num = result['metadatas'][0].get('start_page', 1)
            else:
                page_num = 1 # Default page
            
            # Placeholder coordinates for highlight boxes
            items.append(HighlightItem(page=page_num, x=10.0, y=15.0, width=70.0, height=10.0)) # Example values
            items.append(HighlightItem(page=page_num, x=15.0, y=30.0, width=60.0, height=5.0))
        except Exception as e:
            print(f"Could not generate placeholder highlight for {chunk_id}: {e}")
    return items

# --- API Endpoints ---

@app.get("/api/sections", response_model=List[Section])
async def api_sections():
    """Return the table of contents sections for the PDF."""
    try:
        sections = load_sections_from_data() # Or dynamically generate
        return sections
    except Exception as e:
        print(f"Error in /api/sections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# TODO: Maybe translate user's query into the vernacular of the document for more accurate vector search.
@app.post("/api/query", response_model=QueryResponse)
async def api_query(req: QueryRequest):
    """Handle a user query, performing RAG and returning an LLM answer and citations."""
    if not collection:
        raise HTTPException(status_code=503, detail="ChromaDB collection not available.")
    try:
        print(f"Received query: {req.question}")
        # 1. Embed the question (ChromaDB does this internally if query_embeddings is not provided)
        # 2. Query ChromaDB for relevant context
        results = collection.query(
            query_texts=[req.question],
            n_results=5, # Number of context chunks to retrieve
            include=["documents", "metadatas"]
        )

        if not results or not results.get('ids') or not results['ids'][0]:
            return QueryResponse(answer="No relevant information found in the document for your query.", citations=[])

        context_chunks_data = []
        citations_for_response: List[Citation] = []

        retrieved_ids = results['ids'][0]
        retrieved_documents = results['documents'][0]
        retrieved_metadatas = results['metadatas'][0] if results['metadatas'] else [{} for _ in retrieved_ids]


        for i, chunk_id in enumerate(retrieved_ids):
            doc_text = retrieved_documents[i]
            meta = retrieved_metadatas[i] if i < len(retrieved_metadatas) else {}
            
            # Prepare context for LLM (full text or relevant parts)
            context_chunks_data.append({
                "chunk_id": chunk_id,
                "text": doc_text,
                "start_page": meta.get("start_page", 0),
                "end_page": meta.get("end_page", 0),
                "title": meta.get("title", f"Chunk {chunk_id[:8]}")
                # Add any other metadata useful for the LLM or citations
            })
            # Prepare citations for frontend
            # TODO: Need to decide whether to sort by page number or display "relevancy" (cosine similarity)
            citations_for_response.append(Citation(
                chunk_id=chunk_id,
                title=meta.get("title", None),
                text=doc_text, # Send full chunk text as per frontend expectation
                start_page=meta.get("start_page", 0),
                end_page=meta.get("end_page", 0)
            ))

        # 3. Generate answer using LLM with context
        llm_answer = await generate_llm_answer_with_rag(req.question, context_chunks_data)

        return QueryResponse(answer=llm_answer, citations=citations_for_response)

    except Exception as e:
        print(f"Error in /api/query: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# TODO: Consider moving this to frontend JS.
@app.get("/api/search", response_model=SearchResponse) # Changed to GET for simplicity with query param
async def api_search(query: str = Query(..., min_length=1)):
    """Performs text search within the document chunks."""
    if not collection:
        raise HTTPException(status_code=503, detail="ChromaDB collection not available.")
    try:
        print(f"Received search: {query}")
        # Option 1: Semantic search + filter (more robust for concepts)
        results = collection.query(
            query_texts=[query],
            n_results=10, # Get more results to filter
            include=["documents", "metadatas", "ids"]
        )
        
        search_items: List[SearchResultItem] = []
        if results and results.get('ids') and results['ids'][0]:
            retrieved_ids = results['ids'][0]
            retrieved_documents = results['documents'][0]
            retrieved_metadatas = results['metadatas'][0] if results['metadatas'] else [{} for _ in retrieved_ids]

            for i, chunk_id in enumerate(retrieved_ids):
                doc_text = retrieved_documents[i]
                meta = retrieved_metadatas[i] if i < len(retrieved_metadatas) else {}

                # Filter if document text contains the query (case-insensitive)
                if query.lower() in doc_text.lower():
                    # Create a snippet (e.g., first N chars or around the query term)
                    try:
                        match_start = doc_text.lower().find(query.lower())
                        snippet_start = max(0, match_start - 50)
                        snippet_end = min(len(doc_text), match_start + len(query) + 50)
                        snippet = f"...{doc_text[snippet_start:snippet_end]}..."
                        if snippet_start > 0: snippet = "..." + snippet
                        if snippet_end < len(doc_text): snippet = snippet + "..."
                    except:
                        snippet = doc_text[:250] + "..."


                    search_items.append(SearchResultItem(
                        chunk_id=chunk_id,
                        title=meta.get("title", None),
                        text=doc_text, # Full text
                        snippet=snippet,
                        start_page=meta.get("start_page", 0),
                        end_page=meta.get("end_page", 0)
                    ))
        
        # Option 2: Direct metadata/document filtering if Chroma metadata is relevant
        # results = collection.get(
        #     where_document={"$contains": query} # This depends on Chroma version and setup
        #     include=["documents", "metadatas"]
        # )
        # ... then process results ...

        return SearchResponse(results=search_items)

    except Exception as e:
        print(f"Error in /api/search: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during search: {str(e)}")


@app.post("/api/highlight-coordinates", response_model=HighlightCoordinatesResponse)
async def api_highlight_coordinates(req: HighlightCoordinatesRequest):
    """
    Return highlight coordinate items based on chunk_ids.
    This is a placeholder for actual PDF coordinate calculation.
    """
    try:
        if not req.chunk_ids:
            return HighlightCoordinatesResponse(items=[])
            
        highlight_items = get_highlight_coordinates_for_chunks(req.chunk_ids)
        # The frontend expects { items: [...] }
        return HighlightCoordinatesResponse(items=highlight_items)
    except Exception as e:
        print(f"Error in /api/highlight-coordinates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# To run: uvicorn server:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
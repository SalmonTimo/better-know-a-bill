#!/usr/bin/env python3
import argparse
import json
import chromadb
from chromadb.utils import embedding_functions # For explicit default EF
import os

# Define a batch size for adding documents to ChromaDB
BATCH_SIZE = 100

def load_chunks_from_json(json_path):
    """Loads chunk data from the specified JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        return chunks
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return None

def prepare_chroma_data(chunks):
    """
    Prepares lists of documents, metadatas, and ids for ChromaDB.
    """
    documents = []
    metadatas = []
    ids = []

    for chunk in chunks:
        # The 'text' field is the document content to be embedded
        documents.append(chunk['text'])

        # The 'chunk_id' is the unique identifier for ChromaDB
        ids.append(chunk['chunk_id'])

        # All other fields become metadata
        # Ensure all metadata values are of supported types (str, int, float, bool)
        metadata = {}
        for key, value in chunk.items():
            if key not in ['text', 'chunk_id']:
                if isinstance(value, list):
                    # Convert lists to a comma-separated string for simple metadata
                    # More complex handling might be needed for specific filtering requirements
                    metadata[key] = ",".join(map(str, value))
                elif value is None:
                    metadata[key] = "" # Or skip, Chroma handles missing keys
                else:
                    metadata[key] = value
        metadatas.append(metadata)

    return documents, metadatas, ids

def store_chunks_in_chroma(chunks, collection_name, persist_directory=None):
    """
    Stores the processed chunks into a ChromaDB collection.
    """
    if not chunks:
        print("No chunks to store.")
        return

    print(f"Preparing to store {len(chunks)} chunks in collection '{collection_name}'.")

    documents, metadatas, ids = prepare_chroma_data(chunks)

    # Setup Chroma client
    if persist_directory:
        print(f"Using persistent ChromaDB storage at: {persist_directory}")
        client = chromadb.PersistentClient(path=persist_directory)
    else:
        print("Using in-memory ChromaDB.")
        client = chromadb.Client() # In-memory client

    # Use the default sentence transformer embedding function
    # This will download the model on first run if not already cached
    sentence_transformer_ef = embedding_functions.DefaultEmbeddingFunction()
    print(f"Using embedding function: Default (Sentence Transformers all-MiniLM-L6-v2)")


    # Get or create the collection
    # The embedding function is now typically associated at the collection level upon creation or retrieval
    print(f"Getting or creating collection: {collection_name}")
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef # Specify EF here
        # metadata={"hnsw:space": "cosine"} # Optional: configure index space
    )
    print(f"Collection '{collection_name}' ready.")

    # Add documents to the collection in batches
    num_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(documents))

        batch_documents = documents[start_idx:end_idx]
        batch_metadatas = metadatas[start_idx:end_idx]
        batch_ids = ids[start_idx:end_idx]

        print(f"Adding batch {i+1}/{num_batches} ({len(batch_documents)} chunks) to collection...")
        try:
            collection.add(
                documents=batch_documents,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        except Exception as e:
            print(f"Error adding batch {i+1} to ChromaDB: {e}")
            # Consider how to handle partial failures: skip, retry, log, etc.
            # For this example, we'll just print the error and continue.
            # Might want to log problematic IDs/documents.
            # For instance, check for duplicate IDs if this error occurs.
            print("Problematic IDs in this batch:", batch_ids)


    print(f"Successfully added {collection.count()} chunks to the '{collection_name}' collection.")

    # Example query (optional)
    if collection.count() > 0:
        print("\n--- Example Query ---")
        try:
            query_text = "What are the provisions for border security technology?"
            if len(chunks) > 5: # Use text from an actual chunk if available
                 query_text = chunks[len(chunks)//2]['text'][:100] # Query with start of a middle chunk
            
            print(f"Querying for: '{query_text}'")
            results = collection.query(
                query_texts=[query_text],
                n_results=min(3, collection.count()), # Get 3 results or fewer if collection is small
                # include=['metadatas', 'documents', 'distances'] # Specify what to include
            )
            print("Query Results:")
            for i, doc_id in enumerate(results['ids'][0]):
                print(f"  Result {i+1}:")
                print(f"    ID: {doc_id}")
                print(f"    Distance: {results['distances'][0][i]:.4f}")
                # print(f"    Metadata: {results['metadatas'][0][i]}") # Can be verbose
                print(f"    Document: {results['documents'][0][i][:200]}...") # Print start of document
        except Exception as e:
            print(f"Error during example query: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Load structured chunks from a JSON file and store them in ChromaDB."
    )
    parser.add_argument(
        "json_path",
        help="Path to the input JSON file containing structured chunks."
    )
    parser.add_argument(
        "--collection_name", "-c",
        default=None,
        help="Name of the ChromaDB collection. Defaults to the document_id of the first chunk or 'default_collection'."
    )
    parser.add_argument(
        "--persist_directory", "-p",
        default=None,
        help="Directory to persist ChromaDB data. If not provided, an in-memory database will be used."
    )
    args = parser.parse_args()

    chunks_data = load_chunks_from_json(args.json_path)

    if chunks_data:
        collection_name = args.collection_name
        if not collection_name:
            # Try to derive collection name from the document_id of the first chunk
            if chunks_data[0] and 'document_id' in chunks_data[0]:
                collection_name = str(chunks_data[0]['document_id'])
            else:
                collection_name = "default_collection"
            collection_name = collection_name.replace(" ", "_").lower() # Sanitize
        
        store_chunks_in_chroma(chunks_data, collection_name, args.persist_directory)

if __name__ == "__main__":
    main()
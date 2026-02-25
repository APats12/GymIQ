"""
Chunks, embeds, and stores fitness abstracts in ChromaDB.
Run this once after download.py.
"""

import json
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "gymiq"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 256


def get_hash(text: str) -> str:
    return hashlib.sha256(text.lower().encode()).hexdigest()


def main():
    with open("data/fitness_abstracts.json") as f:
        abstracts = json.load(f)

    print(f"Loaded {len(abstracts)} abstracts")

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)

    chunks = []
    metadatas = []
    seen_hashes = set()

    for item in abstracts:
        full_text = "\n\n".join(item["contexts"])
        if item.get("answer"):
            full_text += "\n\n" + item["answer"]

        for i, chunk in enumerate(splitter.split_text(full_text)):
            h = get_hash(chunk)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            chunks.append(chunk)
            metadatas.append({
                "pubmed_id": item["pubmed_id"],
                "question": item["question"][:200],
                "chunk_index": i,
            })

    print(f"Total unique chunks to embed: {len(chunks)}")
    print("Loading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding chunks (runs locally, no API needed)...")
    embeddings = model.encode(chunks, batch_size=BATCH_SIZE, show_progress_bar=True)

    print("Storing in ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(COLLECTION_NAME)

    CHROMA_BATCH = 5000
    embeddings_list = embeddings.tolist()
    for i in range(0, len(chunks), CHROMA_BATCH):
        collection.add(
            documents=chunks[i: i + CHROMA_BATCH],
            embeddings=embeddings_list[i: i + CHROMA_BATCH],
            metadatas=metadatas[i: i + CHROMA_BATCH],
            ids=[f"chunk_{j}" for j in range(i, min(i + CHROMA_BATCH, len(chunks)))],
        )
        print(f"  Stored {min(i + CHROMA_BATCH, len(chunks))}/{len(chunks)} chunks")

    print(f"\nDone! {len(chunks)} chunks stored in ChromaDB at {CHROMA_PATH}")


if __name__ == "__main__":
    main()

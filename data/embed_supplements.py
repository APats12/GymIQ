"""
Chunks, embeds, and adds supplement abstracts to the existing ChromaDB collection.
Run after fetch_supplements.py. Does NOT wipe the existing PubMed or exercise data.
"""

import json
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PATH      = "./chroma_db"
COLLECTION_NAME  = "gymiq"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
INPUT_FILE       = "data/supplement_abstracts.json"
CHUNK_SIZE       = 400
CHUNK_OVERLAP    = 80


def get_hash(text: str) -> str:
    return hashlib.sha256(text.lower().encode()).hexdigest()


def main():
    with open(INPUT_FILE) as f:
        abstracts = json.load(f)

    print(f"Loaded {len(abstracts)} supplement abstracts")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    chunks    = []
    metadatas = []
    ids       = []
    seen_hashes = set()

    # Load existing IDs from ChromaDB to avoid collisions
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection    = chroma_client.get_collection(COLLECTION_NAME)
    existing_count = collection.count()
    print(f"Collection currently has {existing_count} documents")

    for item in abstracts:
        full_text = "\n\n".join(item["contexts"])
        if item.get("answer"):
            full_text += "\n\n" + item["answer"]

        for i, chunk in enumerate(splitter.split_text(full_text)):
            h = get_hash(chunk)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            chunk_id = f"supp_{item['pubmed_id']}_{i}"
            chunks.append(chunk)
            metadatas.append({
                "source":     "pubmed_supplement",
                "pubmed_id":  item["pubmed_id"],
                "question":   item["question"][:200],
                "supplement": item.get("supplement", ""),
                "chunk_index": i,
            })
            ids.append(chunk_id)

    print(f"Total unique chunks to embed: {len(chunks)}")
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding chunks...")
    embeddings = model.encode(chunks, batch_size=256, show_progress_bar=True).tolist()

    print("Adding to ChromaDB...")
    BATCH = 5000
    for i in range(0, len(chunks), BATCH):
        collection.add(
            documents =chunks[i:i + BATCH],
            embeddings=embeddings[i:i + BATCH],
            metadatas =metadatas[i:i + BATCH],
            ids       =ids[i:i + BATCH],
        )
        print(f"  Stored {min(i + BATCH, len(chunks))}/{len(chunks)} chunks")

    print(f"\nDone! Collection now has {collection.count()} total documents.")


if __name__ == "__main__":
    main()

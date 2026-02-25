"""
One-time migration: exports all vectors from local ChromaDB and uploads to Pinecone.
Run once before deploying. Requires PINECONE_API_KEY in .env.
Takes 5-15 minutes for 65K vectors.
"""

import os
import time
from dotenv import load_dotenv
import chromadb
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "gymiq"
PINECONE_INDEX  = "gymiq"
EMBEDDING_DIM   = 384   # all-MiniLM-L6-v2 output size
CHROMA_FETCH    = 1000  # rows fetched from ChromaDB per round
PINECONE_BATCH  = 200   # vectors upserted to Pinecone per call


def wait_for_index(pc: Pinecone, name: str):
    print("  Waiting for index to be ready...", end="", flush=True)
    while not pc.describe_index(name).status["ready"]:
        print(".", end="", flush=True)
        time.sleep(2)
    print(" ready!")


def main():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY not found in .env")
        return

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    print("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection    = chroma_client.get_collection(COLLECTION_NAME)
    total         = collection.count()
    print(f"  {total} documents to migrate")

    # ── Pinecone ──────────────────────────────────────────────────────────────
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=api_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        print(f"  Creating index '{PINECONE_INDEX}' (dim={EMBEDDING_DIM}, cosine)...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        wait_for_index(pc, PINECONE_INDEX)
    else:
        print(f"  Index '{PINECONE_INDEX}' already exists")

    index = pc.Index(PINECONE_INDEX)

    # ── Fetch all IDs from ChromaDB ───────────────────────────────────────────
    print("Fetching all IDs from ChromaDB...")
    all_ids = collection.get(include=[])["ids"]
    print(f"  {len(all_ids)} IDs fetched")

    # ── Upload in batches ─────────────────────────────────────────────────────
    print("Uploading to Pinecone...")
    uploaded = 0

    for i in range(0, len(all_ids), CHROMA_FETCH):
        batch_ids = all_ids[i : i + CHROMA_FETCH]

        result = collection.get(
            ids=batch_ids,
            include=["documents", "embeddings", "metadatas"],
        )

        vectors = []
        for id_, doc, emb, meta in zip(
            result["ids"],
            result["documents"],
            result["embeddings"],
            result["metadatas"],
        ):
            pinecone_meta = {k: v for k, v in meta.items() if v is not None}
            pinecone_meta["text"] = doc[:3500]  # well under 40KB limit
            vectors.append({"id": id_, "values": emb, "metadata": pinecone_meta})

        # Upsert in Pinecone-sized sub-batches
        for j in range(0, len(vectors), PINECONE_BATCH):
            index.upsert(vectors=vectors[j : j + PINECONE_BATCH])
            uploaded += len(vectors[j : j + PINECONE_BATCH])

        print(f"  {uploaded}/{total} uploaded ({uploaded * 100 // total}%)")
        time.sleep(0.05)

    print(f"\nDone! {uploaded} vectors in Pinecone.")
    stats = index.describe_index_stats()
    print(f"Index stats: {stats.total_vector_count} total vectors")


if __name__ == "__main__":
    main()

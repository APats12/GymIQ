"""
Embeds ExerciseDB exercises and adds them to the existing ChromaDB collection.
Run after fetch_exercises.py. Does NOT wipe the existing PubMed data.
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "gymiq"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INPUT_FILE = "data/exercises.json"


def exercise_to_text(ex: dict) -> str:
    secondary = ", ".join(ex.get("secondaryMuscles", [])) or "none"
    instructions = ex.get("instructions", [])
    if isinstance(instructions, list):
        steps = "\n".join(f"{i+1}. {step}" for i, step in enumerate(instructions))
    else:
        steps = str(instructions)

    description = ex.get("description", "").strip()

    parts = [
        f"Exercise: {ex['name'].title()}",
        f"Body Part: {ex['bodyPart'].title()} | Target Muscle: {ex['target'].title()}",
        f"Secondary Muscles: {secondary}",
        f"Equipment: {ex.get('equipment', 'unknown').title()} | Difficulty: {ex.get('difficulty', 'unknown').title()} | Category: {ex.get('category', 'unknown').title()}",
    ]
    if description:
        parts += ["", "Description:", description]
    if steps:
        parts += ["", "How to perform:", steps]

    return "\n".join(parts)


def main():
    with open(INPUT_FILE) as f:
        exercises = json.load(f)

    print(f"Loaded {len(exercises)} exercises")

    docs = []
    metadatas = []
    ids = []

    for ex in exercises:
        text = exercise_to_text(ex)
        docs.append(text)
        metadatas.append({
            "source": "exercisedb",
            "exercise_id": str(ex["id"]),
            "name": ex["name"],
            "body_part": ex["bodyPart"],
            "target": ex["target"],
            "difficulty": ex.get("difficulty", ""),
            "pubmed_id": "",
            "question": ex["name"],
        })
        ids.append(f"exercise_{ex['id']}")

    print(f"Embedding {len(docs)} exercises...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(docs, show_progress_bar=True).tolist()

    print("Adding to ChromaDB collection...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    collection.add(
        documents=docs,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"\nDone! Added {len(docs)} exercises to '{COLLECTION_NAME}' collection.")
    print(f"Collection now has {collection.count()} total documents.")


if __name__ == "__main__":
    main()

"""
Downloads and filters pubmed_qa for fitness/sports science abstracts.
Saves ~10,000 relevant records to data/fitness_abstracts.json
"""

import json
import os
from datasets import load_dataset

TARGET = 10000

KEYWORDS = [
    "exercise", "training", "muscle", "strength", "cardio", "supplement",
    "creatine", "protein", "fitness", "hypertrophy", "testosterone",
    "fat loss", "weight loss", "nutrition", "athletic", "endurance",
    "resistance training", "aerobic", "anaerobic", "body composition",
    "caffeine", "bcaa", "pre-workout", "recovery", "sport", "gym",
    "bench press", "squat", "deadlift", "vo2", "lean mass", "whey",
]


def is_fitness_related(example) -> bool:
    text = (
        example.get("question", "") + " " +
        example.get("long_answer", "") + " " +
        " ".join(example.get("context", {}).get("contexts", []))
    ).lower()
    return any(kw in text for kw in KEYWORDS)


def main():
    print("Streaming pubmed_qa dataset (pqa_artificial)...")
    ds = load_dataset("pubmed_qa", "pqa_artificial", split="train", streaming=True)

    filtered = []
    checked = 0

    for example in ds:
        checked += 1
        if checked % 10000 == 0:
            print(f"  Checked {checked} records, found {len(filtered)} fitness abstracts...")

        if is_fitness_related(example):
            filtered.append({
                "pubmed_id": str(example["pubid"]),
                "question": example["question"],
                "contexts": example["context"]["contexts"],
                "answer": example["long_answer"],
                "decision": example["final_decision"],
            })

        if len(filtered) >= TARGET:
            break

    print(f"\nDone. Found {len(filtered)} fitness abstracts from {checked} records checked.")

    os.makedirs("data", exist_ok=True)
    with open("data/fitness_abstracts.json", "w") as f:
        json.dump(filtered, f)

    print("Saved to data/fitness_abstracts.json")


if __name__ == "__main__":
    main()

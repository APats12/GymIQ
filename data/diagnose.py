"""
Quick diagnostic to check dataset quality and coverage.
"""
import json
from collections import Counter

GYM_TERMS = [
    "creatine", "protein", "hypertrophy", "muscle mass", "strength training",
    "resistance training", "bench press", "squat", "testosterone", "whey",
    "caffeine", "bcaa", "fat loss", "lean mass", "body composition",
]

with open("data/fitness_abstracts.json") as f:
    all_abstracts = json.load(f)

first_2k = all_abstracts[:2000]
rest = all_abstracts[2000:]

print(f"Total abstracts: {len(all_abstracts)}")
print(f"Embedded (first 2K): {len(first_2k)}")
print(f"Not embedded (remaining): {len(rest)}\n")

print("Term coverage in FIRST 2000 (what's in ChromaDB):")
for term in GYM_TERMS:
    count = sum(1 for a in first_2k if term in (
        " ".join(a["contexts"]) + a.get("answer", "")
    ).lower())
    print(f"  {term:<25} {count:>4} abstracts")

print("\nTerm coverage in REMAINING 8000 (not embedded):")
for term in GYM_TERMS:
    count = sum(1 for a in rest if term in (
        " ".join(a["contexts"]) + a.get("answer", "")
    ).lower())
    print(f"  {term:<25} {count:>4} abstracts")

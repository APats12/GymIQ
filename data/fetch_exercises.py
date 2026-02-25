"""
Fetches all exercises from ExerciseDB API (via RapidAPI) and saves to JSON.
Free tier: 1000 requests/month. ExerciseDB has ~1300 exercises, fetched in pages of 100.
Run once, then run embed_exercises.py to add to ChromaDB.
"""

import json
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
BASE_URL = "https://exercisedb.p.rapidapi.com/exercises"
HEADERS = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": "exercisedb.p.rapidapi.com",
}
OUTPUT_FILE = "data/exercises.json"

BODY_PARTS = [
    "back", "cardio", "chest", "lower arms", "lower legs",
    "neck", "shoulders", "upper arms", "upper legs", "waist",
]

# Target muscles that need dedicated coverage beyond the body part buckets.
# "back" body part returns mostly lats — these fill in upper/lower back,
# quads/hamstrings/glutes (all crammed into "upper legs"), and abs vs obliques.
TARGET_MUSCLES = [
    "upper back", "traps", "spine",           # upper and lower back
    "quads", "hamstrings", "glutes",           # upper legs breakdown
    "abs", "obliques",                         # waist breakdown
    "biceps", "triceps",                       # upper arms breakdown
    "delts",                                   # shoulder detail
]


def fetch_by_body_part(body_part: str) -> list[dict]:
    url = f"{BASE_URL}/bodyPart/{requests.utils.quote(body_part)}"
    params = {"limit": 10, "offset": 0}
    resp = requests.get(url, headers=HEADERS, params=params)
    if resp.status_code != 200:
        print(f"  Error {resp.status_code} for {body_part}: {resp.text}")
        return []
    return resp.json()


def fetch_by_target(target: str) -> list[dict]:
    url = f"{BASE_URL}/target/{requests.utils.quote(target)}"
    params = {"limit": 10, "offset": 0}
    resp = requests.get(url, headers=HEADERS, params=params)
    if resp.status_code != 200:
        print(f"  Error {resp.status_code} for target={target}: {resp.text}")
        return []
    return resp.json()


def main():
    if not RAPIDAPI_KEY:
        print("Error: RAPIDAPI_KEY not found in .env")
        return

    exercises = []
    seen_ids = set()

    print("Fetching exercises by body part...")
    for body_part in BODY_PARTS:
        print(f"  {body_part}...")
        batch = fetch_by_body_part(body_part)
        for ex in batch:
            if ex["id"] not in seen_ids:
                seen_ids.add(ex["id"])
                exercises.append(ex)
        print(f"    +{len(batch)} (total: {len(exercises)})")
        time.sleep(0.3)

    print("\nFetching exercises by target muscle...")
    for target in TARGET_MUSCLES:
        print(f"  {target}...")
        batch = fetch_by_target(target)
        new = 0
        for ex in batch:
            if ex["id"] not in seen_ids:
                seen_ids.add(ex["id"])
                exercises.append(ex)
                new += 1
        print(f"    +{new} new (total: {len(exercises)})")
        time.sleep(0.3)

    print(f"\nTotal exercises fetched: {len(exercises)}")

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(exercises, f)

    print(f"Saved to {OUTPUT_FILE}")

    # Preview first exercise to confirm structure
    if exercises:
        ex = exercises[0]
        print(f"\nSample exercise fields: {list(ex.keys())}")
        print(f"Example: {ex.get('name')} — {ex.get('bodyPart')} / {ex.get('target')}")


if __name__ == "__main__":
    main()

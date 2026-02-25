"""
Fetches targeted PubMed abstracts for top gym supplements via NCBI E-utilities.
No API key needed (rate-limited to 3 req/sec). Saves to data/supplement_abstracts.json.
Run once, then run embed_supplements.py to add to ChromaDB.
"""

import json
import os
import time
import xml.etree.ElementTree as ET
import requests

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
OUTPUT_FILE = "data/supplement_abstracts.json"

RESULTS_PER_SUPPLEMENT = 40
FETCH_BATCH_SIZE = 20
RATE_LIMIT_DELAY = 0.4  # stay under 3 req/sec without an API key

# Top gym supplements - queries are tuned to human exercise/performance studies
SUPPLEMENTS = [
    ("creatine",         '"creatine"[Title/Abstract] AND (exercise OR "muscle strength" OR "lean mass" OR performance) AND "humans"[MeSH Terms]'),
    ("whey protein",     '"whey protein"[Title/Abstract] AND (exercise OR "muscle mass" OR hypertrophy OR recovery) AND "humans"[MeSH Terms]'),
    ("caffeine",         '"caffeine"[Title/Abstract] AND (exercise OR performance OR endurance OR strength OR "fat loss") AND "humans"[MeSH Terms]'),
    ("beta-alanine",     '"beta-alanine"[Title/Abstract] AND (exercise OR performance OR fatigue OR endurance) AND "humans"[MeSH Terms]'),
    ("BCAA",             '("branched-chain amino acids" OR BCAA) AND (exercise OR "muscle protein" OR recovery OR hypertrophy) AND "humans"[MeSH Terms]'),
    ("citrulline",       '"citrulline"[Title/Abstract] AND (exercise OR performance OR "nitric oxide" OR pump OR endurance) AND "humans"[MeSH Terms]'),
    ("fish oil",         '"fish oil"[Title/Abstract] AND (exercise OR "muscle mass" OR recovery OR inflammation OR "body composition") AND "humans"[MeSH Terms]'),
    ("vitamin D",        '"vitamin D"[Title/Abstract] AND (exercise OR "muscle strength" OR performance OR testosterone) AND "humans"[MeSH Terms]'),
    ("magnesium",        '"magnesium"[Title/Abstract] AND (exercise OR "muscle function" OR performance OR recovery OR testosterone) AND "humans"[MeSH Terms]'),
    ("zinc",             '"zinc"[Title/Abstract] AND (exercise OR testosterone OR "muscle mass" OR performance) AND "humans"[MeSH Terms]'),
    ("glutamine",        '"glutamine"[Title/Abstract] AND (exercise OR recovery OR "muscle mass" OR performance) AND "humans"[MeSH Terms]'),
    ("carnitine",        '"carnitine"[Title/Abstract] AND (exercise OR "fat oxidation" OR "body composition" OR endurance OR performance) AND "humans"[MeSH Terms]'),
    ("HMB",              '"HMB"[Title/Abstract] AND (exercise OR "muscle mass" OR "lean body mass" OR strength OR recovery) AND "humans"[MeSH Terms]'),
    ("casein protein",   '"casein"[Title/Abstract] AND (exercise OR "muscle protein synthesis" OR recovery OR "muscle mass") AND "humans"[MeSH Terms]'),
    ("pre-workout",      '("pre-workout" OR "pre workout") AND (exercise OR performance OR strength OR endurance) AND "humans"[MeSH Terms]'),
]


def search_pubmed(query: str, max_results: int) -> list[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    resp = requests.get(ESEARCH_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()["esearchresult"]["idlist"]


def fetch_abstracts_xml(pmids: list[str]) -> str:
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    resp = requests.get(EFETCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    records = []

    for article in root.findall(".//PubmedArticle"):
        pmid_el    = article.find(".//PMID")
        title_el   = article.find(".//ArticleTitle")
        abstract_els = article.findall(".//AbstractText")

        if pmid_el is None or not abstract_els:
            continue

        # Concatenate structured abstract sections (Background, Methods, etc.)
        parts = []
        for el in abstract_els:
            label = el.get("Label")
            text  = "".join(el.itertext()).strip()
            if text:
                parts.append(f"{label}: {text}" if label else text)

        abstract = "\n".join(parts)
        if len(abstract) < 100:
            continue

        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        records.append({
            "pubmed_id": pmid_el.text,
            "question": title,
            "contexts": [abstract],
            "answer": "",
            "decision": "",
        })

    return records


def main():
    all_records = []
    seen_pmids = set()

    for name, query in SUPPLEMENTS:
        print(f"\n[{name}] Searching PubMed...")
        try:
            pmids = search_pubmed(query, RESULTS_PER_SUPPLEMENT)
        except Exception as e:
            print(f"  Search failed: {e}")
            continue

        time.sleep(RATE_LIMIT_DELAY)
        print(f"  Found {len(pmids)} PMIDs — fetching abstracts...")

        new_for_supplement = 0
        for i in range(0, len(pmids), FETCH_BATCH_SIZE):
            batch = [p for p in pmids[i:i + FETCH_BATCH_SIZE] if p not in seen_pmids]
            if not batch:
                continue
            try:
                xml_text = fetch_abstracts_xml(batch)
                records  = parse_xml(xml_text)
                for r in records:
                    if r["pubmed_id"] not in seen_pmids:
                        seen_pmids.add(r["pubmed_id"])
                        r["supplement"] = name  # tag for debugging
                        all_records.append(r)
                        new_for_supplement += 1
            except Exception as e:
                print(f"  Fetch error: {e}")
            time.sleep(RATE_LIMIT_DELAY)

        print(f"  Added {new_for_supplement} abstracts (total: {len(all_records)})")

    print(f"\nTotal supplement abstracts: {len(all_records)}")
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_records, f)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

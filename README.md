<p align="center">
  <a href="https://github.com/APats12/GymIQ/stargazers"><img src="https://img.shields.io/github/stars/APats12/GymIQ.svg?style=for-the-badge" alt="Stars"></a>
  <a href="https://github.com/APats12/GymIQ/issues"><img src="https://img.shields.io/github/issues/APats12/GymIQ.svg?style=for-the-badge" alt="Issues"></a>
  <a href="https://github.com/APats12/GymIQ/network/members"><img src="https://img.shields.io/github/forks/APats12/GymIQ.svg?style=for-the-badge" alt="Forks"></a>
</p>

<div align="center">
  <h1>🏋️ GymIQ</h1>
  <p><strong>Ask any fitness or supplement question — answered by real science, not bro science.</strong></p>

  <a href="https://gym-iq.streamlit.app">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" height="35px" alt="Live Demo">
  </a>
</div>

<h1></h1>

GymIQ is a **RAG-powered fitness assistant** that searches 65,000+ research chunks from PubMed and ExerciseDB to answer your gym and nutrition questions with real evidence.

Ask *"Does creatine improve strength?"* and get a synthesized answer across dozens of actual clinical studies — with sources. Then hit **💪 Gym Bro Mode** to get the same answer delivered with maximum hype.

---

## How It Works

GymIQ uses **Retrieval-Augmented Generation (RAG)**:

1. Your question is embedded into a 384-dim vector using `all-MiniLM-L6-v2`
2. The top 20 most semantically similar chunks are retrieved from Pinecone (65K+ vectors)
3. LLaMA 3.3 70B (via Groq) synthesizes the research into a clear, conflict-aware answer
4. **Gym Bro Mode** optionally retranslates the answer into gym slang

```
Question → Embed → Pinecone Search (top 20) → LLaMA 3.3 70B → Answer
                                                      ↓
                                               [💪 Gym Bro Mode]
```

---

## 📊 The Data

| Source | Content | Chunks |
|--------|---------|--------|
| PubMed (NCBI) | General fitness & exercise research abstracts | ~62,000 |
| PubMed (NCBI) | Targeted supplement abstracts (15 supplements) | ~3,272 |
| ExerciseDB | Exercise instructions, muscles, equipment, difficulty | 149 |
| **Total** | | **65,473 vectors** |

### Supplements Covered

Creatine · Whey Protein · Caffeine · Beta-Alanine · BCAA · Citrulline · Fish Oil · Vitamin D · Magnesium · Zinc · Glutamine · Carnitine · HMB · Casein Protein · Pre-Workout

---

## ✅ Features

- ✅ **65,000+ research chunks** from PubMed and ExerciseDB
- ✅ **Conflict-aware synthesis** — if studies disagree, the answer reports both sides
- ✅ **💪 Gym Bro Mode** — translates science into hype
- ✅ **Source viewer** — see the actual abstracts and PubMed IDs behind every answer
- ✅ **Color-coded source badges** — 🟢 Exercise · 🔵 Research · 🟣 Supplement Research
- ✅ **Cloud deployed** — no local setup needed, just visit the live demo

---

## 🚀 Run Locally

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com) (free)
- A [Pinecone API key](https://www.pinecone.io) (free tier)

### Setup

1. Clone the repo:

```bash
git clone https://github.com/APats12/GymIQ.git
cd GymIQ
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key_here
PINECONE_API_KEY=your_pinecone_key_here
```

4. Run the app:

```bash
streamlit run app.py
```

> The Pinecone index (`gymiq`) is already populated with 65,473 vectors — you don't need to re-run the data pipeline to use the app.

---

## 🗂️ Project Structure

```
GymIQ/
├── app.py                     # Streamlit app — UI + RAG logic
├── requirements.txt           # Python dependencies
├── .env                       # API keys (not committed)
└── data/
    ├── fetch_exercises.py     # Pulls exercises from ExerciseDB API
    ├── embed_exercises.py     # Embeds exercises into ChromaDB
    ├── fetch_supplements.py   # Pulls supplement abstracts from NCBI
    ├── embed_supplements.py   # Embeds supplement abstracts into ChromaDB
    └── upload_to_pinecone.py  # One-time migration: ChromaDB → Pinecone
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Embeddings | `sentence-transformers` · all-MiniLM-L6-v2 (384-dim) |
| Vector DB | Pinecone (serverless, cosine similarity) |
| LLM | LLaMA 3.3 70B via Groq |
| Research Data | NCBI PubMed E-utilities API |
| Exercise Data | ExerciseDB via RapidAPI |
| Hosting | Streamlit Community Cloud |

---

## 🔑 Environment Variables

| Variable | Where to get it |
|----------|----------------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |
| `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io) |
| `RAPIDAPI_KEY` | [rapidapi.com](https://rapidapi.com) — only needed to re-fetch exercises |

---

## 📬 Rebuilding the Data Pipeline

If you want to re-populate the vector database from scratch:

```bash
# 1. Fetch and embed exercises (requires RAPIDAPI_KEY)
python data/fetch_exercises.py
python data/embed_exercises.py

# 2. Fetch and embed supplement abstracts (NCBI, no key required)
python data/fetch_supplements.py
python data/embed_supplements.py

# 3. Upload everything to Pinecone
python data/upload_to_pinecone.py
```

> `fetch_supplements.py` is rate-limited to ~3 req/sec by NCBI. `upload_to_pinecone.py` takes 5–15 minutes for 65K vectors.

---

## 🔎 Example Questions

- *Does creatine improve strength?*
- *Best exercises for upper back?*
- *How much protein do I need to build muscle?*
- *Does caffeine improve athletic performance?*
- *What's the difference between fast and slow carbs for recovery?*

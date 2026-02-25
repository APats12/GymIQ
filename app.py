import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

LLM_MODEL      = "llama-3.3-70b-versatile"
PINECONE_INDEX = "gymiq"

_embedder = None
_pinecone_index = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def embed_query(text: str) -> list[float]:
    return get_embedder().encode(text).tolist()


def get_groq_client() -> Groq:
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        _pinecone_index = pc.Index(PINECONE_INDEX)
    return _pinecone_index


def answer_question(question: str) -> tuple[str, list[dict]]:
    query_embedding = embed_query(question)

    results = get_pinecone_index().query(
        vector=query_embedding,
        top_k=20,
        include_metadata=True,
    )

    docs  = [m.metadata.get("text", "") for m in results.matches]
    metas = [m.metadata for m in results.matches]
    context = "\n\n---\n\n".join(docs)

    response = get_groq_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a fitness and sports science assistant. "
                    "You are given multiple PubMed research abstracts. Your job is to synthesize findings across ALL of them, not just one. "
                    "Important rules:\n"
                    "- If studies conflict, report BOTH sides (e.g. 'some studies show X, while others find Y')\n"
                    "- Do not draw a conclusion from a single study if others contradict it\n"
                    "- Note if results depend on dose, population, or training status\n"
                    "- Be specific and cite findings, but keep the answer practical\n"
                    "- If none of the abstracts are relevant, say: 'I couldn't find relevant research on this in the database.'"
                ),
            },
            {
                "role": "user",
                "content": f"Research abstracts:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0,
        max_tokens=512,
    )

    answer = response.choices[0].message.content

    sources = [
        {
            "text": doc,
            "question": meta.get("question", ""),
            "pubmed_id": meta.get("pubmed_id", ""),
            "source": meta.get("source", "pubmed"),
            "name": meta.get("name", ""),
        }
        for doc, meta in zip(docs, metas)
    ]

    return answer, sources


def translate_to_gymbro(scientific_answer: str) -> str:
    response = get_groq_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a hyped-up gym bro who translates scientific fitness research into "
                    "simple, energetic gym slang. Use words like: bro, gains, swole, jacked, "
                    "crushing it, beast mode, pump, PR, grind, no days off, get after it. "
                    "Keep the actual facts accurate but make it sound like you're hyping up "
                    "your buddy before a workout. Keep it short — 3 to 5 sentences max. "
                    "If the original answer says there's no info available, say something like "
                    "'Bro the science hasn't caught up to your grind yet, but keep lifting!'"
                ),
            },
            {
                "role": "user",
                "content": f"Translate this into gym bro language:\n\n{scientific_answer}",
            },
        ],
        temperature=0.8,
        max_tokens=200,
    )
    return response.choices[0].message.content


# ── UI ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="GymIQ", page_icon="🏋️", layout="centered")

st.markdown("""
<style>
/* Gym Bro button */
div[data-testid="stHorizontalBlock"] div:first-child .stButton > button {
    background: linear-gradient(135deg, #ff6b35, #f7931e);
    color: white;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1.25rem;
    font-size: 1rem;
    transition: opacity 0.2s;
}
div[data-testid="stHorizontalBlock"] div:first-child .stButton > button:hover {
    opacity: 0.85;
    color: white;
    border: none;
}
/* Gym Bro translation box */
.gymbro-box {
    background: linear-gradient(135deg, #1c1c2e, #2d1b4e);
    border-left: 4px solid #f7931e;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-top: 0.25rem;
    color: #f7c59f;
    font-size: 1rem;
    font-weight: 500;
    line-height: 1.6;
}
/* Source badges */
.badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-right: 6px;
    vertical-align: middle;
}
.badge-exercise  { background: #2ecc71; color: #fff; }
.badge-research  { background: #3498db; color: #fff; }
.badge-supplement { background: #9b59b6; color: #fff; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🏋️ GymIQ")
st.caption("Ask any fitness or supplement question — answered by real PubMed research & ExerciseDB.")
st.divider()

if not os.getenv("GROQ_API_KEY") or not os.getenv("PINECONE_API_KEY"):
    st.error("Add GROQ_API_KEY and PINECONE_API_KEY to the .env file before running.")
    st.stop()

question = st.text_input(
    "Ask a question",
    placeholder="Does creatine improve strength? Best exercises for upper back?",
    label_visibility="collapsed",
)

if question:
    with st.spinner("Searching 65,000+ research chunks..."):
        try:
            answer, sources = answer_question(question)
            st.session_state["last_answer"] = answer
            st.session_state["last_sources"] = sources
            st.session_state.pop("bro_translation", None)
        except Exception as e:
            st.error(f"Error: {e}")

if "last_answer" in st.session_state:
    st.markdown("### 🔬 Research Says")
    st.write(st.session_state["last_answer"])

    st.markdown("")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("💪 Gym Bro Mode"):
            with st.spinner("Getting hyped..."):
                bro = translate_to_gymbro(st.session_state["last_answer"])
                st.session_state["bro_translation"] = bro

    if "bro_translation" in st.session_state:
        st.markdown("### 💪 Gym Bro Says")
        st.markdown(
            f'<div class="gymbro-box">{st.session_state["bro_translation"]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    with st.expander("View sources"):
        for i, source in enumerate(st.session_state["last_sources"][:6], 1):
            src = source.get("source", "pubmed")
            if src == "exercisedb":
                badge = '<span class="badge badge-exercise">Exercise</span>'
                title = source.get("name", "").title() or "Exercise"
            elif src == "pubmed_supplement":
                badge = '<span class="badge badge-supplement">Supplement Research</span>'
                title = source.get("question", "")
            else:
                badge = '<span class="badge badge-research">Research</span>'
                title = source.get("question", "")

            pmid = source.get("pubmed_id", "")
            pmid_str = f" &nbsp;·&nbsp; PubMed ID: <code>{pmid}</code>" if pmid else ""
            st.markdown(
                f"{badge} **{title}**{pmid_str}" if title else f"{badge}{pmid_str}",
                unsafe_allow_html=True,
            )
            st.caption(source["text"])
            st.divider()

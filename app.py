# app.py
"""
Flask RAG app using Pinecone v8 + Gemini (Flash default).
Patched: robust env handling, extract_generated_text, context-first RAG with fallback.
"""

from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
from dotenv import load_dotenv
from google import genai
import os, traceback, re

# Pinecone v8 client
from pinecone import Pinecone, ServerlessSpec

app = Flask(__name__)

# ------------------------
# Casual / Offensive checks
# ------------------------

def normalize_text(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def is_casual_message(user_input: str) -> str | None:
    s = normalize_text(user_input)

    casual = ["ok", "okay", "k", "fine", "alright", "hmm", "sure", "great", "thanks"]

    if s in casual:
        return "Sure — let me know your question whenever you're ready."

    greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
    for g in greetings:
        if g in s:
            return "Hello! How can I assist you with your medical questions today?"

    casual_responses = {
        "how are you": "I'm doing well! How can I help you with medical information today?",
        "who are you": "I'm a medical assistant chatbot built to answer healthcare questions.",
        "where are you from": "I'm running on a cloud server to help users from anywhere!",
        "thank you": "You're welcome!",
        "thanks": "Glad to help!",
        "thx": "Glad to help!",
    }
    for key, val in casual_responses.items():
        if key in s:
            return val

    return None


def is_offensive(user_input: str) -> bool:
    s = normalize_text(user_input)
    offensive_keywords = [
        "idiot", "stupid", "dumb", "fool", "nonsense",
        "shut up", "kill yourself", "hate you", "moron", "fuck", "bitch", "bastard"
    ]
    for kw in offensive_keywords:
        if kw in s:
            return True
    return False

# ------------------------
# Load environment variables
# ------------------------

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_KEY = (
    os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
    or os.environ.get("GENAI_API_KEY")
)

if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY in environment/.env")
if not GEMINI_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY / GOOGLE_API_KEY / GENAI_API_KEY in environment/.env")

# Gemini client
client = genai.Client(api_key=GEMINI_KEY)

# model (optional override from .env)
MODEL_NAME = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash")
print(f"[app] Using Gemini model: {MODEL_NAME}")

# Optional: print sample available models (non-fatal)
try:
    print("[app] Available models (sample):")
    for i, m in enumerate(client.models.list()):
        if i >= 20:
            break
        print(i + 1, getattr(m, "name", getattr(m, "display_name", repr(m)[:120])))
except Exception:
    print("[app] Warning: unable to list models at startup (non-fatal).")
    traceback.print_exc()

# ------------------------
# Load embeddings
# ------------------------

try:
    embeddings = download_hugging_face_embeddings()
    print("[app] embeddings ready:", type(embeddings))
except Exception as e:
    print("[app] Failed to create embeddings:", e)
    traceback.print_exc()
    raise

# ------------------------
# Pinecone v8 initialization
# ------------------------

index_name = "medicalbot"
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check existing indexes
try:
    idx_list = pc.list_indexes()
    existing_indexes = idx_list.names() if hasattr(idx_list, "names") else list(idx_list)
except Exception:
    existing_indexes = []

# Create index if needed
if index_name not in existing_indexes:
    print(f"[app] Creating Pinecone v8 index {index_name} ...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Get index handle
index_obj = pc.Index(index_name)
print("[app] Using Pinecone v8 index:", index_obj)

# ------------------------
# Custom Pinecone v8 Retriever
# ------------------------

class SimpleDoc:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

class PineconeV8Retriever:
    def __init__(self, index, embeddings, top_k=5):
        self.index = index
        self.embed = embeddings
        self.top_k = top_k

    def get_relevant_documents(self, query, k=None):
        k = k or self.top_k
        query_vec = self.embed.embed_query(query)
        result = self.index.query(
            vector=query_vec,
            top_k=k,
            include_metadata=True
        )
        docs = []
        for match in result.matches:
            docs.append(SimpleDoc(
                page_content=match.metadata.get("text", ""),
                metadata=match.metadata
            ))
        return docs

retriever = PineconeV8Retriever(index_obj, embeddings, top_k=4)
print("[app] Using custom Pinecone v8 retriever.")

# ------------------------
# Helper: robust extractor for model responses
# ------------------------

def extract_generated_text(resp) -> str:
    """
    Robust extraction of text from different SDK response shapes.
    """
    try:
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
    except Exception:
        pass

    try:
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text.strip()
    except Exception:
        pass

    try:
        if hasattr(resp, "data"):
            parts = []
            for item in getattr(resp, "data"):
                if hasattr(item, "text") and item.text:
                    parts.append(item.text)
                else:
                    parts.append(str(item))
            if parts:
                return "\n".join(parts).strip()
    except Exception:
        pass

    try:
        if hasattr(resp, "__iter__") and not isinstance(resp, (str, bytes, dict)):
            parts = []
            for item in resp:
                try:
                    if hasattr(item, "text") and item.text:
                        parts.append(item.text)
                    elif hasattr(item, "output_text") and item.output_text:
                        parts.append(item.output_text)
                    else:
                        parts.append(str(item))
                except Exception:
                    parts.append(str(item))
            if parts:
                return "\n".join(parts).strip()
    except Exception:
        pass

    return repr(resp)[:2000]

# ------------------------
# Manual RAG with Gemini 
# ------------------------

def _build_context_snippets(docs, max_chars=400):
    items = []
    for i, d in enumerate(docs, start=1):
        text = (d.page_content or "").strip()
        meta = d.metadata or {}
        tag = meta.get("source") or meta.get("id") or f"doc#{i}"
        snippet = text.replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rsplit(" ", 1)[0] + "..."
        items.append((tag, snippet))
    return items

def clean_tldr_prefix(text: str) -> str:
    """
    Remove leading TL;DR or similar prefixes and tidy whitespace.
    Keeps the rest of the generated answer intact.
    """
    if not text:
        return text
    t = text.strip()
    # common TL;DR variants at start
    t = re.sub(r"^(?:-\\s*)?(tl;dr)\\s*[:\\-\\s]*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^tl\\s*;\\s*dr\\s*[:\\-\\s]*", "", t, flags=re.IGNORECASE)
    return t.strip()

def manual_rag_answer(question: str, retriever, system_prompt_text: str, k: int = 4) -> str:
    # 1) Retrieve docs and create short snippets
    docs = retriever.get_relevant_documents(question, k=k)
    snippets = _build_context_snippets(docs)
    context_text = "\n\n".join([f"[{tag}] {s}" for tag, s in snippets]).strip()

    # 2) Build prompt preferring context but allowing labeled general knowledge fallback
    prompt = (
        f"{system_prompt_text}\n\n"
        f"Context snippets (most relevant first):\n{context_text}\n\n"
        f"User question: {question}\n\n"
        "Answer now following the system instructions above."
    )

    model_name = os.environ.get("GEMINI_MODEL", MODEL_NAME)
    resp = client.models.generate_content(model=model_name, contents=prompt)
    out = extract_generated_text(resp) or ""
    out = clean_tldr_prefix(out).strip()

    # 3) If response is too short or explicitly refuses, do labeled general-knowledge fallback
    lowering = out.lower()
    vague = (
        len(out) < 40
        or "i don't know" in lowering
        or "do not know" in lowering
        or "not in the context" in lowering
        or "cannot answer" in lowering
    )

    if vague:
        fallback_prompt = (
            f"{system_prompt_text}\n\n"
            "NOTE: The provided context did not fully answer the question. You MAY now supplement using general, evidence-based medical knowledge. "
            "If you supplement, clearly label those parts as [General knowledge].\n\n"
            f"Context snippets (most relevant first):\n{context_text}\n\n"
            f"User question: {question}\n\n"
            "Answer now following the system instructions above."
        )
        try:
            resp2 = client.models.generate_content(model=model_name, contents=fallback_prompt)
            out2 = extract_generated_text(resp2) or ""
            out2 = clean_tldr_prefix(out2).strip()
            return "[General knowledge fallback] " + out2
        except Exception:
            return out + "\n\n[Note: unable to produce a fuller answer automatically.]"

    return out

# ------------------------
# Flask Routes
# ------------------------

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return jsonify({"error": "empty_input"}), 400

    if is_offensive(msg):
        return jsonify({"answer": "Let's stay respectful. I'm here to help with medical information."})

    casual = is_casual_message(msg)
    if casual:
        return jsonify({"answer": casual})

    try:
        answer = manual_rag_answer(msg, retriever, system_prompt)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "internal_error", "detail": str(e)}), 500

    return jsonify({"answer": answer})

# ------------------------
# Main
# ------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

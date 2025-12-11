# store_index.py
"""
Robust index creation + upsert script compatible with:
 - newer Pinecone client (Pinecone class)
 - legacy pinecone module
It checks for existing index, ignores ALREADY_EXISTS errors, and upserts in batches.
"""

import os
import uuid
from dotenv import load_dotenv
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")  # optional for legacy client
if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY in environment")

# Try modern Pinecone client first, else fallback to legacy package
try:
    # modern API
    from pinecone import Pinecone, ServerlessSpec
    from pinecone.exceptions import PineconeApiException
    NEW_PINECONE = True
    print("[store_index] Using NEW Pinecone client API.")
except Exception:
    import pinecone
    try:
        from pinecone.exceptions import PineconeApiException  # may still exist in legacy packaging
    except Exception:
        PineconeApiException = Exception
    NEW_PINECONE = False
    print("[store_index] Using LEGACY pinecone module API.")

INDEX_NAME = "medicalbot"
DIM = 384
BATCH = 100

# load and chunk PDFs
print("[store_index] Loading PDFs from Data/ ...")
docs = load_pdf_file("Data/")
chunks = text_split(docs)
texts = [getattr(d, "page_content", str(d)) for d in chunks]
if not texts:
    raise RuntimeError("No text chunks found in Data/ — check Data/ contains PDFs")
print(f"[store_index] Found {len(texts)} text chunks.")

# --- embeddings object ---
print("[store_index] Initializing embeddings ...")
emb = download_hugging_face_embeddings()

def embed_texts(txts):
    """Return list[list[float]] for the input txts."""
    if hasattr(emb, "embed_documents"):
        vecs = emb.embed_documents(txts)
    elif hasattr(emb, "embed_query"):
        vecs = [emb.embed_query(t) for t in txts]
    else:
        raise RuntimeError("Embeddings wrapper missing embed_documents/embed_query")
    out = []
    for v in vecs:
        # handle numpy arrays or lists
        out.append(v.tolist() if hasattr(v, "tolist") else list(v))
    return out

# --- create/get index (with ALREADY_EXISTS handling) ---
if NEW_PINECONE:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    # get list of existing indexes (defensive about return types)
    try:
        existing = list(pc.list_indexes())
    except Exception:
        try:
            existing = pc.list_indexes()
        except Exception:
            existing = []

    if INDEX_NAME not in existing:
        print(f"[store_index] Creating index {INDEX_NAME} ...")
        try:
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print(f"[store_index] create_index request submitted for {INDEX_NAME}.")
        except PineconeApiException as e:
            # ignore already-exists conflict; re-raise other errors
            txt = str(e)
            if "ALREADY_EXISTS" in txt or "already exists" in txt:
                print(f"[store_index] Index {INDEX_NAME} already exists (caught during create). Continuing.")
            else:
                raise
        except Exception as e:
            # unknown create error
            raise
    try:
        index = pc.Index(INDEX_NAME)
    except AttributeError:
        index = pc.index(INDEX_NAME)
else:
    # legacy client
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    try:
        existing = list(pinecone.list_indexes())
    except Exception:
        try:
            existing = pinecone.list_indexes()
        except Exception:
            existing = []
    if INDEX_NAME not in existing:
        print(f"[store_index] Creating index {INDEX_NAME} (legacy API) ...")
        try:
            pinecone.create_index(name=INDEX_NAME, dimension=DIM, metric="cosine")
        except PineconeApiException as e:
            txt = str(e)
            if "ALREADY_EXISTS" in txt or "already exists" in txt:
                print(f"[store_index] Index {INDEX_NAME} already exists (caught during create). Continuing.")
            else:
                raise
        except Exception:
            raise
    else:
        print(f"[store_index] Index {INDEX_NAME} already present (legacy).")
    index = pinecone.Index(INDEX_NAME)

print("[store_index] Index handle obtained ->", INDEX_NAME)

# --- upsert loop (robust signatures) ---
n = len(texts)
print(f"[store_index] Starting upsert in batches of {BATCH} ...")
for i in range(0, n, BATCH):
    batch_texts = texts[i : i + BATCH]
    ids = [str(uuid.uuid4()) for _ in batch_texts]
    vectors = embed_texts(batch_texts)

    items_tuple = [(_id, vec, {"text": txt}) for _id, vec, txt in zip(ids, vectors, batch_texts)]
    payload = [{"id": _id, "values": vec, "metadata": {"text": txt}} for _id, vec, txt in zip(ids, vectors, batch_texts)]
    simple_pairs = [{"id": _id, "values": vec} for _id, vec in zip(ids, vectors)]

    upserted = False
    # Try a few variants
    try:
        try:
            index.upsert(vectors=items_tuple)
            upserted = True
        except Exception:
            index.upsert(items=items_tuple)
            upserted = True
    except Exception:
        # try payload style
        try:
            index.upsert(vectors=payload)
            upserted = True
        except Exception:
            try:
                index.upsert(items=payload)
                upserted = True
            except Exception:
                try:
                    index.upsert(vectors=simple_pairs)
                    upserted = True
                except Exception as e:
                    raise RuntimeError(f"Upsert failed for batch {i}-{i+len(batch_texts)}: {e}") from e

    print(f"[store_index] Upserted batch {i//BATCH + 1} ({i}:{i+len(batch_texts)})")

print("[store_index] Upsert complete.")

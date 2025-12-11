# src/helper.py
"""
Robust helper for:
- loading PDFs from a directory
- splitting text into chunks
- providing a LangChain-compatible embeddings object (with fallback)
"""

import traceback

# ---- flexible imports for loaders and splitters (cover multiple langchain versions) ----
def _import_loader_and_splitter():
    loader_cls = None
    text_splitter_cls = None

    # Try langchain_community DirectoryLoader + PyPDFLoader (community package)
    try:
        from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
        loader_cls = (DirectoryLoader, PyPDFLoader)
    except Exception:
        try:
            # older/newer langchain may include community loaders differently
            from langchain.document_loaders import PyPDFLoader, DirectoryLoader
            loader_cls = (DirectoryLoader, PyPDFLoader)
        except Exception:
            loader_cls = None

    # Try text splitter locations
    try:
        # newest langchain
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter_cls = RecursiveCharacterTextSplitter
    except Exception:
        try:
            # fallback package some users install
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            text_splitter_cls = RecursiveCharacterTextSplitter
        except Exception:
            text_splitter_cls = None

    return loader_cls, text_splitter_cls


# ---- PDF loading ----
def load_pdf_file(directory_path: str):
    """
    Load all PDFs from directory_path (glob *.pdf) and return list of LangChain documents.
    """
    loader_cls, text_splitter_cls = _import_loader_and_splitter()
    if loader_cls is None:
        raise RuntimeError(
            "Could not import DirectoryLoader/PyPDFLoader. Install langchain_community or langchain document loaders.\n"
            "Try: python -m pip install langchain_community"
        )
    DirectoryLoader, PyPDFLoader = loader_cls
    loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


# ---- Text splitting ----
def text_split(extracted_documents, chunk_size: int = 500, chunk_overlap: int = 20):
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    Returns a list of documents (same format as LangChain expects).
    """
    _, text_splitter_cls = _import_loader_and_splitter()
    if text_splitter_cls is None:
        raise RuntimeError(
            "Could not import RecursiveCharacterTextSplitter. Install langchain_text_splitters or update langchain.\n"
            "Try: python -m pip install langchain_text_splitters"
        )
    splitter = text_splitter_cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(extracted_documents)
    return chunks


# ---- Robust embeddings loader (tries LangChain wrapper, else fallback) ----
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers short id

def _try_langchain_wrapper(model_name: str):
    """Try several known LangChain wrapper import paths and instantiate if available."""
    try_imports = [
        "langchain.embeddings.HuggingFaceEmbeddings",
        "langchain.embeddings.huggingface.HuggingFaceEmbeddings",
        "langchain_community.embeddings.huggingface.HuggingFaceEmbeddings",
        "langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings",
    ]
    for path in try_imports:
        module_path, cls_name = path.rsplit(".", 1)
        try:
            module = __import__(module_path, fromlist=[cls_name])
            cls = getattr(module, cls_name)
            # instantiate with CPU device to avoid optional dependencies
            return cls(model_name=model_name, model_kwargs={"device": "cpu"})
        except Exception:
            continue
    return None


class _SentenceTransformersWrapper:
    """Minimal wrapper matching LangChain embedder interface: embed_documents & embed_query"""
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        # Try to make sentence-transformers compatible with newer huggingface_hub
        # by aliasing cached_download -> hf_hub_download if necessary (safe, local shim).
        try:
            import huggingface_hub as _hf_hub
            if not hasattr(_hf_hub, "cached_download") and hasattr(_hf_hub, "hf_hub_download"):
                # alias the newer function to the old name expected by sentence-transformers
                _hf_hub.cached_download = _hf_hub.hf_hub_download
        except Exception:
            # ignore any errors here — sentence-transformers import will raise if it's not present
            pass

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise ImportError(
                "sentence-transformers is required for the fallback embeddings wrapper. "
                "Install with: python -m pip install sentence-transformers"
            ) from e


        # SentenceTransformer accepts short ids like "all-MiniLM-L6-v2"
        self.model = SentenceTransformer(model_name)
        try:
            self.model.to(device)
        except Exception:
            pass
        self.device = device

    def embed_documents(self, texts):
        # convert_to_numpy True gives numpy arrays
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True, device=self.device)
        except TypeError:
            embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [list(map(float, vec)) for vec in embeddings]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def download_hugging_face_embeddings(model_name: str = DEFAULT_MODEL):
    """
    Return an embeddings-like object with methods:
      - embed_documents(list[str]) -> list[list[float]]
      - embed_query(str) -> list[float]

    Preference order:
    1) LangChain HuggingFaceEmbeddings wrapper (multiple import paths attempted)
    2) Local sentence-transformers wrapper (minimal, reliable)
    """
    # 1) try LangChain wrappers first
    try:
        obj = _try_langchain_wrapper(model_name)
        if obj is not None:
            print("[helper] Using LangChain HuggingFaceEmbeddings wrapper.")
            return obj
    except Exception as e:
        print("[helper] LangChain wrapper attempt failed:", e)
        traceback.print_exc()

    # 2) fallback to sentence-transformers wrapper
    try:
        wrapper = _SentenceTransformersWrapper(model_name=model_name, device="cpu")
        print("[helper] Using local sentence-transformers wrapper (CPU).")
        return wrapper
    except Exception as e:
        tb = traceback.format_exc()
        raise RuntimeError(
            "Failed to create embeddings object. Install either:\n"
            "  - a LangChain version exposing HuggingFaceEmbeddings, OR\n"
            "  - sentence-transformers (pip install sentence-transformers)\n\n"
            f"Original error:\n{tb}"
        ) from e

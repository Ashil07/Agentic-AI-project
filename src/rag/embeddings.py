"""Step 4 of the pipeline: the Sentence Transformers embedding model.

The same model must be used to build the index and to embed questions,
so everyone gets it from ``get_embeddings()``.
"""

import os
import warnings
from functools import lru_cache

# Keep the console clean for the demo (no progress bars / load reports).
# Must be set before transformers is imported.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

from . import config  # noqa: E402


@lru_cache(maxsize=1)
def get_embeddings(model_name: str = config.EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """Return a cached embedding model (loaded only once per process).

    Vectors are L2-normalised, so the inner product FAISS computes is the
    cosine similarity: 1.0 means identical meaning, about 0 means unrelated.
    """
    try:  # hide the "Loading weights" progress bar of transformers >= 5
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
    except Exception:
        pass
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

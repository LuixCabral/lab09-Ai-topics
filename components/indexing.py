import os
import faiss
import numpy as np


def build_hnsw_index(
    embeddings: np.ndarray,
    index_path: str = "index.faiss",
) -> faiss.Index:
    """Build or load a FAISS HNSW index."""
    dim = embeddings.shape[1]
    
    # Normalize embeddings for cosine similarity (FAISS IndexHNSWFlat with Inner Product)
    # Cosine similarity is Inner Product of normalized vectors.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)

    if os.path.exists(index_path):
        print(f"Carregando índice de {index_path}...")
        index = faiss.read_index(index_path)
    else:
        print("Construindo novo índice FAISS HNSW...")
        # M = 16 (number of links per node)
        index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
        index.add(embeddings.astype('float32'))
        faiss.write_index(index, index_path)

    return index

import os
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer

from components.data import DOCS
from components.hyde import hyde_document
from components.indexing import build_hnsw_index

load_dotenv()

def run(
    query: str,
    bi_encoder: SentenceTransformer,
    cross_encoder: CrossEncoder,
    index: any
) -> None:
    hyde_text = hyde_document(query)
    hyde_vec = bi_encoder.encode([hyde_text], normalize_embeddings=True)
    
    # FAISS search expects float32
    distances, labels = index.search(hyde_vec.astype('float32'), k=10)

    print("\nTop-10 (bi-encoder + HNSW/FAISS):")
    for rank, (idx, score) in enumerate(zip(labels[0], distances[0]), start=1):
        print(f"{rank:02d}. score={score:.4f} | {DOCS[int(idx)]}")

    pairs = [[query, DOCS[int(i)]] for i in labels[0]]
    scores = cross_encoder.predict(pairs)

    reranked = sorted(zip(labels[0], scores), key=lambda x: x[1], reverse=True)

    print("\nTop-3 (cross-encoder):")
    for rank, (idx, score) in enumerate(reranked[:3], start=1):
        print(f"{rank:02d}. score={score:.4f} | {DOCS[int(idx)]}")


def main() -> None:
    print("Carregando modelos...")
    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("Preparando embeddings e índice...")
    doc_embeddings = bi_encoder.encode(DOCS, normalize_embeddings=True)
    index = build_hnsw_index(doc_embeddings)

    while True:
        query = input("\nDigite a query (ou 'sair'): ").strip()
        if query.lower() in ["sair", "exit", "quit"]:
            break
        if not query:
            query = "dor de cabeca latejante e luz incomodando"
            print(f"Usando query padrão: {query}")
        
        run(query, bi_encoder, cross_encoder, index)


if __name__ == "__main__":
    main()

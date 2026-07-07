import csv
from vector_db import VectorDB

def load_chunks_from_csv(path: str) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chunks.append({
                "id": row["id"],
                "text": row["text"],
                "source": row["source"],
                # "categorie" existe dans le CSV mais n'est pas utilisé
                # ici — tu peux l'ajouter aux metadata si tu veux t'en
                # servir plus tard (filtrage par catégorie par exemple).
            })
    return chunks


if __name__ == "__main__":
    chunks = load_chunks_from_csv("05_corpus_rag.csv")
    print(f"{len(chunks)} chunks chargés depuis le CSV.")

    db = VectorDB(chunks=chunks)

    # Test de retrieval
    question = "Quelle est la couleur du chat de Bob ?"
    results = db.retrieve(question, n=3)

    print(f"\nQuestion : {question}")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['source']}] {r['text']}")
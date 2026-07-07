
import os
from sentence_transformers import SentenceTransformer
import chromadb

from config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME


class VectorDB:
    def __init__(self, chunks: list[dict] = None):
        """
        chunks: liste de dicts {"id": ..., "text": ..., "source": ...}
        Si une base existe déjà sur disque -> on la recharge (chunks ignorés).
        Sinon, si chunks est fourni -> on crée la base.
        Sinon -> erreur explicite.
        """
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        existing_collections = [c.name for c in self.client.list_collections()]

        if COLLECTION_NAME in existing_collections:
            self._load_existing()
        elif chunks:
            self._create_new(chunks)
        else:
            raise ValueError(
                "Aucune base existante trouvée et aucun chunk fourni : "
                "impossible de démarrer. Fournis des chunks pour créer la base."
            )

    def _create_new(self, chunks: list[dict]):
        print(f"Création d'une nouvelle base avec le modèle {EMBEDDING_MODEL_NAME}...")

        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # On stocke le nom du modèle DANS les métadonnées de la collection.
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"embedding_model": EMBEDDING_MODEL_NAME},
        )

        texts = [c["text"] for c in chunks]
        ids = [c["id"] for c in chunks]
        sources = [{"source": c["source"]} for c in chunks]

        embeddings = self._encode(texts)

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=sources,
        )
        print(f"{len(chunks)} chunks indexés.")

    def _load_existing(self):
        print("Base existante détectée, rechargement...")

        self.collection = self.client.get_collection(name=COLLECTION_NAME)

        # On relit le modèle utilisé à la création, PAS celui de config.py
        model_name = self.collection.metadata["embedding_model"]
        print(f"Modèle d'embedding utilisé à l'indexation : {model_name}")

        self.model = SentenceTransformer(model_name)

    def _encode(self, texts: list[str]):
        return self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def retrieve(self, question: str, n: int = 3) -> list[dict]:
        query_embedding = self._encode([question])

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n,
        )

        chunks = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append({"text": doc, "source": meta["source"]})
        return chunks


# --- Modèle d'embedding (indexation + recherche) ---
EMBEDDING_MODEL_NAME = "distiluse-base-multilingual-cased-v2"

# --- Modèle LLM du RAG
LLM_MODEL_NAME = "llama-3.3-70b-versatile"

# --- Modèle de modération 
MODERATION_MODEL_NAME = "meta-llama/llama-guard-4-12b"  


# --- Chemins ---
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "mon_premier_rag"

# --- Prompts 
RAG_PROMPT_PATH = "prompts/rag_prompt.txt"
MODERATOR_PROMPT_PATH = "prompts/moderator_prompt.txt"

# --- Paramètres de retrieval ---
N_CHUNKS_RETRIEVED = 3


import os
from dotenv import load_dotenv
from groq import Groq

from config import LLM_MODEL_NAME, RAG_PROMPT_PATH, N_CHUNKS_RETRIEVED, LLM_TEMPERATURE
from vector_db import VectorDB


class RAG:
    def __init__(self, vector_db: VectorDB):
        load_dotenv()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.vector_db = vector_db

        with open(RAG_PROMPT_PATH, encoding="utf-8") as f:
            self.system_prompt_template = f.read()

    def _build_system_prompt(self, chunks: list[dict]) -> str:
        chunks_text = "\n".join(
            f"- {c['text']} (source: {c['source']})" for c in chunks
        )
        return self.system_prompt_template.replace("{{Chunks}}", chunks_text)

    def answer_question(self, question: str) -> str:
        chunks = self.vector_db.retrieve(question, n=N_CHUNKS_RETRIEVED)
        system_prompt = self._build_system_prompt(chunks)

        response = self.client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
             temperature=LLM_TEMPERATURE,
        )

        return response.choices[0].message.content
"""
title: Llama Index Groq Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Groq API.
requirements: llama-index, groq
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import os
import time
from pydantic import BaseModel
from groq import Groq


class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_GROQ_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_API_KEY: str

    def __init__(self):
        self.documents = None
        self.index = None

        self.valves = self.Valves(
            **{
                "LLAMAINDEX_GROQ_BASE_URL": os.getenv("LLAMAINDEX_GROQ_BASE_URL", "https://api.groq.com/openai/v1/models"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama-3.2-3b-preview"),
                "LLAMAINDEX_API_KEY": os.getenv("LLAMAINDEX_API_KEY", "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"),
            }
        )

        self.client = Groq(api_key=self.valves.LLAMAINDEX_API_KEY)

    async def on_startup(self):
        print(f"Starting pipelines on_startup {time.time()}")

        from llama_index.llms.openai import OpenAI
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

        # Configure OpenAI compatible settings
        Settings.llm = OpenAI(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            api_base=self.valves.LLAMAINDEX_GROQ_BASE_URL,
            api_key=self.valves.LLAMAINDEX_API_KEY,
        )

        Settings.embed_model = OpenAIEmbedding(
            model="llama-3.2-3b-preview",  # Replace with a compatible embedding model
            api_base=self.valves.LLAMAINDEX_GROQ_BASE_URL,
            api_key=self.valves.LLAMAINDEX_API_KEY,
        )

        print(f"Model Loaded {time.time()}")

        data_dir = "/app/backend/data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created missing data directory at {data_dir}")
        
        self.documents = SimpleDirectoryReader(data_dir).load_data()

        if not self.documents:
            print("No documents loaded. Using hardcoded data.")
            self.documents = [
                {"text": "Nirav"},
                {"text": "Aman"}
            ]

        self.index = VectorStoreIndex.from_documents(self.documents)
        print(f"Index created: {self.index is not None}")

    async def on_shutdown(self):
        print("Shutting down pipeline...")

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(messages)
        print(user_message)

        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message},
            ],
            model=self.valves.LLAMAINDEX_MODEL_NAME,
        )

        return chat_completion.choices[0].message.content

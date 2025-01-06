"""
title: Llama Index Ollama Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import os
import time
from pydantic import BaseModel

class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_OPENAI_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str

    def __init__(self):
        self.documents = None
        self.index = None

        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OPENAI_BASE_URL": os.getenv("LLAMAINDEX_OPENAI_BASE_URL", "https://ollama.rishika.chat"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama-3.2-3b-preview"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "llama-3.2-3b-preview"),
                "LLAMAINDEX_API_KEY":"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" 

            }
        )

    async def on_startup(self):
        print(f"Starting pipelines on_script{time.now()}")
        from llama_index.llms.openai import OpenAI
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

        Settings.embed_model = OpenAIEmbedding(
            model=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,  # Replace with a Groq-supported embedding model if available
            api_base=self.value.LLAMAINDEX_OPENAI_BASE_URL,  # Base URL for Groq
            api_key=self.valves.LLAMAINDEX_API_KEY,
        )
    
        # Configure OpenAI-compatible LLM Model
        Settings.llm = OpenAI(
            model=self.valves.LLAMAINDEX_MODEL_NAME,  # Use a Groq-compatible model like llama3.3-70b-versatile
            api_base=self.value.LLAMAINDEX_OPENAI_BASE_URL,
            api_key=self.valves.LLAMAINDEX_API_KEY,
        )
        print(f"Model Loaded {time.now()}")
        # This function is called when the server is started.
        global documents, index

        data_dir = "/app/backend/data"
        files = os.listdir(data_dir)

        print(f"Contents of {data_dir}:")
        for file in files:
            print(file)

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created missing data directory at {data_dir}")
            
        if not os.listdir(data_dir):
            print("Warning: Data directory is empty. Adding hardcoded documents.")
            
        self.documents = SimpleDirectoryReader(data_dir).load_data()
        
        if not self.documents:
            print("No documents loaded. Using hardcoded data.")
            self.documents = [
                {"text": "Nirav"},
                {"text": "Aman"}
            ]
        
        # Create the index from documents (whether loaded or hardcoded)
        self.index = VectorStoreIndex.from_documents(self.documents)
        print(f"Index created: {self.index is not None}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        print(messages)
        print(user_message)
        query_engine = self.index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)

        return response.response_gen
# sudo docker run -d -p 0.0.0.0:9099:9099 --add-host=host.docker.internal:host-gateway -v /app/backend/data:/app/backend/data --name pipelines --restart always ghcr.io/open-webui/pipelines:main

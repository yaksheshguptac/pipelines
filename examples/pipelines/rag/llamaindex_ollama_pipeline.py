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
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str

    def __init__(self):
        self.documents = None
        self.index = None

        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL":  "http://172.172.162.230:11434/",
                "LLAMAINDEX_MODEL_NAME": "llama3.2:3b",
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": "nomic-embed-text:latest",
            }
        )

    async def on_startup(self):
        print(f"Starting pipelines on_script{time.time()}")
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        print(f"Model Loaded {time.time()}")
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

        print(f"Message:- {messages}")
        print(f"User Message:- {user_message}")
        query_engine = self.index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)

        return response.response_gen
# sudo docker run -d -p 0.0.0.0:9099:9099 --add-host=host.docker.internal:host-gateway -v /app/backend/data:/app/backend/data --name pipelines --restart always ghcr.io/open-webui/pipelines:main

"""
title: Llama Index Ollama Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings.
requirements: llama-index, llama-index-llms-openai, llama-index-embeddings-openai
"""

from typing import List, Union, Generator, Iterator, Optional

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
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
            }
        )
    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet")
    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"outlet:{__name__}")
    async def on_startup(self):
        a="sk-proj-"
        b="RVKw6X9XuysMvIvPDljO1rLOSFpXe_gJvlW1F-ekIG20MRVo91jk-5jKBkmAPA0-3Cj-zZ_ZQIT3BlbkFJmzMX2lJuwQrLPxgfNBJ3zx5oGtG4cb-N0YVRO6F41MtM9Pbu5MTEvVlwIxABq_Jsw0f090nAsA"
        os.environ["OPENAI_API_KEY"] = a+b
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print(f"OPENAI_API_KEY: {api_key}")
        else:
            print("OPENAI_API_KEY is not set!")

        
        print(f"Starting pipelines on_script{time.time()}")
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI
        from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
        embed_model = OpenAIEmbedding(model="text-embedding-3-small",embed_batch_size=10)
        Settings.embed_model = embed_model
        Settings.llm = OpenAI(model="gpt-3.5-turbo-0125")

        print("LLM openAi")
        print(f"Model Loaded {time.time()}")
        # This function is called when the server is started.
        global documents, index

        self.documents = SimpleDirectoryReader("/home/azureuser/data").load_data()
        print(self.documents)
        self.index = VectorStoreIndex.from_documents(self.documents)
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

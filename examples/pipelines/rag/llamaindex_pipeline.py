"""
title: Llama Index Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library.
requirements: llama-index
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage


class Pipeline:
    def __init__(self):
        self.documents = None
        self.index = None

    async def on_startup(self):
        import os
        

        # Set the OpenAI API key
        a="sk-proj-"
        b="RVKw6X9XuysMvIvPDljO1rLOSFpXe_gJvlW1F-ekIG20MRVo91jk-5jKBkmAPA0-3Cj-zZ_ZQIT3BlbkFJmzMX2lJuwQrLPxgfNBJ3zx5oGtG4cb-N0YVRO6F41MtM9Pbu5MTEvVlwIxABq_Jsw0f090nAsA"
        os.environ["OPENAI_API_KEY"] = a+b
        
        # Check and print the API key
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print(f"OPENAI_API_KEY: {api_key}")
        else:
            print("OPENAI_API_KEY is not set!")
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

        self.documents = SimpleDirectoryReader("/app/backend/data").load_data()
        self.index = VectorStoreIndex.from_documents(self.documents)
        # This function is called when the server is started.
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

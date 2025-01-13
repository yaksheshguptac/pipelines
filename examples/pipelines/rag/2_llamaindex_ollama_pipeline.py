from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import os
import time
from llama_index.llms.groq import Groq
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class Pipeline:

    def __init__(self):
        self.documents = None
        self.index = None
        # self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.huggingface_embedder = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    async def on_startup(self):
        print(f"Starting pipelines on script {time.time()}")
        
        # Set the Groq model for LLM tasks (query generation, etc.)
        Settings.llm = Groq(model="llama3-70b-8192", api_key="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")

        # Set the Hugging Face embedding model for document embeddings
        Settings.embed_model = self.huggingface_embedder
        
        print(f"Models loaded: {time.time()}")
        
        # Load documents
        data_dir = "/app/backend/data"
        files = os.listdir(data_dir)
        
        print(f"Contents of {data_dir}:")
        for file in files:
            print(file)

        self.documents = SimpleDirectoryReader(data_dir).load_data()
        print(f"Documents loaded: {self.documents is not None}")

        # Create the index from documents (whether loaded or hardcoded)
        self.index = VectorStoreIndex.from_documents(self.documents)
        print(f"Index created: {self.index is not None}")
    
    async def on_shutdown(self):
        pass

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        print(messages)
        print(user_message)

        # Query using Groq LLM
        query_engine = self.index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)

        return response.response_gen

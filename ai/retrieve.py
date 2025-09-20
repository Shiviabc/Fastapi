import numpy as np
import json
from sentence_transformers import SentenceTransformer
import faiss
import os


class Retriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        DATA_DIR = "data"
        EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embed.npy")   # changed here
        JSON_DATA_PATH = os.path.join(DATA_DIR, "college_data.json")

        try:
            print("Retriever: Loading embedding model...")
            self.model = SentenceTransformer(model_name)

            print(f"Retriever: Loading pre-computed embeddings from {EMBEDDINGS_PATH} ...")
            self.embeddings = np.load(EMBEDDINGS_PATH)

            print(f"Retriever: Loading data from {JSON_DATA_PATH} ...")
            with open(JSON_DATA_PATH, "r") as f:
                self.college_data = json.load(f)

            print("Retriever: Creating FAISS index...")
            self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self.index.add(self.embeddings)
            print("Retriever initialized successfully.")

        except FileNotFoundError as e:
            print(f"Error initializing Retriever: {e}. Did you run embeddings.py first?")
            raise
        except Exception as e:
            print(f"Unexpected error initializing Retriever: {e}")
            raise

    def find_similar_colleges(self, query: str, top_k: int = 5) -> list:
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        results = [self.college_data[i] for i in indices[0]]
        return results

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from typing import Optional

# Constants from the processing script should match
INDEX_FILE = "vector_store/review_index.faiss"
PROCESSED_DATA_FILE = "vector_store/processed_data.parquet"
MODEL_NAME = 'all-MiniLM-L6-v2'

class SemanticSearchEngine:
    def __init__(self):
        """Loads all necessary artifacts for the search engine to work."""
        self.model = SentenceTransformer(MODEL_NAME)
        
        if not os.path.exists(INDEX_FILE):
            raise FileNotFoundError(
                f"FAISS index not found. Did you run `python src/data_processing.py` first?"
            )
        self.index = faiss.read_index(INDEX_FILE)
        self.data = pd.read_parquet(PROCESSED_DATA_FILE)
        print("Search engine initialized successfully.")

    def search(self, query: str, k: int = 5) -> Optional[pd.DataFrame]:
        """
        Takes a text query and returns the top k most relevant reviews.
        """
        if not query.strip():
            return None
            
        # 1. Embed the user's query
        query_embedding = self.model.encode([query])
        
        # 2. Search the FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        if len(indices[0]) == 0:
            return None
        
        # 3. Fetch the results from our dataframe
        results_df = self.data.loc[indices[0]].copy()
        
        # FAISS returns L2 distance, so we can convert it to a more intuitive similarity score
        # A simple way is 1 - distance, but let's just keep it simple and show the raw text.
        # This could be a TODO: Implement a better similarity score calculation.
        results_df['distance'] = distances[0]
        
        return results_df.sort_values('distance')
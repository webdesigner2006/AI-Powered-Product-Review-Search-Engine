import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import time

# --- Configuration Constants ---
DATA_DIR = "data"
VECTOR_STORE_DIR = "vector_store"
RAW_DATA_FILE = os.path.join(DATA_DIR, "reviews.csv")
INDEX_FILE = os.path.join(VECTOR_STORE_DIR, "review_index.faiss")
PROCESSED_DATA_FILE = os.path.join(VECTOR_STORE_DIR, "processed_data.parquet")

# Using 'all-MiniLM-L6-v2' because it's fast and lightweight for a demo app.
MODEL_NAME = 'all-MiniLM-L6-v2' 

def build_vector_store():
    """The main data engineering pipeline."""
    print("--- Starting Data Processing Pipeline ---")
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    # 1. Load Data
    try:
        df = pd.read_csv(RAW_DATA_FILE)
        print(f"[OK] Loaded {len(df)} reviews from '{RAW_DATA_FILE}'")
    except FileNotFoundError:
        print(f"[ERROR] Data file not found at '{RAW_DATA_FILE}'. Please add it.")
        return

    # 2. Clean Data
    df.dropna(subset=['review_text'], inplace=True)
    df = df[df['review_text'].str.len() > 10].reset_index(drop=True)
    if df.empty:
        print("[ERROR] No valid reviews found after cleaning.")
        return
    print(f"[OK] Cleaned data, {len(df)} reviews remaining.")

    # 3. Generate Embeddings
    print(f"Loading embedding model: '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("Generating embeddings for all reviews...")
    start_time = time.time()
    embeddings = model.encode(df['review_text'].tolist(), show_progress_bar=True)
    print(f"[OK] Embeddings generated in {time.time() - start_time:.2f} seconds.")

    # 4. Build FAISS Index
    embedding_dim = embeddings.shape[1]
    # Using IndexFlatL2 for basic Euclidean distance search. It's simple and effective.
    index = faiss.IndexFlatL2(embedding_dim) 
    # IndexIDMap lets us map the vector's position back to our original dataframe index. Crucial!
    index_with_ids = faiss.IndexIDMap(index)
    index_with_ids.add_with_ids(embeddings.astype('float32'), df.index.values)
    print(f"[OK] FAISS index built. Contains {index_with_ids.ntotal} vectors.")

    # 5. Save Artifacts
    print(f"Saving FAISS index to '{INDEX_FILE}'...")
    faiss.write_index(index_with_ids, INDEX_FILE)
    
    # Using Parquet for efficiency - faster reads and smaller file size than CSV.
    print(f"Saving processed data to '{PROCESSED_DATA_FILE}'...")
    df.to_parquet(PROCESSED_DATA_FILE)
    
    print("\n--- Pipeline Finished Successfully! ---")
    print("You're all set to run the Streamlit app.")

if __name__ == "__main__":
    build_vector_store()s
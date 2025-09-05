import streamlit as st
from src.search_engine import SemanticSearchEngine
import time

# --- Page Config ---
st.set_page_config(page_title="Review Search", page_icon="🔎", layout="wide")

# --- Load Engine ---
@st.cache_resource
def load_engine():
    """Load the search engine once and cache it."""
    try:
        engine = SemanticSearchEngine()
        return engine
    except FileNotFoundError as e:
        # Provide helpful error message if user hasn't run the setup script
        st.error(f"🚨 {e}")
        st.info("Please build the search index first by running this in your terminal:")
        st.code("python src/data_processing.py")
        return None

search_engine = load_engine()

# --- UI Layout ---
st.title("🔎 AI-Powered Product Review Search")
st.markdown(
    "Go beyond keywords. Find reviews by asking questions like "
    "`reviews that mention bad battery life` or `comfortable but with a bad microphone`."
)

st.sidebar.title("About")
st.sidebar.info(
    "This demo uses a `SentenceTransformer` model to turn product reviews and your search query into vectors. "
    "It then uses a **FAISS** index to find the most similar reviews in milliseconds."
)
st.sidebar.markdown("---")

if search_engine:
    search_query = st.text_input("Search reviews:", key="search_bar")
    
    if search_query:
        start_time = time.time()
        results_df = search_engine.search(query=search_query, k=10)
        duration = time.time() - start_time
        
        st.markdown("---")
        st.write(f"Found **{len(results_df) if results_df is not None else 0}** results in {duration:.2f} seconds.")
        
        if results_df is not None:
            for _, row in results_df.iterrows():
                with st.container(border=True):
                    st.markdown(f"**Product:** `{row['product_name']}`")
                    st.write(row['review_text'])
        else:
            st.warning("No matching reviews found.")
else:
    st.warning("Search engine could not be loaded. Please see the error message above.")
# AI-Powered-Product-Review-Search-Engine
This project builds an intelligent search engine for a large dataset of e-commerce product reviews. Instead of basic keyword matching, it uses semantic search, allowing users to find relevant reviews using natural language queries.The data engineering script pre-processes and indexes the entire dataset for highly efficient searching in the web app.
 # AI-Powered Product Review Search

[](https://www.google.com/search?q=https://your-streamlit-app-url.streamlit.app/)

This isn't your typical keyword search. This project uses AI to understand the meaning behind your words, allowing you to search a large database of product reviews using natural language. For example, a search for "bad battery" can find a review that says, "the charge barely lasts two hours."

-----

## Core Features

  - **Semantic Search**: Understands the intent behind your query to find the most relevant results.
  - **Extremely Fast**: Uses a pre-built FAISS vector index for near-instant search results.
  - **Efficient Data Pipeline**: A dedicated script processes and indexes data offline for maximum app performance.

-----

## Tech Stack

  - **Framework**: Streamlit
  - **Embeddings**: Sentence Transformers (Hugging Face)
  - **Vector Search**: FAISS
  - **Data Handling**: Pandas, PyArrow

-----

## Getting Started

This is a two-step process: first, you build the search index, and then you run the app.

1.  **Clone the repository and install dependencies:**
    ```bash
    git clone https://github.com/your-username/product_review_search.git
    cd product_review_search
    pip install -r requirements.txt
    ```
2.  **Build the Search Index (One-time step):**
    Run the data processing script. This reads your `data/reviews.csv` and creates the fast search index.
    ```bash
    python src/data_processing.py
    ```
3.  **Run the App:**
    Once the index is built, you can launch the application.
    ```bash
    streamlit run app.py
    ```

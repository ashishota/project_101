import streamlit as st
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Connect to PostgreSQL
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        dbname="healthcare_vector",
        user="postgres",
        password="ashis"
    )

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="Healthcare Q&A Finder")
st.title("🧠 Healthcare Question Answering")
st.write("Ask a health-related question and get instant answers from our database!")

query = st.text_input("🔍 Enter your question here")

if query:
    # Encode query
    query_embedding = model.encode([query])

    # Connect to PostgreSQL and fetch embeddings
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Perform vector search (using pgvector extension)
    cur.execute("SELECT id, question, answer, embedding FROM healthcare_embeddings")
    rows = cur.fetchall()

    similarities = []
    for row in rows:
        id, question, answer, embedding = row
        # Convert string representation of embedding to list of floats
        embedding = np.fromstring(embedding[1:-1], sep=',')
        similarity = cosine_similarity(query_embedding, [embedding])[0][0]
        similarities.append((id, question, answer, similarity))

    # Sort by similarity and get the top 3 results
    similarities = sorted(similarities, key=lambda x: x[3], reverse=True)
    top_k = 3
    top_results = similarities[:top_k]

    st.subheader("📋 Top Answers")
    for result in top_results:
        st.markdown(f"**Q:** {result[1]}")
        st.markdown(f"**A:** {result[2]}")
        st.markdown(f"**Similarity:** {result[3]:.4f}")
        st.markdown("---")

    cur.close()
    conn.close()

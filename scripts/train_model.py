# train_model.py

import pandas as pd
from sentence_transformers.SentenceTransformer import SentenceTransformer
from tqdm import tqdm
import pickle

# Load dataset
df = pd.read_csv('train.csv')

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and effective

# Combine question + answer for better context
combined_text = df['question'] + " " + df['answer']

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(combined_text.tolist(), show_progress_bar=True)

# Save embeddings and data
with open('embeddings.pkl', 'wb') as f:
    pickle.dump({'ids': df['id'].tolist(), 'questions': df['question'].tolist(),
                 'answers': df['answer'].tolist(), 'embeddings': embeddings}, f)

print("Embeddings saved to embeddings.pkl")

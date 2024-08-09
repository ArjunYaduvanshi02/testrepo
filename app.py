import streamlit as st
import torch
import pickle
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import io

# Load the saved Sentence Transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load saved embeddings
embeddings = torch.load('embeddings.pt')

# Load descriptions and images
with open('descriptions.pkl', 'rb') as f:
    descriptions = pickle.load(f)

with open('images.pkl', 'rb') as f:
    images = pickle.load(f)


# Function to find top N matches
def find_top_n_matches(prompt, embeddings, descriptions, images, top_n=5):
    prompt_embedding = embedding_model.encode(prompt, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(prompt_embedding, embeddings)[0]

    # Get the top N similarities
    top_results = similarities.topk(k=top_n)

    # Retrieve the indices of the top N results
    top_indices = top_results.indices.tolist()
    top_scores = top_results.values.tolist()

    # Get the top N descriptions, images, and their similarity scores
    top_descriptions = [descriptions[idx] for idx in top_indices]
    top_images = [images[idx] for idx in top_indices]

    return top_descriptions, top_images, top_scores


# Streamlit interface
st.title("Product Description Finder")

prompt = st.text_input("Enter your search prompt:")

if prompt:
    top_descriptions, top_images, top_scores = find_top_n_matches(prompt, embeddings, descriptions, images, top_n=5)

    for i in range(len(top_descriptions)):
        st.write(f"**Score:** {top_scores[i]:.4f}")
        st.write(f"**Description:** {top_descriptions[i]}")

        # Display the image
        image_path = top_images[i]
        try:
            with open(image_path, "rb") as img_file:
                st.image(img_file.read(), caption=f"Image {i + 1}")
        except Exception as e:
            st.write(image_path)


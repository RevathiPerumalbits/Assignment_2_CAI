import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Sample knowledge base
documents = [
    "The capital of France is Paris.",
    "The Eiffel Tower is a famous landmark in Paris.",
    "France is known for its wine and cheese.",
    "The Louvre Museum is located in Paris, France.",
    "French is the official language of France."
]

# Initialize models and FAISS index
@st.cache_resource
def initialize_rag():
    # Load retriever model
    retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate document embeddings
    doc_embeddings = retriever_model.encode(documents, convert_to_tensor=True).cpu().numpy()
    
    # Create FAISS index
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)
    
    # Load generator model
    generator = pipeline('text2text-generation', model='t5-small')
    
    return retriever_model, index, generator

# RAG pipeline function
def rag_pipeline(query, retriever_model, index, documents, generator, k=3):
    # Embed query
    query_embedding = retriever_model.encode([query], convert_to_tensor=True).cpu().numpy()
    
    # Retrieve top-k documents
    distances, indices = index.search(query_embedding, k)
    retrieved_docs = [documents[idx] for idx in indices[0]]
    
    # Create prompt with context
    context = " ".join(retrieved_docs)
    prompt = f"Question: {query} Context: {context}"
    
    # Generate response
    response = generator(prompt, max_length=50, num_return_sequences=1)
    return retrieved_docs, response[0]['generated_text']

# Streamlit app
st.title("RAG System with Streamlit")
st.write("Enter a query to retrieve relevant documents and generate an answer.")

# Initialize models
retriever_model, index, generator = initialize_rag()

# Input query
query = st.text_input("Your Query:", "What is the capital of France?")

# Button to trigger RAG
if st.button("Get Answer"):
    if query:
        with st.spinner("Retrieving and generating answer..."):
            # Run RAG pipeline
            retrieved_docs, answer = rag_pipeline(query, retriever_model, index, documents, generator)
            
            # Display results
            st.subheader("Retrieved Documents:")
            for i, doc in enumerate(retrieved_docs, 1):
                st.write(f"{i}. {doc}")
            
            st.subheader("Generated Answer:")
            st.write(answer)
    else:
        st.warning("Please enter a query.")

# Run instructions
st.markdown("---")
st.markdown("**How to Run Locally:**")
st.markdown("Save this script as `app.py` and run `streamlit run app.py` in your terminal.")
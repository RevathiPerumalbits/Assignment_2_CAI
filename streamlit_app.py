import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Sample knowledge base
documents = [
    "Consolidated Balance Sheet as of March 31, (Dollars in millions except equity share data) Note ASSETS Current assets Cash and cash equivalents 1,481 2,305 Current investments Trade receivables 3,094 2,995 Unbilled revenues 1,861 1,526 Prepayments and other current assets 1,336 1,133 Income tax assets Derivative financial instruments Total current assets 8,626 8,865 Non-current assets Property, plant and equipment 1,679 1,793 Right-of-use assets Goodwill Intangible assets Non-current investments 1,530 1,801 Unbilled revenues Deferred income tax assets Income tax assets Other non-current assets Total non-current assets 6,686 6,690 Total assets 15,312 15,555 LIABILITIES AND EQUITY Current liabilities Trade payables Lease liabilities Derivative financial instruments Current income tax liabilities Unearned revenues Employee benefit obligations Provisions Other current liabilities 2,403 2,170 Total current liabilities 4,769 4,433 Non-current liabilities Lease liabilities Deferred income tax liabilities Employee benefit obligations Other non-current liabilities Total liabilities 6,088 5,561 Equity Share capital – ₹5/- ($0.16) par value 4,800,000,000 (4,800,000,000) authorized equity shares, issued and outstanding 4,136,387,925 (4,193,012,929) equity shares fully paid up, net of 12,172,119 (13,725,712) treasury shares each as of March 31, 2023 (March 31, 2022), respectively Share premium Retained earnings 11,401 11,672 Cash flow hedge reserve – Other reserves 1,370 1,170 Capital redemption reserve Other components of equity (4,314) (3,588) Total equity attributable to equity holders of the company 9,172 9,941 Non-controlling interests Total equity 9,224 9,994 Total liabilities and equity 15,312",
    "Consolidated Statements of Comprehensive Income for the years ended March 31, (Dollars in millions except equity share and per equity share data) Note Revenues 18,212 16,311 13,561 Cost of sales 12,709 10,996 8,828 Gross profit 5,503 5,315 4,733 Operating expenses: Selling and marketing expenses Administrative expenses Total operating expenses 1,678 1,560 1,408 Operating profit 3,825 3,755 3,325 Other income, net Finance cost Profit before income taxes 4,125 4,036 3,596 Income tax expense 1,142 1,068 Net profit 2,983 2,968 2,623 Other comprehensive income Items that will not be reclassified subsequently to profit or loss: Remeasurements of the net defined benefit liability / asset, net (11) Equity instruments through other comprehensive income, net and (3) Items that will be reclassified subsequently to profit or loss: Fair valuation of investments, net and (30) (6) (14) Fair value changes on derivatives designated as cash flow hedge, net and (1) (1) Exchange differences on translation of foreign operations (697) (320) (728) (327) Total other comprehensive income/(loss), net of tax (727) (326) Total comprehensive income 2,256 2,642 2,979 Profit attributable to: Owners of the company 2,981 2,963 2,613 Non-controlling interests 2,983 2,968 2,623 Total comprehensive income attributable to: Owners of the company 2,254 2,637 2,968 Non-controlling interests 2,256 2,642 2,979 Earnings per equity share Basic (in $ per share) 0.70 Diluted (in $ per share) 0.70.",
    "Consolidated Balance Sheet as of March 31, (Dollars in millions except equity share data) Note ASSETS Current assets Cash and cash equivalents 1,773 1,481 Current investments 1,548 Trade receivables 3,620 3,094 Unbilled revenues 1,531 1,861 Prepayments and other current assets 1,473 1,336 Income tax assets Derivative financial instruments Total current assets 10,722 8,626 Non-current assets Property, plant and equipment 1,537 1,679 Right-of-use assets Goodwill Intangible assets Non-current investments 1,404 1,530 Unbilled revenues Deferred income tax assets Income tax assets Other non-current assets Total non-current assets 5,801 6,686 Total assets 16,523 15,312 LIABILITIES AND EQUITY Current liabilities Trade payables Lease liabilities Derivative financial instruments Current income tax liabilities Unearned revenues Employee benefit obligations Provisions Other current liabilities 2,099 2,403 Total current liabilities 4,651 4,769 Non-current liabilities Lease liabilities Deferred income tax liabilities Employee benefit obligations Other non-current liabilities Total liabilities 5,918 6,088 Equity Share capital – ₹5/- ($0.16) par value 4,800,000,000 (4,800,000,000) authorized equity shares, issued and outstanding 4,139,950,635 (4,136,387,925) equity shares fully paid up, net of 10,916,829 (12,172,119) treasury shares each as of March 31, 2024 (March 31, 2023), respectively Share premium Retained earnings 12,557 11,401 Cash flow hedge reserve – Other reserves 1,623 1,370 Capital redemption reserve Other components of equity (4,396) (4,314) Total equity attributable to equity holders of the company 10,559 9,172 Non-controlling interests Total equity 10,605 9,224 Total liabilities and equity 16,523.",
    "TConsolidated Statements of Comprehensive Income for the years ended March 31, (Dollars in millions except equity share and per equity share data) Note Revenues 18,562 18,212 16,311 Cost of sales 12,975 12,709 10,996 Gross profit 5,587 5,503 5,315 Operating expenses: Selling and marketing expenses Administrative expenses Total operating expenses 1,753 1,678 1,560 Operating profit 3,834 3,825 3,755 Other income, net Finance cost Profit before income taxes 4,346 4,125 4,036 Income tax expense 1,177 1,142 1,068 Net profit 3,169 2,983 2,968 Other comprehensive income Items that will not be reclassified subsequently to profit or loss: Remeasurements of the net defined benefit liability / asset, net (11) Equity instruments through other comprehensive income, net and (3) Items that will be reclassified subsequently to profit or loss: Fair valuation of investments, net and (30) (6) Fair value changes on derivatives designated as cash flow hedge, net and (1) (1) Exchange differences on translation of foreign operations (117) (697) (320) (99) (728) (327) Total other comprehensive income/(loss), net of tax (82) (727) (326) Total comprehensive income 3,087 2,256 2,642 Profit attributable to: Owners of the company 3,167 2,981 2,963 Non-controlling interests 3,169 2,983 2,968 Total comprehensive income attributable to: Owners of the company 3,086 2,254 2,637 Non-controlling interests 3,087 2,256 2,642 Earnings per equity share Basic (in $ per share) 0.71 Diluted (in $ per share) 0.71."
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
query = st.text_input("Your Query:", "What was the revenues in 2024?")

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
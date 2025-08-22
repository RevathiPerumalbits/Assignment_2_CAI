import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import numpy as np
import streamlit as st
import os
import sys
import re  # Ensure re is imported at the top
import time
from typing import List, Dict, Tuple
import pdfplumber
import pytesseract

from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
import json
import pickle
import faiss
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import nltk
from nltk.tokenize import word_tokenize
import json
nltk.download('punkt_tab')
nltk.download('stopwords')
# New imports for Google Colab

# --- Initial Setup ---
# Download necessary NLTK data
nltk.download('punkt', quiet=True)
# Sample knowledge base
documents = [
    "Consolidated Balance Sheet as of March 31, (Dollars in millions except equity share data) Note ASSETS Current assets Cash and cash equivalents 1,481 2,305 Current investments Trade receivables 3,094 2,995 Unbilled revenues 1,861 1,526 Prepayments and other current assets 1,336 1,133 Income tax assets Derivative financial instruments Total current assets 8,626 8,865 Non-current assets Property, plant and equipment 1,679 1,793 Right-of-use assets Goodwill Intangible assets Non-current investments 1,530 1,801 Unbilled revenues Deferred income tax assets Income tax assets Other non-current assets Total non-current assets 6,686 6,690 Total assets 15,312 15,555 LIABILITIES AND EQUITY Current liabilities Trade payables Lease liabilities Derivative financial instruments Current income tax liabilities Unearned revenues Employee benefit obligations Provisions Other current liabilities 2,403 2,170 Total current liabilities 4,769 4,433 Non-current liabilities Lease liabilities Deferred income tax liabilities Employee benefit obligations Other non-current liabilities Total liabilities 6,088 5,561 Equity Share capital – ₹5/- ($0.16) par value 4,800,000,000 (4,800,000,000) authorized equity shares, issued and outstanding 4,136,387,925 (4,193,012,929) equity shares fully paid up, net of 12,172,119 (13,725,712) treasury shares each as of March 31, 2023 (March 31, 2022), respectively Share premium Retained earnings 11,401 11,672 Cash flow hedge reserve – Other reserves 1,370 1,170 Capital redemption reserve Other components of equity (4,314) (3,588) Total equity attributable to equity holders of the company 9,172 9,941 Non-controlling interests Total equity 9,224 9,994 Total liabilities and equity 15,312",
    "Consolidated Statements of Comprehensive Income for the years ended March 31, (Dollars in millions except equity share and per equity share data) Note Revenues 18,212 16,311 13,561 Cost of sales 12,709 10,996 8,828 Gross profit 5,503 5,315 4,733 Operating expenses: Selling and marketing expenses Administrative expenses Total operating expenses 1,678 1,560 1,408 Operating profit 3,825 3,755 3,325 Other income, net Finance cost Profit before income taxes 4,125 4,036 3,596 Income tax expense 1,142 1,068 Net profit 2,983 2,968 2,623 Other comprehensive income Items that will not be reclassified subsequently to profit or loss: Remeasurements of the net defined benefit liability / asset, net (11) Equity instruments through other comprehensive income, net and (3) Items that will be reclassified subsequently to profit or loss: Fair valuation of investments, net and (30) (6) (14) Fair value changes on derivatives designated as cash flow hedge, net and (1) (1) Exchange differences on translation of foreign operations (697) (320) (728) (327) Total other comprehensive income/(loss), net of tax (727) (326) Total comprehensive income 2,256 2,642 2,979 Profit attributable to: Owners of the company 2,981 2,963 2,613 Non-controlling interests 2,983 2,968 2,623 Total comprehensive income attributable to: Owners of the company 2,254 2,637 2,968 Non-controlling interests 2,256 2,642 2,979 Earnings per equity share Basic (in $ per share) 0.70 Diluted (in $ per share) 0.70.",
    "Consolidated Balance Sheet as of March 31, (Dollars in millions except equity share data) Note ASSETS Current assets Cash and cash equivalents 1,773 1,481 Current investments 1,548 Trade receivables 3,620 3,094 Unbilled revenues 1,531 1,861 Prepayments and other current assets 1,473 1,336 Income tax assets Derivative financial instruments Total current assets 10,722 8,626 Non-current assets Property, plant and equipment 1,537 1,679 Right-of-use assets Goodwill Intangible assets Non-current investments 1,404 1,530 Unbilled revenues Deferred income tax assets Income tax assets Other non-current assets Total non-current assets 5,801 6,686 Total assets 16,523 15,312 LIABILITIES AND EQUITY Current liabilities Trade payables Lease liabilities Derivative financial instruments Current income tax liabilities Unearned revenues Employee benefit obligations Provisions Other current liabilities 2,099 2,403 Total current liabilities 4,651 4,769 Non-current liabilities Lease liabilities Deferred income tax liabilities Employee benefit obligations Other non-current liabilities Total liabilities 5,918 6,088 Equity Share capital – ₹5/- ($0.16) par value 4,800,000,000 (4,800,000,000) authorized equity shares, issued and outstanding 4,139,950,635 (4,136,387,925) equity shares fully paid up, net of 10,916,829 (12,172,119) treasury shares each as of March 31, 2024 (March 31, 2023), respectively Share premium Retained earnings 12,557 11,401 Cash flow hedge reserve – Other reserves 1,623 1,370 Capital redemption reserve Other components of equity (4,396) (4,314) Total equity attributable to equity holders of the company 10,559 9,172 Non-controlling interests Total equity 10,605 9,224 Total liabilities and equity 16,523.",
    "TConsolidated Statements of Comprehensive Income for the years ended March 31, (Dollars in millions except equity share and per equity share data) Note Revenues 18,562 18,212 16,311 Cost of sales 12,975 12,709 10,996 Gross profit 5,587 5,503 5,315 Operating expenses: Selling and marketing expenses Administrative expenses Total operating expenses 1,753 1,678 1,560 Operating profit 3,834 3,825 3,755 Other income, net Finance cost Profit before income taxes 4,346 4,125 4,036 Income tax expense 1,177 1,142 1,068 Net profit 3,169 2,983 2,968 Other comprehensive income Items that will not be reclassified subsequently to profit or loss: Remeasurements of the net defined benefit liability / asset, net (11) Equity instruments through other comprehensive income, net and (3) Items that will be reclassified subsequently to profit or loss: Fair valuation of investments, net and (30) (6) Fair value changes on derivatives designated as cash flow hedge, net and (1) (1) Exchange differences on translation of foreign operations (117) (697) (320) (99) (728) (327) Total other comprehensive income/(loss), net of tax (82) (727) (326) Total comprehensive income 3,087 2,256 2,642 Profit attributable to: Owners of the company 3,167 2,981 2,963 Non-controlling interests 3,169 2,983 2,968 Total comprehensive income attributable to: Owners of the company 3,086 2,254 2,637 Non-controlling interests 3,087 2,256 2,642 Earnings per equity share Basic (in $ per share) 0.71 Diluted (in $ per share) 0.71."
]
def pdf_to_text(pdf_path, ocr=False):
    """Extract text from a PDF, preserving the natural reading order."""
    logger.info(f"Extracting text from {pdf_path}...")
    full_text = ""
    try:
        if not ocr:
            # Using pdfplumber for better layout preservation
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text(layout=True, x_tolerance=2)
                    if page_text:
                        full_text += page_text + "\n"
            logger.info(f"Direct text extracted from {pdf_path} in reading order.")
        else:
            # OCR fallback
            images = convert_from_path(pdf_path)
            for i, img in enumerate(images):
                full_text += pytesseract.image_to_string(img) + "\n"
                logger.info(f"OCR processed page {i+1}/{len(images)}")

        return full_text.strip()

    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""


# ## 2. Text Cleaning (`_02_data_clean.py`)
# 
# Once the text is extracted, it often contains unwanted elements like page numbers, headers, footers, and excessive whitespace. This script defines a `clean_text` function that uses regular expressions (`re`) to remove this noise, making the text much cleaner for the subsequent processing steps.

# In[8]:


import re
import os
import logging

logging.basicConfig(level=logging.INFO, force=True, )
logger = logging.getLogger(__name__)

def clean_text(raw_text):
    """Clean raw text by removing page numbers, headers, footers, and extra whitespace."""
    logger.info("Text cleaning started...")
    try:
        # Remove page numbers
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', raw_text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

        # Remove common report headers/footers (add more specific patterns as needed)
        text = re.sub(r'Infosys Limited and subsidiaries', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Annual Report \d{4}', '', text, flags=re.IGNORECASE)

        # Remove multiple blank lines
        text = re.sub(r'\n\s*\n+', '\n', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        logger.info("Text cleaned successfully...")
        return text
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return raw_text


# ## 3. Document Segmentation (`_03_data_segment.py`)
# 
# Financial reports are long and contain many different sections. For our purpose, we're interested in specific statements. This script segments the cleaned text to isolate the **Income Statement** and the **Balance Sheet**. It uses powerful regular expressions to identify the start and end points of these sections. The script also orchestrates the full extract-clean-segment pipeline in its `main` function and includes a validation step.

# In[13]:


import re
import os
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from collections import Counter

import logging

# NEW: Imports and setup for Google Drive in Colab


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_number(value):
    """Normalize numbers by removing commas and handling thousands/millions."""
    try:
        value = str(value).replace(",", "").strip()
        # Convert millions (e.g., "1,661" to "1661000000" if in thousands)
        if "." not in value and len(value) < 6:
            return str(int(value) * 1_000_000)
        return value
    except ValueError:
        return value

def segment_report(text: str) -> dict:
    """
    Segments the cleaned text into logical sections like balance sheet and income statement.
    """
    logger.info("Segmenting report into logical sections...")
    sections = {}

    # Define start and end patterns for each section
    patterns = {
        'income_statement': re.compile(r'Consolidated Statements of Comprehensive Income for the years ended March 31, \(Dollars in millions.*?Diluted \(in \$ per share\).*?\d+\.\d+', re.IGNORECASE | re.DOTALL),
        'balance_sheet': re.compile(r'Consolidated Balance Sheet as of March 31, \(Dollars in millions.*?Total liabilities and equity.*?\d{1,3}(?:,\d{3})*', re.IGNORECASE | re.DOTALL)
    }

    for section_name, pattern in patterns.items():
        match = pattern.search(text)
        if match:
            sections[section_name] = match.group(0).strip()
            logger.info(f"Found and extracted '{section_name}'.")
        else:
            sections[section_name] = "Not found"
            logger.warning(f"Could not find '{section_name}'.")

    return sections

def clean_processed_sheet(text):
    lines = text.split("\n")
    cleaned_lines = []
    seen_headers = set()

    for line in lines:
        stripped = line.strip()
        # Remove repeated column headers
        if stripped in seen_headers:
            continue
        if re.match(r'^\s*(March\s+\d{1,2},\s+\d{4}|USD.*)$', stripped):
            seen_headers.add(stripped)

        # Remove "Note" column values (e.g., "2.11", "2.18", etc.)
        # Match patterns like "2.11", "2.11|", or "2.11 " followed by non-digits or end of line
        cleaned_line = re.sub(r'^\s*\d+\.\d+\s*(?:\||\s|$)', '', stripped)  # Note at line start
        cleaned_line = re.sub(r'\s+\d+\.\d+\s*(?:\||\s|$)', ' ', cleaned_line)  # Note within line
        cleaned_lines.append(cleaned_line.strip())

    return "\n".join(cleaned_lines)

def main():
    # MODIFIED: Use Google Drive paths
    
    files = [os.path.join( "data/raw/infosys_2023.pdf"),
             os.path.join( "data/raw/infosys_2024.pdf")]

    os.makedirs(os.path.join( "data/output"), exist_ok=True)
    os.makedirs(os.path.join( "data/clean"), exist_ok=True)
    os.makedirs(os.path.join( "data/segmented"), exist_ok=True)

    for file_path in files:
        year = os.path.basename(file_path).split("_")[1].split(".")[0]
        logger.info(f"Processing {file_path} for year {year}")

        # Direct text extraction
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"

        # OCR fallback if no text found
        if not text.strip():
            logger.warning(f"Direct extraction failed for {file_path}, trying OCR")
            text = pdf_to_text(file_path, ocr=True)

        if not text.strip():
            logger.error(f"Failed to extract text from {file_path}")
            continue

        # Save raw text
        raw_text_path = os.path.join( f"data/output/infosys_{year}_raw.txt")
        with open(raw_text_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Saved raw text to {raw_text_path}")

        # Clean text and save
        cleaned_text = clean_text(text)
        # Apply additional cleaning to remove Note column values
        cleaned_text_processed = clean_processed_sheet(cleaned_text)
        cleaned_text_path = os.path.join( f"data/clean/infosys_{year}_cleaned.txt")
        with open(cleaned_text_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text_processed)
        logger.info(f"Saved cleaned text to {cleaned_text_path}")

        # Step 3: Segment
        sections = segment_report(cleaned_text_processed)

        for name, content in sections.items():
            segmented_path = os.path.join( f"data/segmented/infosys_{year}_{name}.txt")
            with open(segmented_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Saved {name} to {segmented_path}")



if __name__ == "__main__":
    logger.info("Executing the segmentation module")
    main()

# ## 4. Data Validation (`_04_data_validate.py`)
# 
# Garbage in, garbage out. This validation script is a crucial quality control step. It checks the segmented text files to ensure they contain the correct headers (e.g., "Consolidated Balance Sheet") and key financial figures (e.g., "Total assets"). It also checks for the presence of *forbidden keywords* to ensure that text from other sections (like "CASH FLOWS") hasn't accidentally leaked into our segments. This helps guarantee the quality of the data before we proceed to more complex tasks.

# In[9]:


import re
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import re
import logging
import os



# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Google Drive Mounting ---
# This will prompt you for authorization when you run it in a Colab cell.


# --- Core Validation Functions (No changes needed here) ---

def normalize_number(value):
    """Normalize numbers by removing commas."""
    try:
        return str(value).replace(",", "").strip()
    except (ValueError, AttributeError):
        return str(value)



# ## 6. Data Chunking (`_06_data_create_chunks.py`)
# 
# To prepare our data for modern retrieval systems (like semantic search), we need to break it down into small, digestible pieces called "chunks." This script takes the financial statements and converts each line item into a self-contained sentence. For example, a row for 'Total assets' becomes a chunk like: `"For the year 2024, the total assets was $16,523 million."`. Each chunk is given a unique ID and enriched with metadata (like the source file, section, and year), which is crucial for filtering and context-aware retrieval. All chunks are saved into a single JSON file for easy access in the next step.

# In[15]:


import os
import logging
import re
from uuid import uuid4
import json

# New imports for Google Colab
#from google.colab import drive

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Google Drive Mounting ---
# This will prompt you for authorization when you run it in a Colab cell.
#drive.mount('/content/drive')

# --- Core Chunking Functions (No changes needed here) ---

def create_sentence_chunks(text_content, file_path):
    """Processes a block of financial text into meaningful, sentence-like chunks."""
    logger.info(f"Creating sentence chunks for {os.path.basename(file_path)}")

    file_name = os.path.basename(file_path)
    section = "balance_sheet" if "balance_sheet" in file_name.lower() else "income_statement"
    year_match = re.search(r'_(\d{4})_', file_name)
    if not year_match:
        return []
    main_year = int(year_match.group(1))
    # Create tuples of years to process for each line item
    years = (main_year, main_year - 1)

    generated_chunks = []
    # Normalize and clean the text block for easier parsing
    text_data = ' '.join(text_content.split())
    header_pattern = re.compile(r'.*?\(Dollars in millions.*?data\)\s*Note\s*', re.IGNORECASE)
    text_data = header_pattern.sub('', text_data)

    # Regex to find financial values
    val_pattern = r'[\d,.-]+|\([\d,.-]+\)'
    # Regex to split text by line items followed by two values
    delimiter_pattern = re.compile(f'\\s+({val_pattern})\\s+({val_pattern})\\s*')
    parts = delimiter_pattern.split(text_data)

    i = 0
    while i < len(parts):
        item_name = parts[i].strip().lower().rstrip('.-–: ')

        # Check if the next parts are valid numbers
        num_values = 0
        if (i + 1 < len(parts)) and re.fullmatch(val_pattern, parts[i+1].strip()):
            num_values = 1
            if (i + 2 < len(parts)) and re.fullmatch(val_pattern, parts[i+2].strip()):
                num_values = 2

        # If a valid item and values are found, create sentence chunks
        if item_name and num_values > 0:
            values = parts[i+1 : i+1+num_values]

            for j, year in enumerate(years):
                if j < len(values):
                    value_str = values[j]
                    clean_val = value_str.strip().replace(",", "")
                    sentence = f"For the year {year}, the {item_name} was ${clean_val} million."

                    # Append the chunk with its metadata
                    generated_chunks.append({
                        "id": str(uuid4()),
                        "text": sentence,
                        "metadata": {
                            "file_path": file_path,
                            "section": section,
                            "year": year,
                            "original_item": item_name
                        }
                    })
            i += (1 + num_values)
        else:
            i += 1

    logger.info(f"Generated {len(generated_chunks)} sentence chunks from {file_name}.")
    return generated_chunks

def main_chunker():
    # --- MODIFIED: Define base path for Google Drive ---
    #drive_base_path = "/content/drive/My Drive/"

    # --- MODIFIED: Create full input file paths for Google Drive ---
   
    input_files = [
        "data/segmented/infosys_2023_balance_sheet.txt",
       "data/segmented/infosys_2024_balance_sheet.txt",
        "data/segmented/infosys_2023_income_statement.txt",
        "data/segmented/infosys_2024_income_statement.txt"
    ]

   

    all_chunks = []
    for file_path in input_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            if text:
                chunks = create_sentence_chunks(text, file_path)
                all_chunks.extend(chunks)
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}. Skipping.")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")

    # --- MODIFIED: Define the full output path for the JSON file ---
    output_json_path = "data/chunks", "all_sentence_chunks.json"
    with open("data/chunks/all_sentence_chunks.json", 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=4)

    logger.info(f"✅ Successfully saved all {len(all_chunks)} chunks to {"data/chunks/all_sentence_chunks.json"}")

if __name__ == "__main__":
    main_chunker()

class RetrievalConfig:
    INITIAL_CANDIDATE_COUNT = 80
    BM25_TOP_MULTIPLIER = 2
    DENSE_WEIGHT = 0.5
    SPARSE_WEIGHT = 0.5
    FINAL_TOP_K = 8
    CTX_MAX_TOKENS = 900
    EMB_MODEL_NAME = "intfloat/e5-small-v2"
    GEN_MODEL_NAME = "distilgpt2"
    FAISS_INDEX_IS_INNER_PRODUCT = True

# =============================
# Utilities
# =============================
def load_chunks(file_path: str = "data/chunks/all_sentence_chunks.json") -> List[Dict]:

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        
        return []
    except Exception as e:
        
        return []

def load_faiss_index(index_path: str = "data/retrieval/faiss_index.bin") -> Tuple[faiss.Index, List[int]]:
    
    try:
        index = faiss.read_index(index_path)
        id_path = index_path.replace(".bin", "_ids.pkl")
        with open(id_path, 'rb') as f:
            chunk_ids = pickle.load(f)
        
        return index, chunk_ids
    except Exception as e:
        
        return None, None

def load_bm25_index(index_path: str = "data/retrieval/bm25_index.pkl") -> BM25Okapi:
    
    try:
        with open(index_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        return None
def preprocess_query(query: str) -> Tuple[str, List[str]]:
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(query.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_tokens), filtered_tokens

def _normalize_minmax(d: Dict[int, float]) -> Dict[int, float]:
    if not d:
        return d
    vals = list(d.values())
    vmin, vmax = min(vals), max(vals)
    if vmax - vmin < 1e-9:
        return {k: 1.0 for k in d}
    return {k: (v - vmin) / (vmax - vmin) for k, v in d.items()}

def _faiss_scores_to_similarity(distances: np.ndarray) -> np.ndarray:
    if RetrievalConfig.FAISS_INDEX_IS_INNER_PRODUCT:
        return distances
    return -distances
def hybrid_retrieval(query: str,
                     chunks: List[Dict],
                     faiss_index: faiss.Index,
                     bm25: BM25Okapi,
                     chunk_ids: List[int],
                     emb_model: SentenceTransformer) -> List[Dict]:

    processed_query, query_tokens = preprocess_query(query)
    q_emb = emb_model.encode([f"query: {processed_query}"], show_progress_bar=False, normalize_embeddings=True)[0]

    distances, indices = faiss_index.search(q_emb.reshape(1, -1), RetrievalConfig.INITIAL_CANDIDATE_COUNT)
    sim = _faiss_scores_to_similarity(distances)[0]
    dense_scores = {chunk_ids[i]: float(sim[j]) for j, i in enumerate(indices[0]) if i != -1}

    bm25_scores = bm25.get_scores(query_tokens)
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:RetrievalConfig.INITIAL_CANDIDATE_COUNT * RetrievalConfig.BM25_TOP_MULTIPLIER]
    sparse_scores = {chunk_ids[i]: float(bm25_scores[i]) for i in top_bm25_idx}

    dn = _normalize_minmax(dense_scores)
    sn = _normalize_minmax(sparse_scores)

    combined = {cid: RetrievalConfig.DENSE_WEIGHT * dn.get(cid, 0.0) +
                        RetrievalConfig.SPARSE_WEIGHT * sn.get(cid, 0.0)
                for cid in set(dn) | set(sn)}

    top_ids = sorted(combined, key=combined.get, reverse=True)
    seen_texts = set()
    candidate_chunks = []
    for cid in top_ids:
        chunk = next((c for c in chunks if c["id"] == cid), None)
        if chunk and chunk["text"].strip() not in seen_texts:
            seen_texts.add(chunk["text"].strip())
            candidate_chunks.append(chunk)
        if len(candidate_chunks) >= RetrievalConfig.FINAL_TOP_K:
            break
    return candidate_chunks
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
def rag_generate(query: str, retrieved_chunks: List[Dict], cfg: RetrievalConfig) -> str:

    if not retrieved_chunks:
        return "No relevant information was found to generate an answer."


    keyword_variants = [
        "total assets", "total asset", "assets total", "total liabilities",
        "total equity", "cash and cash equivalents", "revenues", "net profit",
        "income tax expense"
    ]
    tokenizer = AutoTokenizer.from_pretrained(cfg.GEN_MODEL_NAME)
    context_parts = []
    token_budget = cfg.CTX_MAX_TOKENS
    for ch in retrieved_chunks:
        t = ch.get("text", "").strip()
        tokens = tokenizer(t, return_tensors='pt')['input_ids'].shape[1]
        if tokens <= token_budget:
            context_parts.append(t)
            token_budget -= tokens
        if token_budget <= 0:
            break
    context = "\n\n".join(context_parts)

    
    prompt = (
        "You are a precise financial assistant. Answer ONLY using the exact words or numbers from the context.\n"
        "If the exact answer is not present, reply 'Not found'. Do not invent numbers.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
    truncated_prompt = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)

    try:
        generator = pipeline('text-generation', model=cfg.GEN_MODEL_NAME, device=-1)
        response = generator(
            truncated_prompt,
            max_new_tokens=120,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )[0]['generated_text']
        answer = response.replace(truncated_prompt, "").strip().split('\n')[0].strip()
        
        return answer
    except ValueError as ve:
        
        return "Failed to load the generative model."
    except RuntimeError as excp:
        
        return "An error occurred during answer generation."
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
def main():
    st.set_page_config(layout="wide")
    st.title("Infosys Financial RAG System 📈 (Hybrid Retrieval Only)")
    st.write("Ask questions about Infosys's financial statements. Hybrid retrieval = BM25 + Dense.")

    @st.cache_resource(show_spinner=False)
    def load_resources():
        try:
            emb_model = SentenceTransformer(RetrievalConfig.EMB_MODEL_NAME)
            chunks = load_chunks()
            faiss_index, chunk_ids = load_faiss_index()
            bm25 = load_bm25_index()
            if faiss_index is None or bm25 is None or not chunks:
                return None, None, None, None, None
            return emb_model, chunks, faiss_index, chunk_ids, bm25
        except Exception as e:
            st.error(f"Resource loading error: {e}")
            return None, None, None, None, None

    resources = load_resources()
    if not resources or len(resources) < 5:
        st.error("Failed to load one or more critical resources.")
        return

    emb_model, chunks, faiss_index, chunk_ids, bm25 = resources
    if not all([emb_model, chunks, faiss_index, chunk_ids, bm25]):
        st.error("Failed to load all resources.")
        return

    query = st.text_input("Your Query:", "What was the revenues in 2024?")

    if st.button("Submit Query"):
        if not query.strip():
            st.warning("Please enter a query.")
            return

        start = time.time()
        with st.spinner("Retrieving relevant documents..."):
            results = hybrid_retrieval(query, chunks, faiss_index, bm25, chunk_ids, emb_model)

        if not results:
            st.error("No relevant information found.")
            return

        with st.spinner("Generating answer..."):
            answer = rag_generate(query, results, RetrievalConfig)

        elapsed = time.time() - start

        st.subheader("Answer")
        st.markdown(f"**{answer}**")

        with st.expander("Show Retrieval Details"):
            st.write(f"**Response Time**: {elapsed:.2f} sec")
            st.write(f"**Merged Context Blocks**: {len(results)}")
            for i, ch in enumerate(results, 1):
                src = ch['metadata'].get('file_path', 'unknown')
                st.info(f"**[{i}] Source**: {src}\n\n**Text**: {ch['text'][:1200]}{'...' if len(ch['text'])>1200 else ''}")

if __name__ == "__main__":
    main()


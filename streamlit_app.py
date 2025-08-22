#!/usr/bin/env python
# coding: utf-8

# # Financial Document Analysis and Q&A Pipeline
# 
# This Jupyter Notebook consolidates a series of Python scripts into a cohesive pipeline for extracting, cleaning, segmenting, and analyzing financial data from PDF documents. The ultimate goal is to create a retrieval system capable of answering questions based on the document's content.
# 
# The pipeline is organized into the following logical steps:
# 1.  **Data Extraction**: Reading text from PDF files.
# 2.  **Text Cleaning**: Removing noise and standardizing the extracted text.
# 3.  **Document Segmentation**: Isolating specific financial statements (e.g., Balance Sheet, Income Statement).
# 4.  **Validation**: Ensuring the segmented sections are correct and complete.
# 5.  **Q&A Generation**: Creating question-answer pairs from the financial data (for fine-tuning or evaluation).
# 6.  **Chunking**: Breaking down the text into small, meaningful sentences for embedding.
# 7.  **Embedding & Indexing**: Converting text chunks into numerical vectors and building search indexes (FAISS and BM25).
# 8.  **Data Loading**: Utility functions to load the generated indexes and chunks for the final application.

# In[ ]:




# ## 1. Data Extraction (`_01_data_extract.py`)
# 
# This script is the first step in our pipeline. It's responsible for extracting raw text from PDF documents. It uses the `pdfplumber` library, which is excellent at preserving the layout and reading order of the text within a PDF. It also includes a fallback mechanism to use Optical Character Recognition (OCR) via `pytesseract` if a PDF contains images of text instead of selectable text.

# In[ ]:


import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
import os
import logging
import pdfplumber

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# --- Google Drive Mounting ---
# This will prompt you for authorization when you run it in a Colab cell.
#drive.mount('/content/drive')

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

def validate_key_figures(file_path, expected_values, section_header, forbidden_keywords):
    """Validate key figures, header, and section purity in a segmented file."""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Check section header
        if section_header.upper() not in text.upper():
            logger.warning(f"Missing header {section_header} in {file_path}")
            return False
        else:
            logger.info(f"Found header {section_header} in {file_path}")

        # Check key figures
        all_found = True
        for key, value in expected_values.items():
            normalized_value = normalize_number(value)
            # Try direct match, regex with key-value, and standalone value
            if (normalized_value in text or
                re.search(rf"{key}\s*[:=]?\s*{value}", text, re.IGNORECASE) or
                re.search(rf"\b{normalized_value}\b", text, re.IGNORECASE)):
                logger.info(f"Found {key}: {value} in {file_path}")
            else:
                logger.warning(f"Missing or incorrect {key}: {value} in {file_path}")
                all_found = False

        # Check for contamination
        for keyword in forbidden_keywords:
            if keyword.upper() in text.upper():
                logger.warning(f"Found forbidden {keyword} in {file_path}")
                all_found = False

        return all_found
    except Exception as e:
        logger.error(f"Error validating {file_path}: {e}")
        return False

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

def validate_steps():
    # MODIFIED: Use Google Drive paths
   

    # Define expected values and headers
    validation_config = [
        {
            "file": os.path.join( "data/segmented/infosys_2023_balance_sheet.txt"),
            "year": "2023",
            "section": "balance_sheet",
            "header": "Consolidated Balance Sheet",
            "expected_values": {
                "Total assets": "15,312",  # Approximate, in USD millions
                "Total equity": "9,224"
            }
        },
        {
            "file": os.path.join( "data/segmented/infosys_2023_income_statement.txt"),
            "year": "2023",
            "section": "income_statement",
            "header": "Consolidated Statements of Comprehensive Income",
            "expected_values": {
                "Revenues": "18,212",
                "Net profit": "2,983"
            }
        },
        {
            "file": os.path.join( "data/segmented/infosys_2024_balance_sheet.txt"),
            "year": "2024",
            "section": "balance_sheet",
            "header": "Consolidated Balance Sheet",
            "expected_values": {
                "Total assets": "16,523",  # From prior input, in USD millions
                "Total equity": "10,605"
            }
        },
        {
            "file": os.path.join( "data/segmented/infosys_2024_income_statement.txt"),
            "year": "2024",
            "section": "income_statement",
            "header": "Consolidated Statements of Comprehensive Income",
            "expected_values": {
                "Revenues": "18,562",
                "Net profit": "3,169"
            }
        }
    ]

    # Forbidden keywords to check for contamination
    forbidden_keywords = [
        "CASH FLOWS",
        "CHANGES IN EQUITY",
        "NOTES TO THE CONSOLIDATED"
    ]

    # Validate each file
    for config in validation_config:
        logger.info(f"Validating {config['section']} for {config['year']}")
        success = validate_key_figures(
            config["file"],
            config["expected_values"],
            config["header"],
            forbidden_keywords
        )
        if success:
            logger.info(f"Validation successful for {config['file']}")
        else:
            logger.warning(f"Validation failed for {config['file']}")

if __name__ == "__main__":
    logger.info("Executing the segmentation module")
    main()
    logger.info("validating the segmentation module")
    validate_steps()


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

def validate_key_figures(file_path, expected_values, section_header, forbidden_keywords):
    """Validate key figures, header, and section purity in a segmented file."""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Check section header
        if section_header.upper() not in text.upper():
            logger.warning(f"Missing header '{section_header}' in {file_path}")
            return False

        # Check key figures
        all_found = True
        for key, value in expected_values.items():
            normalized_value = normalize_number(value)
            # Use regex to find the key followed by the value anywhere in the text
            if re.search(rf"{re.escape(key)}.*?{re.escape(normalized_value)}", text, re.IGNORECASE | re.DOTALL):
                logger.info(f"Found '{key}: {value}' in {file_path}")
            else:
                logger.warning(f"Missing or incorrect '{key}: {value}' in {file_path}")
                all_found = False

        # Check for contamination from other sections
        for keyword in forbidden_keywords:
            if keyword.upper() in text.upper():
                logger.warning(f"Found forbidden keyword '{keyword}' in {file_path}")
                all_found = False

        return all_found
    except Exception as e:
        logger.error(f"Error validating {file_path}: {e}")
        return False

def main_validator():
    # --- MODIFIED: Define base path for Google Drive ---


    # --- MODIFIED: Use os.path.join to create full paths for Google Drive ---
    # The relative paths are now joined with your Drive's base path.
    validation_config = [
        {
            "file": "data/segmented/infosys_2023_balance_sheet.txt",
            "header": "Consolidated Balance Sheet",
            "expected_values": {"Total assets": "15,312", "Total equity": "9,224"}
        },
        {
            "file": "data/segmented/infosys_2023_income_statement.txt",
            "header": "Consolidated Statements of Comprehensive Income",
            "expected_values": {"Revenues": "18,212", "Net profit": "2,983"}
        },
        # You can add configurations for 2024 as well
        {
            "file":  "data/segmented/infosys_2024_balance_sheet.txt",
            "header": "Consolidated Balance Sheet",
            "expected_values": {"Total assets": "16,523", "Total equity": "10,605"}
        },
        {
            "file":  "data/segmented/infosys_2024_income_statement.txt",
            "header": "Consolidated Statements of Comprehensive Income",
            "expected_values": {"Revenues": "18,562", "Net profit": "3,169"}
        }
    ]

    forbidden_keywords = ["CASH FLOWS", "CHANGES IN EQUITY"]

    for config in validation_config:
        # The file path is now the full, correct path in Google Drive
        logger.info(f"--- Validating {config['file']} ---")
        success = validate_key_figures(
            config["file"],
            config["expected_values"],
            config["header"],
            forbidden_keywords
        )
        if success:
            logger.info(f"✅ Validation successful for {config['file']}")
        else:
            logger.warning(f"❌ Validation failed for {config['file']}")

if __name__ == "__main__":
    main_validator()


# ## 5. Q&A Generation (`_05_data_qa_generate.py`)
# 
# This script parses the validated, segmented financial statements to automatically generate question-and-answer pairs. For example, from a line item like `Revenues ... 18,562`, it creates a question: "What was the Revenues in 2024?" and an answer: "For the year 2024, the Revenues was $18,562 million." These Q&A pairs can be invaluable for fine-tuning a language model or for creating a test set to evaluate the final retrieval system's performance.

# In[14]:


import os
import re
import json
import logging

# NEW: Imports and setup for Google Drive in Colab


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_qa_from_text(text_content, file_path):
    """Generates Q&A pairs from a block of financial text."""
    qa_pairs = []
    year_match = re.search(r'_(\d{4})_', file_path)
    if not year_match:
        return []
    main_year = int(year_match.group(1))

    text_data = ' '.join(text_content.split())
    header_pattern = re.compile(r'.*?\(Dollars in millions.*?data\)\s*Note\s*', re.IGNORECASE)
    text_data = header_pattern.sub('', text_data)

    val_pattern = r'[\d,.-]+|\([\d,.-]+\)'
    delimiter_pattern_2_col = re.compile(f'\\s+({val_pattern})\\s+({val_pattern})\\s*')
    parts = delimiter_pattern_2_col.split(text_data)

    i = 0
    while i < len(parts):
        item_name = parts[i].strip().lower().rstrip('.--: ')

        num_values = 0
        if (i + 1 < len(parts)) and re.fullmatch(val_pattern, parts[i+1].strip()):
            num_values = 1
            if (i + 2 < len(parts)) and re.fullmatch(val_pattern, parts[i+2].strip()):
                num_values = 2

        if item_name and num_values > 0:
            values = parts[i+1 : i+1+num_values]
            if values[0]:
                question = f"What was the {item_name} in {main_year}?"
                answer = f"For the year {main_year}, the {item_name} was ${values[0]} million."
                qa_pairs.append({"question": question, "answer": answer})
            i += (1 + num_values)
        else:
            i += 1

    return qa_pairs

def main_qa_generator():
    # MODIFIED: Use Google Drive paths
  # Adjust if your folder structure is different
    input_files = [
        os.path.join("data/segmented/infosys_2023_balance_sheet.txt"),
        os.path.join("data/segmented/infosys_2024_balance_sheet.txt"),
        os.path.join("data/segmented/infosys_2023_income_statement.txt"),
        os.path.join("data/segmented/infosys_2024_income_statement.txt")
    ]

    all_qa_pairs = []
    for file_path in input_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            if content:
                qa_pairs = generate_qa_from_text(content, file_path)
                all_qa_pairs.extend(qa_pairs)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    output_path = os.path.join( "data/qa/financial_qa_pairs.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_qa_pairs, f, indent=4)

    logger.info(f"Successfully generated {len(all_qa_pairs)} Q&A pairs to {output_path}")

if __name__ == "__main__":
    main_qa_generator()


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
        os.path.join( "infosys_2023_balance_sheet.txt"),
        os.path.join( "infosys_2024_balance_sheet.txt"),
        os.path.join( "infosys_2023_income_statement.txt"),
        os.path.join( "infosys_2024_income_statement.txt")
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
    output_json_path = os.path.join("data/chunks", "all_sentence_chunks.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=4)

    logger.info(f"✅ Successfully saved all {len(all_chunks)} chunks to {output_json_path}")

if __name__ == "__main__":
    main_chunker()


# ## 7. Embedding and Indexing (`_07_data_create_embedding.py`)
# 
# This is where we build the core of our retrieval system. The script performs two main tasks:
# 
# 1.  **Embedding**: It loads the sentence chunks and uses a `SentenceTransformer` model (`intfloat/e5-small-v2`) to convert each chunk's text into a high-dimensional numerical vector (an embedding). These embeddings capture the semantic meaning of the text.
# 
# 2.  **Indexing**: It builds two different types of search indexes:
#     * **FAISS Index**: A library for efficient similarity search. We use it to create an index of our text embeddings, allowing us to quickly find the most semantically similar chunks to a user's query.
#     * **BM25 Index**: A classical keyword-based search algorithm. This index is great for matching specific terms and numbers, complementing the semantic search of FAISS.
# 
# Both indexes are saved to disk for later use.

# In[17]:


import os
import logging
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import nltk
from nltk.tokenize import word_tokenize
import json
nltk.download('punkt_tab')
# New imports for Google Colab

# --- Initial Setup ---
# Download necessary NLTK data
nltk.download('punkt', quiet=True)
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Google Drive Mounting ---
# This will prompt you for authorization when you run it in a Colab cell.
#drive.mount('/content/drive')

# --- Core Indexing Functions ---

def load_chunks_from_json(file_path):
    """Loads chunks from the consolidated JSON file."""
    logger.info(f"Reading chunks from {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: Chunks file not found at {file_path}.")
        return []
    except Exception as e:
        logger.error(f"Error reading chunks from {file_path}: {e}")
        return []

def embed_chunks(chunks, model_name="intfloat/e5-small-v2"):
    """Embeds chunks using a sentence transformer model."""
    logger.info(f"Embedding chunks with {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        # Prepend "passage: " as recommended by the e5 model documentation for documents
        texts = [f"passage: {chunk['text']}" for chunk in chunks]
        return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    except Exception as e:
        logger.error(f"Error embedding chunks: {e}")
        return np.array([])

def build_faiss_index(embeddings, chunk_ids, output_dir):
    """Builds and saves a FAISS index for semantic search."""
    logger.info("Building FAISS index...")
    os.makedirs(output_dir, exist_ok=True)
    dimension = embeddings.shape[1]
    # Using IndexFlatIP for cosine similarity with normalized embeddings
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))

    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
    with open(os.path.join(output_dir, "faiss_index_ids.pkl"), 'wb') as f:
        pickle.dump(chunk_ids, f)
    logger.info(f"Saved FAISS index with {index.ntotal} vectors to {output_dir}")

def build_bm25_index(chunks, output_dir):
    """Builds and saves a BM25 index for keyword search."""
    logger.info("Building BM25 index...")
    os.makedirs(output_dir, exist_ok=True)
    tokenized_chunks = [word_tokenize(chunk["text"].lower()) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    with open(os.path.join(output_dir, "bm25_index.pkl"), 'wb') as f:
        pickle.dump(bm25, f)
    logger.info(f"Saved BM25 index with {len(tokenized_chunks)} documents to {output_dir}")

def main_indexer():
    """Main function to run the full embedding and indexing pipeline."""
    logger.info("--- 🚀 Starting Embedding and Indexing Pipeline 🚀 ---")

    # --- MODIFIED: Define paths for Google Drive ---
    #drive_base_path = "/content/drive/My Drive/"
    

    # Step 1: Load the processed chunks
    chunks = load_chunks_from_json("data/chunks/all_sentence_chunks.json")
    if not chunks:
        logger.error("No chunks found. Please run the chunking script first. Exiting.")
        return

    # Step 2: Generate embeddings for the chunks
    embeddings = embed_chunks(chunks)
    if embeddings.size == 0:
        logger.error("No embeddings were generated. Exiting.")
        return

    # Step 3: Build and save the search indexes
    chunk_ids = [chunk["id"] for chunk in chunks]
    build_faiss_index(embeddings, chunk_ids, "data/retrieval")
    build_bm25_index(chunks, "data/retrieval")

    logger.info("--- ✅ Pipeline Finished Successfully ---")

if __name__ == "__main__":
    main_indexer()


# ## 8. Data Loading Utilities (`_08_data_load_data.py`)
# 
# Finally, this script provides a set of simple, reusable functions to load the artifacts we created in the previous steps. It contains functions to load:
# - The JSON file of sentence chunks.
# - The FAISS index and its corresponding chunk IDs.
# - The BM25 index.
# 
# These functions will be used by the final application (e.g., a chatbot or an API) to quickly load the necessary data into memory to perform searches and answer user questions.

# In[18]:


import logging
from typing import List, Dict, Tuple
import json
import pickle
import faiss
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_chunks(file_path: str = "data/chunks/all_sentence_chunks.json") -> List[Dict]:
    logger.info(f"Loading chunks from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}.")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return []

def load_faiss_index(index_path: str = "data/retrieval/faiss_index.bin") -> Tuple[faiss.Index, List[int]]:
    logger.info(f"Loading FAISS index from {index_path}")
    try:
        index = faiss.read_index(index_path)
        id_path = index_path.replace(".bin", "_ids.pkl")
        with open(id_path, 'rb') as f:
            chunk_ids = pickle.load(f)
        logger.info(f"Loaded FAISS index with {index.ntotal} vectors.")
        return index, chunk_ids
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        return None, None

def load_bm25_index(index_path: str = "data/retrieval/bm25_index.pkl") -> BM25Okapi:
    logger.info(f"Loading BM25 index from {index_path}")
    try:
        with open(index_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load BM25 index: {e}")
        return None

if __name__ == "__main__":
    logger.info("This module provides data loading functions. Example usage:")

    # Example of how to use the functions:
    # chunks = load_chunks()
    # if chunks:
    #     logger.info(f"Loaded {len(chunks)} chunks.")

    # faiss_index, faiss_ids = load_faiss_index()
    # if faiss_index:
    #     logger.info("FAISS index loaded.")

    # bm25_index = load_bm25_index()
    # if bm25_index:
    #     logger.info("BM25 index loaded.")


# # Financial Document Analysis and Q&A Pipeline

# ### **Hybrid Retrieval Pipeline**
# 
# For each user query, the following steps are performed:
# 1.  **Preprocess**: The query is cleaned, converted to lowercase, and stopwords are removed.
# 2.  **Generate Query Embedding**: A numerical vector representation of the query is created.
# 3.  **Retrieve Top-N Chunks**: The most relevant chunks are retrieved from the knowledge base using two methods:
#     * **Dense Retrieval**: Based on vector similarity (e.g., cosine similarity).
#     * **Sparse Retrieval**: Based on keyword matching using an algorithm like BM25.
# 4.  **Combine Results**: The results from both dense and sparse retrieval are combined, either by taking the union of the two sets or by using a weighted score fusion to rank the combined results.
# 
# ---
# 
# ### **Advanced RAG Technique (Select One)**
# 
# Based on your group number, you will implement one of the following advanced Retrieval-Augmented Generation (RAG) techniques:
# 
# | Remainder (Group Number mod 5) | Advanced Technique                | Description                                                                     | **Hybrid Search** | Combine BM25 keyword search with dense vector retrieval for a balance of recall and precision. |
# |                                
# ---
# 
# ### **Response Generation**
# 
# To generate the final answer, follow these steps:
# 1.  **Use a small, open-source generative model** (e.g., DistilGPT2, GPT-2 Small, or Llama-2 7B if available).
# 2.  **Concatenate the retrieved passages and the user query** to form the input prompt for the model.
# 3.  **Limit the total input tokens** to ensure the prompt fits within the model's context window.
# 
# ---
# 
# ### **Guardrail Implementation**
# 
# Implement one of the following guardrails to improve the reliability and safety of your system:
# 
# * **Input-side Guardrail**: Validate user queries to filter out irrelevant, inappropriate, or harmful inputs before they are processed.
# * **Output-side Guardrail**: Check the generated response to filter or flag any hallucinated (non-factual) or undesirable outputs before they are shown to the user.

# ## 8. RAG  Retrieval

# In[ ]:


import logging
import numpy as np
import streamlit as st
import os
import sys
import re  # Ensure re is imported at the top
import time
from typing import List, Dict, Tuple

from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
import json
import pickle
import faiss
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# NEW: Imports and setup for Google Drive in Colab

# Set the base directory in Google Drive (adjust if your project folder is different, e.g., '/content/drive/MyDrive/your_project_folder/')
#base_dir = '/content/drive/MyDrive/'
#os.chdir(base_dir)  # Change working directory to Google Drive base to handle relative paths

#sys.path.append(os.path.join(base_dir, os.path.dirname(os.path.dirname(os.path.abspath('__file__')))))  # Adjusted for potential __file__ issues; may need tweaking based on structure



# =============================
# Initial Setup
# =============================
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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

# =============================
# Hybrid Retrieval (Deduplicated)
# =============================

def hybrid_retrieval(query: str,
                     chunks: List[Dict],
                     faiss_index: faiss.Index,
                     bm25: BM25Okapi,
                     chunk_ids: List[int],
                     emb_model: SentenceTransformer) -> List[Dict]:
    logger.info(f"Hybrid retrieval for query: {query}")

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

    logger.info(f"Retrieved {len(candidate_chunks)} unique chunks for generation.")
    for i, chunk in enumerate(candidate_chunks, 1):
        logger.info(f"Chunk {i}: {chunk['text'][:200]}... (Source: {chunk['metadata'].get('file_path', 'unknown')})")
    return candidate_chunks

# =============================
# RAG Generation
# =============================

def rag_generate(query: str, retrieved_chunks: List[Dict], cfg: RetrievalConfig) -> str:
    logger.info("Generating answer from merged chunks...")
    if not retrieved_chunks:
        return "No relevant information was found to generate an answer."

    # Updated regex to match full financial numbers
    number_pattern = re.compile(
        r"\$?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:\s*(?:million|billion|mn|bn|m|b))\b",
        re.IGNORECASE
    )
    year_pattern = re.compile(r"\b(19|20)\d{2}\b")
    keyword_variants = [
        "total assets", "total asset", "assets total", "total liabilities",
        "total equity", "cash and cash equivalents", "revenues", "net profit",
        "income tax expense"
    ]

    def find_number_near_keyword(chunks, keywords, window_chars=200):
        # Prioritize chunks containing the query's main keyword
        query_keywords = [kw for kw in keywords if kw in query.lower()]
        for chunk in chunks:
            txt = chunk.get("text", "").lower()
            src = chunk.get("metadata", {}).get("file_path", "unknown")
            # Log chunk metadata for debugging
            logger.debug(f"Processing chunk: {txt[:200]}... (Source: {src})")
            # Check for query-specific keywords first
            for kw in query_keywords:
                idx = txt.find(kw)
                if idx != -1:
                    start = max(0, idx - 50)
                    end = min(len(txt), idx + window_chars)
                    window = txt[start:end]
                    matches = number_pattern.finditer(window)
                    for m in matches:
                        val = m.group(0).strip()
                        # Log all matches for debugging
                        logger.debug(f"Found potential number: {val} in window: {window[:100]}...")
                        # Check if the number is part of a year
                        if not year_pattern.search(txt[max(0, m.start() - 10):m.end() + 10]):
                            # Verify metadata consistency
                            if "2023" in query.lower() and "2023" in txt and "2024" in src:
                                logger.warning(f"Metadata mismatch: 2023 data in {src}")
                            logger.info(f"Extracted number: {val} from chunk: {txt[:200]}...")
                            return val, chunk
        # Fallback to any chunk with any keyword
        for chunk in chunks:
            txt = chunk.get("text", "").lower()
            src = chunk.get("metadata", {}).get("file_path", "unknown")
            logger.debug(f"Processing chunk: {txt[:200]}... (Source: {src})")
            for kw in keywords:
                idx = txt.find(kw)
                if idx != -1:
                    start = max(0, idx - 50)
                    end = min(len(txt), idx + window_chars)
                    window = txt[start:end]
                    matches = number_pattern.finditer(window)
                    for m in matches:
                        val = m.group(0).strip()
                        logger.debug(f"Found potential number: {val} in window: {window[:100]}...")
                        if not year_pattern.search(txt[max(0, m.start() - 10):m.end() + 10]):
                            if "2023" in query.lower() and "2023" in txt and "2024" in src:
                                logger.warning(f"Metadata mismatch: 2023 data in {src}")
                            logger.info(f"Extracted number: {val} from chunk: {txt[:200]}...")
                            return val, chunk
        return None, None

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

    numeric_query = bool(re.search(r"\b(19|20)\d{2}\b", query)) or any(w in query.lower() for w in ["assets", "revenue", "profit", "income", "liabilities", "cash"])
    if numeric_query:
        match_text, match_chunk = find_number_near_keyword(retrieved_chunks, keyword_variants)
        if match_text:
            src = match_chunk.get("metadata", {}).get("file_path", "unknown")
            logger.info(f"Numeric extraction success: '{match_text}' from {src}")
            return match_text

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
        logger.info(f"Generated answer: {answer}")
        return answer
    except ValueError as ve:
        logger.error(f"Model loading error: {ve}")
        return "Failed to load the generative model."
    except RuntimeError as excp:
        logger.error(f"Generation error: {excp}")
        return "An error occurred during answer generation."

# =============================
# Streamlit App
# =============================

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

    query = st.text_input("Enter your financial query:", "What were the total assets in 2023?")

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


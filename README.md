# PDF-Based Question-Answering System


This project provides a Python-based question-answering (QA) system that extracts text from a PDF, encodes it for similarity search, and retrieves relevant context to answer user questions using a language model. It combines several libraries to extract, embed, search, and interpret text from a specified PDF file, making it ideal for summarizing, creating overviews, or generating README documentation from the contents of PDF documents.

Features
Extracts and segments text from PDF files
Uses FAISS (Facebook AI Similarity Search) for efficient similarity-based context retrieval
Generates embeddings of PDF text sections with SentenceTransformer
Answers questions by retrieving relevant sections and passing them to a language model
Supports customization with different PDFs or models
Requirements
Python 3.7+
Libraries:
pdfplumber for PDF text extraction
sentence-transformers for text embeddings
faiss for similarity-based retrieval
langchain and langchain_ollama for handling LLMs (language models)
To install the required packages, run:pip install pdfplumber sentence-transformers faiss-cpu langchain langchain-ollama
Setup
Download a PDF file: Save the PDF file you want to use in the same directory as this script.
Update the PDF Path: Change "attention is all you need.pdf" in read_pdf_text() to your PDF's filename.
Choose a Model: This code uses the all-MiniLM-L6-v2 embedding model and llama3.2 language model. Ensure you have access to the necessary models.
Usage
The code includes three main functions:

read_pdf_text(pdf_path): Extracts and segments text from the specified PDF.
retrieve_relevant_context(question, index, text_sections, embedder, top_k=3): Finds relevant sections from the PDF based on the question's embedding.
answer_question(question): Uses the extracted context and language model to generate an answer to the user’s question.

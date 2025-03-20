# Agentic RAG Chatbot for Intelligent Document Processing

Welcome to the **Agentic RAG Chatbot**, a Retrieval-Augmented Generation (RAG) system designed to answer questions using a vector database of processed PDF documents. This project processes PDF files, chunks them, embeds the content, stores it in a Qdrant vector database, and provides a chat interface to query the data.

## Functionalities

- **PDF Processing**: Extracts text from PDF files and converts it to markdown format for downstream use.
- **Document Chunking**: Splits large documents into smaller, manageable chunks for efficient retrieval.
- **Embedding Generation**: Creates vector embeddings using OpenAI's `text-embedding-3-small` model to represent document content.
- **Vector Storage**: Stores embeddings and metadata in a Qdrant vector database for fast similarity search.
- **Intelligent Querying**: Classifies relevant documents and retrieves precise answers using a GPT-4o-mini model, limited to the vector database content.
- **Chat Interface**: Offers an interactive Gradio-based UI for querying and viewing responses with source attribution.

## Use Cases

- **Corporate Knowledge Base**: Query internal company documents (e.g., reports, manuals) to extract insights or answer employee questions without manual search.
- **Legal Research**: Analyze legal PDFs (e.g., contracts, case law) to quickly find relevant clauses or precedents.
- **Academic Research**: Process research papers or theses to summarize findings or answer specific questions based on the corpus.
- **Customer Support**: Enable a chatbot to respond to customer inquiries using product manuals or FAQs stored as PDFs.
- **Compliance Monitoring**: Search regulatory documents to ensure adherence or identify relevant guidelines efficiently.
- **Historical Analysis**: Extract information from archived PDFs (e.g., news articles, government records) for historical insights or trends.

## Tech Stack

- **Python**: Core programming language for the application.
- **LangChain**: Framework for building context-aware language model applications.
- **Qdrant**: Vector database for storing and querying embeddings.
- **Gradio**: Library for creating the interactive chat interface.
- **OpenAI**: Provides embedding (`text-embedding-3-small`) and language model (GPT-4o-mini) capabilities.
- **Docling**: Tool for converting PDFs into structured markdown content.

## Project Folder Structure
![image](https://github.com/user-attachments/assets/49d5ed74-74b8-4543-bfef-29015dd90a0a)

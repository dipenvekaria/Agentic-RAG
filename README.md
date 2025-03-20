# Agentic RAG Chatbot

Welcome to the **Agentic RAG Chatbot**, a Retrieval-Augmented Generation (RAG) system designed to answer questions using a vector database of processed PDF documents. Built with Python, LangChain, Qdrant, and Gradio, this project processes PDF files, chunks them, embeds the content, stores it in a Qdrant vector database, and provides a chat interface to query the data.

## Features

- **PDF Processing**: Extracts text from PDF files and converts it to markdown format.
- **Document Chunking**: Splits documents into manageable chunks for efficient retrieval.
- **Embedding Generation**: Uses OpenAI's `text-embedding-3-small` model to create vector embeddings.
- **Vector Storage**: Stores embeddings and metadata in a Qdrant vector database.
- **Intelligent Querying**: Classifies relevant documents and retrieves answers using a GPT-4o-mini model, constrained to the vector database content.
- **Chat Interface**: Provides an interactive Gradio-based chat UI for asking questions and viewing responses with source attribution.

## Prerequisites

- Python 3.9+
- Qdrant server running locally (default: `localhost:6333`)
- OpenAI API key (set in a `.env` file)
- Required Python packages (see [Installation](#installation))

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/agentic-rag-chatbot.git
   cd agentic-rag-chatbot

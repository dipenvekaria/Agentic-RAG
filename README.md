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
Set Up a Virtual Environment (optional but recommended)
bash

Collapse

Wrap

Copy
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies
bash

Collapse

Wrap

Copy
pip install -r requirements.txt
If no requirements.txt exists yet, install the following:
bash

Collapse

Wrap

Copy
pip install langchain langchain-openai qdrant-client gradio docling python-dotenv
Set Up Environment Variables Create a .env file in the root directory and add your OpenAI API key:
text

Collapse

Wrap

Copy
OPENAI_API_KEY=your-openai-api-key
Run Qdrant Server Ensure a Qdrant instance is running locally. You can use Docker:
bash

Collapse

Wrap

Copy
docker run -p 6333:6333 qdrant/qdrant
Usage
Prepare PDF Documents Place your PDF files in the documents/ directory.
Run the Application
bash

Collapse

Wrap

Copy
python main.py
The script will automatically index new or updated PDFs, chunk them, embed them, and store them in Qdrant.
Once indexing is complete, a Gradio chat interface will launch in your browser.
Ask Questions
Type a question (e.g., "What is discussed in the documents?") in the text box.
The chatbot will respond with an answer based solely on the processed documents, including a source reference (e.g., document1.pdf).
Clear Chat
Click the "Clear Chat" button to reset the conversation history.
Project Structure
text

Collapse

Wrap

Copy
agentic-rag-chatbot/
├── documents/              # Directory for input PDF files
├── convertedoc/            # Extracted JSON files from PDFs
├── chunked_docs/          # Chunked document dehydration
├── embedded_docs/         # Embedded chunk files with vectors
├── main.py                # Main script with workflows and Gradio UI
├── extraction.py          # PDF extraction logic (PdfExtractor)
├── chunking.py            # Document chunking logic (DocumentChunker)
├── embedding.py           # Embedding generation logic (DocumentEmbedder)
├── qdrant_storage.py      # Qdrant storage logic (QdrantStorage)
├── .env                   # Environment variables (e.g., API keys)
└── README.md              # This file
How It Works
Indexing Workflow:
Extract: Converts PDFs to JSON with markdown content.
Chunk: Splits markdown into chunks (default: 1000 chars, 200 overlap).
Embed: Generates embeddings using OpenAI's embedding model.
Store: Saves embeddings and metadata in Qdrant.
Query Workflow:
Classify: Identifies the most relevant document using metadata and GPT-4o-mini.
Query: Retrieves context from Qdrant and generates an answer using GPT-4o-mini.
Chat Interface: Gradio provides a user-friendly way to interact with the system.
Configuration
Chunking Parameters: Modify chunk_size and chunk_overlap in main.py (lines ~150-151) if needed.
Embedding Model: Change the model in embedding.py (default: text-embedding-3-small).
Qdrant Settings: Adjust host/port in qdrant_storage.py if using a remote server.
Limitations
Answers are limited to the content in the vector database; no external knowledge is used.
Requires a local Qdrant instance and OpenAI API access.
Assumes PDFs are text-based (scanned images may not work without OCR).
Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.
License
This project is licensed under the MIT License. See LICENSE for details.

Acknowledgments
Built with LangChain, Qdrant, and Gradio.
Powered by OpenAI's embedding and language models.

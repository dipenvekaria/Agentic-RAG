import os
import json
import uuid
import hashlib
from datetime import datetime
from typing import List, Dict, Any
import gradio as gr
import nltk
import spacy
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from qdrant_client import QdrantClient
from qdrant_client.http import models
import spacy
try:
    spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
load_dotenv()

# Ensure required models are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    spacy_nlp = spacy.load('en_core_web_sm')
except OSError:
    spacy.cli.download('en_core_web_sm')
    spacy_nlp = spacy.load('en_core_web_sm')


class PdfExtractor:
    def __init__(self):
        self.converter = DocumentConverter()

    def extract_pdfs(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for file in os.listdir(input_dir):
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(input_dir, file)
                result = self.converter.convert(file_path)
                metadata = {
                    "original_file_name": file,
                    "input_path": file_path,
                    "conversion_date": datetime.now().isoformat(),
                    "output_format": "json"
                }
                output_data = {
                    "metadata": metadata,
                    "content": {"markdown": result.document.export_to_markdown()}
                }
                output_file_name = os.path.splitext(file)[0] + ".json"
                output_path = os.path.join(output_dir, output_file_name)
                with open(output_path, "w", encoding="utf-8") as json_file:
                    json.dump(output_data, json_file, indent=4, ensure_ascii=False)
                print(f"Extracted {file} to {output_file_name}")
        return output_dir


class DocumentChunker:
    def __init__(self, strategy: str = "semantic", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.strategy = strategy.lower()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if self.strategy == "semantic":
            self.splitter = lambda text: self._semantic_split(text)
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

    def _semantic_split(self, text: str) -> List[str]:
        doc = spacy_nlp(text)
        chunks = []
        current_chunk = []
        current_length = 0
        for sent in doc.sents:
            if current_length + len(sent.text) > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [current_chunk[-1]] if self.chunk_overlap > 0 else []
                current_length = len(current_chunk[0]) if current_chunk else 0
            current_chunk.append(sent.text)
            current_length += len(sent.text)
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def chunk_document(self, input_dir: str, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        for file in os.listdir(input_dir):
            if file.lower().endswith(".json"):
                file_path = os.path.join(input_dir, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                markdown_content = data["content"]["markdown"]
                metadata = data["metadata"]
                chunks = self.splitter(markdown_content)
                chunked_data = []
                for i, chunk in enumerate(chunks):
                    chunk_info = {
                        "chunk_id": i,
                        "text": chunk,
                        "metadata": {
                            **metadata,
                            "chunk_start_index": i * (self.chunk_size - self.chunk_overlap),
                            "chunk_length": len(chunk)
                        }
                    }
                    chunked_data.append(chunk_info)
                output_file_name = os.path.splitext(file)[0] + "_chunked.json"
                output_path = os.path.join(output_dir, output_file_name)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(chunked_data, f, indent=4, ensure_ascii=False)
                print(f"Chunked {file}: {len(chunks)} chunks created")
        return output_dir


class DocumentEmbedder:
    def __init__(self, model="text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(model=model, openai_api_key=os.getenv("OPENAI_API_KEY"))

    def embed_chunks(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for file in os.listdir(input_dir):
            if file.lower().endswith("_chunked.json"):
                file_path = os.path.join(input_dir, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                embedded_chunks = []
                for chunk_data in chunks:
                    embedding = self.embeddings.embed_query(chunk_data["text"])
                    chunk_data["embedding"] = embedding
                    embedded_chunks.append(chunk_data)
                output_file_name = os.path.splitext(file)[0] + "_embedded.json"
                output_path = os.path.join(output_dir, output_file_name)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(embedded_chunks, f, indent=4, ensure_ascii=False)
                print(f"Embedded {file}: {len(embedded_chunks)} chunks processed")
        return output_dir


class QdrantStorage:
    def __init__(self, host="localhost", port=6333, collection_name="document_embeddings"):
        self.client = QdrantClient(host=host, port=port, timeout=60)
        self.collection_name = collection_name
        self._setup_collection()

    def _setup_collection(self):
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )

    def _generate_summary(self, text: str) -> str:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        summary = '. '.join(sentences[:2]) if len(sentences) >= 2 else text[:100]
        return summary if summary.endswith('.') else summary + '.'

    def store_embeddings(self, input_dir):
        points = []
        batch_size = 100
        for file in os.listdir(input_dir):
            if file.lower().endswith("_embedded.json"):
                file_path = os.path.join(input_dir, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    embedded_chunks = json.load(f)
                for chunk in embedded_chunks:
                    point_id = str(uuid.uuid4())
                    summary = self._generate_summary(chunk["text"])
                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector=chunk["embedding"],
                            payload={
                                "text": chunk["text"],
                                "source_file": file,
                                "chunk_id": chunk["chunk_id"],
                                "summary": summary,
                                **chunk["metadata"]
                            }
                        )
                    )
                print(f"Prepared {file} for storage: {len(embedded_chunks)} chunks")
        if points:
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(collection_name=self.collection_name, points=batch)
                print(f"Stored batch {i // batch_size + 1}: {len(batch)} points")
            print(f"Stored total of {len(points)} points in Qdrant collection '{self.collection_name}'")
        return self.collection_name


def load_processed_files(pdf_dir):
    config_path = os.path.join(pdf_dir, "processed_files.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {"processed": [], "to_process": []}


def save_processed_files(pdf_dir, processed_files):
    with open(os.path.join(pdf_dir, "processed_files.json"), "w") as f:
        json.dump(processed_files, f, indent=4)


@tool
def query_vector_db(query: str, relevant_file: str = None) -> str:
    """Query the Qdrant vector database to retrieve context and source based on the input query.

    Args:
        query (str): The query string to search for.
        relevant_file (str, optional): A file name to pass (not used for filtering). Defaults to None.

    Returns:
        str: A JSON string containing 'context' and 'source' keys with the retrieved information.
    """
    qdrant_client = QdrantClient(host="localhost", port=6333)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    query_embedding = embeddings.embed_query(query)
    query_response = qdrant_client.query_points(
        collection_name="document_embeddings",
        query=query_embedding,
        limit=5,
        with_payload=True
    )
    search_results = query_response.points
    if not search_results:
        return json.dumps({"context": "No relevant information found.", "source": "None"})
    context = "\n\n".join(
        [result.payload.get("summary", result.payload.get("text", "No summary available")) for result in
         search_results])
    source = search_results[0].payload.get("original_file_name", "Unknown")
    return json.dumps({"context": context, "source": source})


def process_documents(pdf_dir="documents"):
    processed_files = load_processed_files(pdf_dir)
    files_to_process = [f for f in os.listdir(pdf_dir) if
                        f.lower().endswith(".pdf") and f not in processed_files["processed"]]

    if not files_to_process and os.path.exists("embedded_docs") and os.listdir("embedded_docs"):
        print("No new PDFs to process and embeddings exist. Skipping indexing.")
        return {"pdf_dir": pdf_dir, "extracted_dir": "convertedoc", "chunked_dir": "chunked_docs",
                "embedded_dir": "embedded_docs", "collection_name": "document_embeddings"}

    # Extract PDFs
    extractor = PdfExtractor()
    extracted_dir = extractor.extract_pdfs(pdf_dir, "convertedoc")
    processed_files["processed"].extend(files_to_process)
    save_processed_files(pdf_dir, processed_files)

    # Chunk documents
    chunker = DocumentChunker()
    chunked_dir = chunker.chunk_document(extracted_dir, "chunked_docs")

    # Embed chunks
    embedder = DocumentEmbedder()
    embedded_dir = embedder.embed_chunks(chunked_dir, "embedded_docs")

    # Store in Qdrant
    qdrant_store = QdrantStorage()
    collection_name = qdrant_store.store_embeddings(embedded_dir)

    return {"pdf_dir": pdf_dir, "extracted_dir": extracted_dir, "chunked_dir": chunked_dir,
            "embedded_dir": embedded_dir, "collection_name": collection_name}


def classify_relevant_file(question, embedded_dir):
    metadata_list = []
    for f in os.listdir(embedded_dir):
        if f.endswith("_embedded.json"):
            with open(os.path.join(embedded_dir, f), "r") as ef:
                embedded_data = json.load(ef)
                if embedded_data and isinstance(embedded_data, list) and len(embedded_data) > 0:
                    file_name = embedded_data[0].get("metadata", {}).get("original_file_name",
                                                                         f.replace("_chunked_embedded.json", ".pdf"))
                    sample_content = embedded_data[0].get("summary",
                                                          embedded_data[0].get("text", "No summary available"))[:200]
                    metadata_list.append({"file_name": file_name, "sample_content": sample_content})

    if not metadata_list:
        return None

    classify_prompt = PromptTemplate(
        input_variables=["question", "metadata"],
        template="""Given the following question and metadata about available files, determine which file is most likely to contain the answer. Return only the file name without additional explanation.

Question: {question}
Metadata (file_name, sample_content):
{metadata}"""
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    metadata_str = "\n".join([f"{m['file_name']}: {m['sample_content']}" for m in metadata_list])
    chain = classify_prompt | llm
    return chain.invoke({"question": question, "metadata": metadata_str}).content.strip()


def query_documents(question, embedded_dir):
    relevant_file = classify_relevant_file(question, embedded_dir)
    system_prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "relevant_file"],
        template="""You are an intelligent assistant designed to answer questions based solely on a vector database of document embeddings. Use the 'query_vector_db' tool exactly once to retrieve relevant context from the Qdrant vector database. Pass the 'relevant_file' parameter to the tool explicitly, though it won’t filter the results. The tool will return a JSON string with 'context' and 'source' keys. Parse this JSON string and, based only on the 'context', provide a concise, accurate answer. Return your response as a single JSON object with 'answer' and 'source' keys. Do not include additional text or external knowledge.

Question: {input}
Agent Scratchpad: {agent_scratchpad}

Return your response as this exact JSON object:
```json
{{
  "answer": "Your concise answer here",
  "source": "filename"
}}
```"""
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [query_vector_db]
    agent = create_openai_tools_agent(llm, tools, system_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10)
    response = agent_executor.invoke({"input": question, "agent_scratchpad": "", "relevant_file": relevant_file})
    output = response["output"]
    try:
        tool_output = json.loads(output)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'```json\s*({.*?})\s*```', output, re.DOTALL)
        if json_match:
            tool_output = json.loads(json_match.group(1))
        else:
            raise ValueError("No valid JSON object found in agent output")
    return f"{tool_output['answer']} [Source: {tool_output['source']}]"


# Initial setup and processing (unchanged)
state = process_documents()

def chat_function(message, history):
    # History is now a list of dicts with 'role' and 'content'
    answer = query_documents(message, state["embedded_dir"])
    # Append new message and response to history
    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer}
    ]
    return new_history

# Gradio interface with updated chatbot type
with gr.Blocks(title="Agentic RAG Chatbot") as demo:
    gr.Markdown("# Agentic RAG Chatbot")
    gr.Markdown("Ask questions based on processed documents.")
    # Set type='messages' to use the new format
    chatbot = gr.Chatbot(label="Chat History", type="messages")
    msg = gr.Textbox(label="Your Question", placeholder="Type your question here")
    clear = gr.Button("Clear Chat")

    # Update inputs and outputs to work with the new format
    msg.submit(chat_function, inputs=[msg, chatbot], outputs=[chatbot])
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)  # Match Spaces default port
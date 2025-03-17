import os
import json
import hashlib
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from extraction import PdfExtractor
from chunking import DocumentChunker
from embedding import DocumentEmbedder
from qdrant_storage import QdrantStorage
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
import gradio as gr
import glob

load_dotenv()

class GraphState(TypedDict):
    pdf_dir: str
    extracted_dir: str
    chunked_dir: str
    embedded_dir: str
    collection_name: str
    question: str
    answer: str
    messages: Annotated[Sequence[AIMessage | HumanMessage], "A list of chat messages"]
    processed_files: dict
    chunk_config: dict
    relevant_file: str

def load_processed_files(pdf_dir):
    config_path = os.path.join(pdf_dir, "processed_files.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {"processed": [], "to_process": []}

def load_chunk_config(pdf_dir):
    config_path = os.path.join(pdf_dir, "chunk_config.json")
    default_config = {"chunk_size": 512, "chunk_overlap": 50}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return default_config

def save_configs(pdf_dir, processed_files, chunk_config):
    with open(os.path.join(pdf_dir, "processed_files.json"), "w") as f:
        json.dump(processed_files, f, indent=4)
    with open(os.path.join(pdf_dir, "chunk_config.json"), "w") as f:
        json.dump(chunk_config, f, indent=4)

@tool
def query_vector_db(query: str, relevant_file: str = None) -> str:
    """Query the Qdrant vector database to retrieve context and source based on the query and an optional relevant file filter."""
    print("Received relevant_file in query_vector_db:", relevant_file)
    qdrant_client = QdrantClient(host="localhost", port=6333)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    query_embedding = embeddings.embed_query(query)

    query_filter = None
    if relevant_file:
        query_filter = Filter(
            must=[FieldCondition(key="file_name", match=MatchValue(value=relevant_file))]
        )

    query_response = qdrant_client.query_points(
        collection_name="document_embeddings",
        query=query_embedding,
        query_filter=query_filter,
        limit=5,
        with_payload=True
    )
    search_results = query_response.points
    if not search_results:
        return json.dumps({
            "context": "No relevant information found in the vector database for this query.",
            "source": relevant_file if relevant_file else "None"
        })
    print("Query response points:", query_response.points)
    search_results = query_response.points
    print("Search results:", search_results)

    context = "\n\n".join(
        [result.payload.get("summary", result.payload.get("text", "No summary available")) for result in search_results]
    )
    source = relevant_file if relevant_file else "Unknown"

    return json.dumps({
        "context": context,
        "source": source
    })

def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def _generate_summary(text: str) -> str:
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    summary = '. '.join(sentences[:2]) if len(sentences) >= 2 else text[:100]
    return summary if summary.endswith('.') else summary + '.'

def extract_node(state: GraphState) -> GraphState:
    processed_files = state["processed_files"]
    files_to_process = [f for f in os.listdir(state["pdf_dir"])
                        if f.lower().endswith(".pdf") and f not in processed_files["processed"]]
    if not files_to_process:
        print("No new PDF files to extract. Skipping extraction.")
        state["extracted_dir"] = "convertedoc"
        return state
    extractor = PdfExtractor()
    state["extracted_dir"] = extractor.extract_pdfs(state["pdf_dir"], "convertedoc")
    processed_files["processed"].extend(files_to_process)
    processed_files["to_process"] = [f for f in processed_files["to_process"] if f not in files_to_process]
    save_configs(state["pdf_dir"], processed_files, state["chunk_config"])
    print(f"PDF extraction complete. Files saved in {state['extracted_dir']}")
    return state

def chunk_node(state: GraphState) -> GraphState:
    processed_files = state["processed_files"]
    chunk_config = state["chunk_config"]
    current_chunk_size = 512
    current_chunk_overlap = 50
    config_changed = (chunk_config.get("chunk_size") != current_chunk_size or
                      chunk_config.get("chunk_overlap") != current_chunk_overlap)
    files_to_chunk = []
    for f in os.listdir(state["extracted_dir"]):
        base_name = os.path.splitext(f)[0]
        pdf_name = f"{base_name}.pdf"
        chunked_file = os.path.join("chunked_docs", f"{base_name}_chunked.json")
        if (f.lower().endswith(".json") and
                pdf_name in processed_files["processed"] and
                (not os.path.exists(chunked_file) or config_changed)):
            files_to_chunk.append(f)
    if not files_to_chunk:
        print("No new files to chunk or chunking params unchanged. Skipping chunking.")
        state["chunked_dir"] = "chunked_docs"
        return state
    chunker = DocumentChunker(chunk_size=current_chunk_size, chunk_overlap=current_chunk_overlap)
    state["chunked_dir"] = chunker.chunk_document(state["extracted_dir"], "chunked_docs")
    state["chunk_config"] = {"chunk_size": current_chunk_size, "chunk_overlap": current_chunk_overlap}
    save_configs(state["pdf_dir"], processed_files, state["chunk_config"])
    print(f"Chunking complete with size={current_chunk_size}, overlap={current_chunk_overlap}. Files saved in {state['chunked_dir']}")
    return state

def embed_node(state: GraphState) -> GraphState:
    processed_files = state["processed_files"]
    chunk_config = state["chunk_config"]
    current_chunk_size = 1000
    current_chunk_overlap = 200
    config_changed = (chunk_config.get("chunk_size") != current_chunk_size or
                      chunk_config.get("chunk_overlap") != current_chunk_overlap)
    files_to_embed = []
    hash_file = os.path.join(state["embedded_dir"], "chunk_hashes.json")
    chunk_hashes = {}
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            chunk_hashes = json.load(f)

    for f in os.listdir(state["chunked_dir"]):
        if not f.lower().endswith("_chunked.json"):
            continue
        base_name = os.path.splitext(f)[0]
        embedded_file = os.path.join("embedded_docs", f"{base_name}_embedded.json")
        chunked_file = os.path.join(state["chunked_dir"], f)
        chunk_hash = get_file_hash(chunked_file)
        prev_hash = chunk_hashes.get(f)
        needs_embedding = (
                not os.path.exists(embedded_file) or
                config_changed or
                (prev_hash is None or prev_hash != chunk_hash)
        )
        if needs_embedding:
            files_to_embed.append(f)
            chunk_hashes[f] = chunk_hash

    if not files_to_embed:
        print("No new files to embed, chunking params unchanged, or embeddings up-to-date. Skipping embedding.")
        state["embedded_dir"] = "embedded_docs"
        return state

    embedder = DocumentEmbedder(model="text-embedding-3-small")
    state["embedded_dir"] = embedder.embed_chunks(state["chunked_dir"], "embedded_docs")

    for f in files_to_embed:
        base_name = os.path.splitext(f)[0]
        embedded_file = os.path.join(state["embedded_dir"], f"{base_name}_embedded.json")
        chunked_file = os.path.join(state["chunked_dir"], f)
        if os.path.exists(embedded_file) and os.path.exists(chunked_file):
            try:
                with open(embedded_file, "r") as ef:
                    embedded_data = json.load(ef)
                with open(chunked_file, "r") as cf:
                    chunked_data = json.load(cf)
                original_metadata = chunked_data.get("metadata", {})
                for chunk in embedded_data:
                    if "metadata" not in chunk:
                        chunk["metadata"] = {}
                    chunk["metadata"]["original_file_name"] = original_metadata.get("original_file_name",
                                                                                    f.replace("_chunked.json", ".pdf"))
                    chunk["metadata"]["file_name"] = f.replace("_chunked.json", ".pdf")
                    chunk["summary"] = _generate_summary(chunk["text"])
                with open(embedded_file, "w") as ef:
                    json.dump(embedded_data, ef, indent=4)
                print(f"Updated metadata and summary for {embedded_file}")
            except Exception as e:
                print(f"Error updating metadata for {embedded_file}: {e}")
        else:
            print(f"Embedded or chunked file not found: {embedded_file} or {chunked_file}. Skipping metadata update.")

    with open(hash_file, "w") as f:
        json.dump(chunk_hashes, f, indent=4)
    print(f"Embedding complete. Files saved in {state['embedded_dir']}")
    return state

def store_node(state: GraphState) -> GraphState:
    processed_files = state["processed_files"]
    files_to_store = [f for f in os.listdir(state["embedded_dir"])
                      if f.lower().endswith("_embedded.json")]
    qdrant_client = QdrantClient(host="localhost", port=6333)
    collection_exists = qdrant_client.collection_exists("document_embeddings")
    if collection_exists:
        collection_info = qdrant_client.get_collection("document_embeddings")
        stored_points = collection_info.points_count
        expected_points = sum(len(json.load(open(os.path.join(state["embedded_dir"], f), "r")))
                              for f in files_to_store)
        if stored_points >= expected_points and expected_points > 0:
            print("Collection exists with sufficient points. Skipping storage.")
            state["collection_name"] = "document_embeddings"
            return state
    qdrant_store = QdrantStorage(host="localhost", port=6333, collection_name="document_embeddings")
    state["collection_name"] = qdrant_store.store_embeddings(state["embedded_dir"])
    print(f"Storage complete. Embeddings saved in Qdrant collection: {state['collection_name']}")
    return state

def classify_node(state: GraphState) -> GraphState:
    if not state["question"]:
        return state

    metadata_list = []
    for f in os.listdir(state["embedded_dir"]):
        if f.endswith("_embedded.json"):
            with open(os.path.join(state["embedded_dir"], f), "r") as ef:
                embedded_data = json.load(ef)
                if embedded_data:
                    file_name = embedded_data[0].get("metadata", {}).get("file_name", os.path.splitext(f)[0] + ".pdf")
                    sample_content = embedded_data[0].get("summary",
                                                          embedded_data[0].get("text", "No summary available"))[:200]
                    metadata_list.append({
                        "file_name": file_name,
                        "sample_content": sample_content
                    })

    if not metadata_list:
        print("No metadata available for classification.")
        return state

    classify_prompt = PromptTemplate(
        input_variables=["question", "metadata"],
        template="""Given the following question and metadata about available files, determine which file is most likely to contain the answer. Return only the file name (e.g., 'document1.pdf') without additional explanation.

Question: {question}
Metadata (file_name, sample_content):
{metadata}"""
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    metadata_str = "\n".join([f"{m['file_name']}: {m['sample_content']}" for m in metadata_list])
    chain = classify_prompt | llm
    relevant_file = chain.invoke({"question": state["question"], "metadata": metadata_str}).content.strip()
    state["relevant_file"] = relevant_file
    print("Relevant file in query_node:", state["relevant_file"])
    source = relevant_file if relevant_file else "No relevant file provided"
    print("Assigned source:", source)
    print(f"Classified relevant file: {relevant_file}")
    return state

def query_node(state: GraphState) -> GraphState:
    if not state["question"]:
        return state
    system_prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "relevant_file"],
        template="""You are an intelligent assistant designed to answer questions based solely on a vector database of document embeddings. Use the 'query_vector_db' tool exactly once to retrieve relevant context (summaries of chunks) from the Qdrant vector database, filtered by the relevant file '{relevant_file}' if provided. Pass the 'relevant_file' parameter to the tool explicitly. The tool will return a JSON string with 'context' and 'source' keys. Parse this JSON string and, based only on the 'context', provide a concise, accurate answer. Return your response as a single JSON object with 'answer' and 'source' keys. Do not include the raw tool output, any additional text, or debugging information outside the JSON object. Do not add external knowledge or assumptions. If the context does not explicitly relate to the query subject, use the context as the answer.

Question: {input}
Agent Scratchpad: {agent_scratchpad}

Return your response as this exact JSON object and nothing else:
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
    response = agent_executor.invoke({
        "input": state["question"],
        "agent_scratchpad": "",
        "relevant_file": state.get("relevant_file", None)
    })
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
    answer = tool_output["answer"]
    source = tool_output["source"]
    state["answer"] = f"{answer} [Source: {source}]"
    state["messages"].append(HumanMessage(content=state["question"]))
    state["messages"].append(AIMessage(content=state["answer"]))
    return state

# Indexing Workflow
indexing_workflow = StateGraph(GraphState)
indexing_workflow.add_node("extract", extract_node)
indexing_workflow.add_node("chunk", chunk_node)
indexing_workflow.add_node("embed", embed_node)
indexing_workflow.add_node("store", store_node)
indexing_workflow.add_edge("extract", "chunk")
indexing_workflow.add_edge("chunk", "embed")
indexing_workflow.add_edge("embed", "store")
indexing_workflow.set_entry_point("extract")
indexing_workflow.set_finish_point("store")
indexing_app = indexing_workflow.compile()

# Query Workflow
query_workflow = StateGraph(GraphState)
query_workflow.add_node("classify", classify_node)
query_workflow.add_node("query", query_node)
query_workflow.add_edge("classify", "query")
query_workflow.set_entry_point("classify")
query_workflow.set_finish_point("query")
query_app = query_workflow.compile()

def needs_indexing(pdf_dir, processed_files, chunk_config):
    files_to_process = [f for f in os.listdir(pdf_dir) if
                        f.lower().endswith(".pdf") and f not in processed_files["processed"]]
    if files_to_process:
        print("New PDF files detected.")
        return True
    current_chunk_size = 1000
    current_chunk_overlap = 200
    if (chunk_config.get("chunk_size") != current_chunk_size or
            chunk_config.get("chunk_overlap") != current_chunk_overlap):
        print("Chunk configuration changed.")
        return True
    for f in os.listdir("convertedoc"):
        base_name = os.path.splitext(f)[0]
        pdf_name = f"{base_name}.pdf"
        chunked_file = os.path.join("chunked_docs", f"{base_name}_chunked.json")
        if (pdf_name in processed_files["processed"] and not os.path.exists(chunked_file)):
            print("Missing chunk file detected.")
            return True
    return False

# Initial setup
pdf_dir = "documents"
processed_files = load_processed_files(pdf_dir)
chunk_config = load_chunk_config(pdf_dir)
initial_state = {
    "pdf_dir": pdf_dir,
    "extracted_dir": "convertedoc",
    "chunked_dir": "chunked_docs",
    "embedded_dir": "embedded_docs",
    "collection_name": "document_embeddings",
    "question": "",
    "answer": "",
    "messages": [],
    "processed_files": processed_files,
    "chunk_config": chunk_config,
    "relevant_file": ""
}

# Run indexing if needed
if needs_indexing(pdf_dir, processed_files, chunk_config):
    print("Changes detected. Running indexing pipeline...")
    initial_state = indexing_app.invoke(initial_state)
    print("Indexing complete.")
else:
    print("No changes detected in PDFs or chunking config. Running embedding check...")
    initial_state = embed_node(initial_state)
    initial_state = store_node(initial_state)
    print("Embedding and storage check complete.")

# Gradio chat function
def chat_function(message, history):
    state = initial_state.copy()
    state["question"] = message
    state["messages"] = []
    for entry in history:
        if entry["role"] == "user":
            state["messages"].append(HumanMessage(content=entry["content"]))
        elif entry["role"] == "assistant":
            state["messages"].append(AIMessage(content=entry["content"]))
    state = query_app.invoke(state)
    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": state["answer"]}
    ]
    return new_history

# Gradio interface
with gr.Blocks(title="Uber RAG Chatbot") as demo:
    gr.Markdown("# Uber RAG Chatbot")
    gr.Markdown("Ask questions about Uber based on processed documents. Only data from the vector database is used.")
    chatbot = gr.Chatbot(label="Chat History", type="messages")
    msg = gr.Textbox(label="Your Question", placeholder="Type your question here (e.g., 'What did Uber do?')")
    clear = gr.Button("Clear Chat")

    msg.submit(chat_function, inputs=[msg, chatbot], outputs=[chatbot])
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()
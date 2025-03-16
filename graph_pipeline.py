import os
import json
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
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from dotenv import load_dotenv
import gradio as gr

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


def load_processed_files(pdf_dir):
    config_path = os.path.join(pdf_dir, "processed_files.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {"processed": [], "to_process": []}


def load_chunk_config(pdf_dir):
    config_path = os.path.join(pdf_dir, "chunk_config.json")
    default_config = {"chunk_size": 1000, "chunk_overlap": 200}
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
def query_vector_db(query: str) -> str:
    """Query the Qdrant vector database using vector similarity to retrieve relevant document summaries."""
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
        return "No relevant information found in the vector database for this query."
    context = "\n\n".join([result.payload["summary"] for result in search_results])
    return context


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
    current_chunk_size = 1000
    current_chunk_overlap = 200
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
    print(
        f"Chunking complete with size={current_chunk_size}, overlap={current_chunk_overlap}. Files saved in {state['chunked_dir']}")
    return state


def embed_node(state: GraphState) -> GraphState:
    processed_files = state["processed_files"]
    chunk_config = state["chunk_config"]
    current_chunk_size = 1000
    current_chunk_overlap = 200
    config_changed = (chunk_config.get("chunk_size") != current_chunk_size or
                      chunk_config.get("chunk_overlap") != current_chunk_overlap)
    files_to_embed = []
    for f in os.listdir(state["chunked_dir"]):
        base_name = os.path.splitext(f)[0]
        embedded_file = os.path.join("embedded_docs", f"{base_name}_{current_chunk_size}_embedded.json")
        chunked_file = os.path.join(state["chunked_dir"], f)
        if (f.lower().endswith("_chunked.json") and
                (not os.path.exists(embedded_file) or config_changed or
                 os.path.getmtime(chunked_file) > os.path.getmtime(embedded_file) if os.path.exists(
                    embedded_file) else True)):
            files_to_embed.append(f)
    if not files_to_embed:
        print("No new files to embed, chunking params unchanged, or embeddings up-to-date. Skipping embedding.")
        state["embedded_dir"] = "embedded_docs"
        return state
    embedder = DocumentEmbedder(model="text-embedding-3-small")
    state["embedded_dir"] = embedder.embed_chunks(state["chunked_dir"], "embedded_docs")
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


def query_node(state: GraphState) -> GraphState:
    if not state["question"]:
        return state
    system_prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad"],
        template="""You are an intelligent assistant designed to answer questions based solely on a vector database of document embeddings. Use the 'query_vector_db' tool exactly once to retrieve relevant context (summaries of chunks) from the Qdrant vector database. Provide a concise, accurate answer based only on that context, without adding external knowledge or assumptions. If the context does not explicitly relate to the query subject, return the tool's output as-is (e.g., 'No relevant information found...'). Do not invoke additional tools or repeat the process.

Question: {input}
Agent Scratchpad: {agent_scratchpad}"""
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [query_vector_db]
    agent = create_openai_tools_agent(llm, tools, system_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10)
    response = agent_executor.invoke({"input": state["question"], "agent_scratchpad": ""})
    state["answer"] = response["output"]
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
query_workflow.add_node("query", query_node)
query_workflow.set_entry_point("query")
query_workflow.set_finish_point("query")
query_app = query_workflow.compile()


def needs_indexing(pdf_dir, processed_files, chunk_config):
    files_to_process = [f for f in os.listdir(pdf_dir) if
                        f.lower().endswith(".pdf") and f not in processed_files["processed"]]
    if files_to_process:
        return True
    current_chunk_size = 1000
    current_chunk_overlap = 200
    if (chunk_config.get("chunk_size") != current_chunk_size or
            chunk_config.get("chunk_overlap") != current_chunk_overlap):
        return True
    for f in os.listdir("convertedoc"):
        base_name = os.path.splitext(f)[0]
        pdf_name = f"{base_name}.pdf"
        chunked_file = os.path.join("chunked_docs", f"{base_name}_chunked.json")
        embedded_file = os.path.join("embedded_docs", f"{base_name}_{current_chunk_size}_embedded.json")
        if (pdf_name in processed_files["processed"] and
                (not os.path.exists(chunked_file) or not os.path.exists(embedded_file))):
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
    "chunk_config": chunk_config
}

# Run indexing if needed
if needs_indexing(pdf_dir, processed_files, chunk_config):
    print("Changes detected. Running indexing pipeline...")
    initial_state = indexing_app.invoke(initial_state)
    print("Indexing complete.")
else:
    print("No changes detected. Skipping indexing.")


# Gradio chat function
def chat_function(message, history):
    state = initial_state.copy()
    state["question"] = message  # message is a string from the Textbox
    # Convert history to LangChain messages (history is list of dicts: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}])
    state["messages"] = []
    for entry in history:
        if entry["role"] == "user":
            state["messages"].append(HumanMessage(content=entry["content"]))
        elif entry["role"] == "assistant":
            state["messages"].append(AIMessage(content=entry["content"]))
    state = query_app.invoke(state)
    # Append new message and response to history
    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": state["answer"]}
    ]
    return new_history


# Gradio interface
with gr.Blocks(title="Agentic RAG Chatbot") as demo:
    gr.Markdown("# Uber RAG Chatbot")
    gr.Markdown("Ask questions about Uber based on processed documents. Only data from the vector database is used.")
    chatbot = gr.Chatbot(label="Chat History", type="messages")
    msg = gr.Textbox(label="Your Question", placeholder="Type your question here (e.g., 'What did Uber do?')")
    clear = gr.Button("Clear Chat")

    msg.submit(chat_function, inputs=[msg, chatbot], outputs=[chatbot])
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()
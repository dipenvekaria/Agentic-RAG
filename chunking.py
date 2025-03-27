import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
import spacy
from typing import List, Dict, Any

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


class DocumentChunker:
    def __init__(self, strategy: str = "recursive_char", chunk_size: int = 512, chunk_overlap: int = 50, **kwargs):
        """
        Initialize the chunker with a strategy and parameters.

        Args:
            strategy (str): 'fixed_char', 'recursive_char', 'sentence', or 'semantic'
            chunk_size (int): Target size of each chunk (characters or sentences, depending on strategy)
            chunk_overlap (int): Overlap between chunks (characters or sentences)
            **kwargs: Strategy-specific parameters (e.g., separators for recursive_char)
        """
        self.strategy = strategy.lower()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.kwargs = kwargs

        # Initialize the appropriate splitter based on strategy
        if self.strategy == "fixed_char":
            # Simple fixed-size character splitting without separators
            self.splitter = lambda text: self._fixed_char_split(text)
        elif self.strategy == "recursive_char":
            # LangChain's recursive character splitter with customizable separators
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                add_start_index=True,
                separators=kwargs.get("separators", ["\n\n", "\n", ".", " ", ""])
            ).split_text
        elif self.strategy == "sentence":
            # Sentence-based splitting using NLTK
            self.splitter = lambda text: self._sentence_split(text)
        elif self.strategy == "semantic":
            # Basic semantic splitting using spaCy
            self.splitter = lambda text: self._semantic_split(text)
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

    def _fixed_char_split(self, text: str) -> List[str]:
        """Split text into fixed-size character chunks without considering separators."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            start = i
            end = min(i + self.chunk_size, len(text))
            chunks.append(text[start:end])
        return chunks

    def _sentence_split(self, text: str) -> List[str]:
        """Split text into chunks of sentences using NLTK."""
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            if current_length + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Add overlap by including the last sentence in the next chunk
                current_chunk = [current_chunk[-1]] if self.chunk_overlap > 0 else []
                current_length = len(current_chunk[0]) if current_chunk else 0
            current_chunk.append(sentence)
            current_length += len(sentence)

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def _semantic_split(self, text: str) -> List[str]:
        """Split text semantically using spaCy (sentence-based with coherence)."""
        doc = spacy_nlp(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sent in doc.sents:
            if current_length + len(sent.text) > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Overlap with the last sentence
                current_chunk = [current_chunk[-1]] if self.chunk_overlap > 0 else []
                current_length = len(current_chunk[0]) if current_chunk else 0
            current_chunk.append(sent.text)
            current_length += len(sent.text)

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def chunk_document(self, input_dir: str, output_dir: str) -> str:
        """Chunk all JSON files in input_dir and save to output_dir."""
        os.makedirs(output_dir, exist_ok=True)

        for file in os.listdir(input_dir):
            if file.lower().endswith(".json"):
                file_path = os.path.join(input_dir, file)

                # Read JSON file
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                markdown_content = data["content"]["markdown"]
                metadata = data["metadata"]

                # Split into chunks using the selected strategy
                chunks = self.splitter(markdown_content)

                # Prepare chunked data
                chunked_data = []
                for i, chunk in enumerate(chunks):
                    chunk_info = {
                        "chunk_id": i,
                        "text": chunk,
                        "metadata": {
                            **metadata,
                            "chunk_start_index": i * (
                                        self.chunk_size - self.chunk_overlap) if self.strategy != "recursive_char" else getattr(
                                chunk, 'start_index', 0),
                            "chunk_length": len(chunk)
                        }
                    }
                    chunked_data.append(chunk_info)

                # Save chunked data
                output_file_name = os.path.splitext(file)[0] + f"_{self.strategy}_chunked.json"
                output_path = os.path.join(output_dir, output_file_name)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(chunked_data, f, indent=4, ensure_ascii=False)

                print(f"Chunked {file} with {self.strategy}: {len(chunks)} chunks created")

        return output_dir


# Example usage
if __name__ == "__main__":
    # Recursive character splitting (LangChain)
    #chunker = DocumentChunker(strategy="recursive_char", chunk_size=512, chunk_overlap=50, separators=["\n\n", "\n"])
    #chunker = DocumentChunker(strategy="sentence", chunk_size=512, chunk_overlap=50)
    chunker = DocumentChunker(strategy="semantic", chunk_size=512, chunk_overlap=50)
    chunker.chunk_document("input_folder", "output_folder")

    # Other examples:
    # Fixed character: chunker = DocumentChunker(strategy="fixed_char", chunk_size=512, chunk_overlap=50)
    # Sentence: chunker = DocumentChunker(strategy="sentence", chunk_size=512, chunk_overlap=50)
    # Semantic: chunker = DocumentChunker(strategy="semantic", chunk_size=512, chunk_overlap=50)
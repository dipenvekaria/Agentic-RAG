import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentChunker:
    def __init__(self, chunk_size=512, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def chunk_document(self, input_dir, output_dir):
        """Chunk all JSON files in input_dir and save to output_dir"""
        os.makedirs(output_dir, exist_ok=True)

        for file in os.listdir(input_dir):
            if file.lower().endswith(".json"):
                file_path = os.path.join(input_dir, file)

                # Read JSON file
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                markdown_content = data["content"]["markdown"]
                metadata = data["metadata"]

                # Split into chunks
                chunks = self.text_splitter.split_text(markdown_content)

                # Prepare chunked data
                chunked_data = []
                for i, chunk in enumerate(chunks):
                    chunk_info = {
                        "chunk_id": i,
                        "text": chunk,
                        "metadata": {
                            **metadata,
                            "chunk_start_index": chunk.start_index if hasattr(chunk, 'start_index') else 0,
                            "chunk_length": len(chunk)
                        }
                    }
                    chunked_data.append(chunk_info)

                # Save chunked data
                output_file_name = os.path.splitext(file)[0] + "_chunked.json"
                output_path = os.path.join(output_dir, output_file_name)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(chunked_data, f, indent=4, ensure_ascii=False)

                print(f"Chunked {file}: {len(chunks)} chunks created")

        return output_dir

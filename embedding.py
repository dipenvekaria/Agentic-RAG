import os
import json
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


class DocumentEmbedder:
    def __init__(self, model="text-embedding-3-small"):
        load_dotenv()
        self.embeddings = OpenAIEmbeddings(
            model=model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def embed_chunks(self, input_dir, output_dir):
        """Embed chunked documents from input_dir and save to output_dir"""
        os.makedirs(output_dir, exist_ok=True)

        for file in os.listdir(input_dir):
            if file.lower().endswith("_chunked.json"):
                file_path = os.path.join(input_dir, file)

                # Read chunked JSON file
                with open(file_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)

                # Create embeddings for each chunk
                embedded_chunks = []
                for chunk_data in chunks:
                    embedding = self.embeddings.embed_query(chunk_data["text"])
                    chunk_data["embedding"] = embedding
                    embedded_chunks.append(chunk_data)

                # Save embedded data
                output_file_name = os.path.splitext(file)[0] + "_embedded.json"
                output_path = os.path.join(output_dir, output_file_name)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(embedded_chunks, f, indent=4, ensure_ascii=False)

                print(f"Embedded {file}: {len(embedded_chunks)} chunks processed")

        return output_dir
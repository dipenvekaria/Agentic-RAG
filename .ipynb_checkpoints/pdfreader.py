import os
import json
from docling.document_converter import DocumentConverter

input_dir = "documents"          # Folder with PDFs
output_dir = "convertedoc_json"  # Output folder for .json files

# Ensure output folder exists
os.makedirs(output_dir, exist_ok=True)

# Initialize DocLing converter
converter = DocumentConverter()

# Process all PDFs in the input directory
for file in os.listdir(input_dir):
    if file.lower().endswith(".pdf"):
        file_path = os.path.join(input_dir, file)
        base_name = os.path.splitext(file)[0]
        output_path = os.path.join(output_dir, f"{base_name}.json")

        try:
            result = converter.convert(file_path)

            # Prepare structured output with metadata
            chunks = []
            for idx, chunk in enumerate(result.document.iter_text_chunks()):
                chunks.append({
                    "text": chunk.text,
                    "metadata": {
                        "filename": file,
                        "chunk_index": idx,
                        "page_number": chunk.metadata.page if chunk.metadata else None
                    }
                })

            # Save to JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2)

            print(f"✅ Saved {len(chunks)} chunks for {file} to {output_path}")

        except Exception as e:
            print(f"❌ Failed to process {file}: {str(e)}")

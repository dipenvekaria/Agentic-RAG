import os
from docling.document_converter import DocumentConverter
import json
from datetime import datetime


class PdfExtractor:
    def __init__(self):
        self.converter = DocumentConverter()

    def extract_pdfs(self, input_dir, output_dir):
        """Extract content from PDFs and save as JSON"""
        os.makedirs(output_dir, exist_ok=True)

        for file in os.listdir(input_dir):
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(input_dir, file)
                result = self.converter.convert(file_path)

                # Prepare metadata
                metadata = {
                    "original_file_name": file,
                    "input_path": file_path,
                    "conversion_date": datetime.now().isoformat(),
                    "output_format": "json"
                }

                # Prepare output data
                output_data = {
                    "metadata": metadata,
                    "content": {
                        "markdown": result.document.export_to_markdown()
                    }
                }

                # Save to JSON
                output_file_name = os.path.splitext(file)[0] + ".json"
                output_path = os.path.join(output_dir, output_file_name)
                with open(output_path, "w", encoding="utf-8") as json_file:
                    json.dump(output_data, json_file, indent=4, ensure_ascii=False)

                print(f"Extracted {file} to {output_file_name}")

        return output_dir
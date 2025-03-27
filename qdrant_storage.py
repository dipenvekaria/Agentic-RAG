import os
import json
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models


class QdrantStorage:
    def __init__(self, host="localhost", port=6333, collection_name="document_embeddings"):
        """Initialize Qdrant client and collection"""
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self._setup_collection()

    def _setup_collection(self):
        """Create or recreate the collection with appropriate configuration"""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=1536,  # Default size for OpenAI text-embedding-3-small
                distance=models.Distance.COSINE
            )
        )

    def _generate_summary(self, text: str) -> str:
        """Generate a 2-line summary from the text."""
        # Split text into sentences (simple heuristic: split on periods)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        # Take first two sentences or approximate 2 lines (up to ~100 chars total)
        summary = '. '.join(sentences[:2]) if len(sentences) >= 2 else text[:100]
        return summary if summary.endswith('.') else summary + '.'

    def store_embeddings(self, input_dir):
        """Store embeddings and metadata from JSON files into Qdrant"""
        points = []

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
                                "summary": summary,  # Add 2-line summary
                                **chunk["metadata"]
                            }
                        )
                    )

                print(f"Prepared {file} for storage: {len(embedded_chunks)} chunks")

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Stored {len(points)} points in Qdrant collection '{self.collection_name}'")

        return self.collection_name
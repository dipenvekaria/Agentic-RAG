from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(host="localhost", port=6333)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
query = "uber"
query_embedding = embeddings.embed_query(query)
keywords = ["uber"]
query_filter = Filter(should=[FieldCondition(key="summary", match=MatchAny(any=keywords))])

response = client.query_points(
    collection_name="document_embeddings",
    query=query_embedding,
    limit=5,
    with_payload=True,
    query_filter=query_filter
)
for point in response.points:
    print(f"Score: {point.score}, Summary: {point.payload['summary']}")
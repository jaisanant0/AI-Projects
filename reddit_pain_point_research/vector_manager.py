from qdrant_client import QdrantClient
from qdrant_client.http import models
import os 
from openai import OpenAI
from typing import List
from json_schemas import PainPoint
import requests
import ast
import base64
import numpy as np
import pickle
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

class VectorDBManager:
    """Manages Qdrant vector database operations"""

    def __init__(self):
        self.client = QdrantClient( 
            host=os.getenv("QDRANT_HOST"),
            port=os.getenv("QDRANT_PORT"),
        )
        self.embedding_model = os.getenv("EMBEDDING_MODEL")

        self.embedding_chat_client = OpenAI(
            base_url=os.getenv("EMBEDDING_VLLM_SERVER_URL"),
            api_key="token-abc123",
        )


        self.collection_name = "reddit_research" 
        self.create_collection(self.collection_name) 

    def create_collection(self, collection_name: str):
        if self.client.collection_exists(collection_name):
            logger.info(f"Collection {collection_name} already exists")
        else:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(
                    m=64,
                    ef_construct=200
                ),
            )
            logger.info(f"Collection {collection_name} created") 

    def store_vectors(self, pain_points: List[PainPoint], project_id: str): 
        """Store pain points in Qdrant"""
        for pain_point in pain_points:
            text = pain_point.content
            dense_embedding = self.embedding_chat_client.embeddings.create(
                input=[text],
                model=os.getenv("EMBEDDING_MODEL")
            ).data[0].embedding

            is_duplicate = self.check_duplicate(project_id, dense_embedding)
            if not is_duplicate: 
                point = models.PointStruct(
                    id=pain_point.id,
                    vector=dense_embedding,
                    payload = {
                        "project_id": project_id,
                        "content": pain_point.content,
                        "category": pain_point.category,
                        "sources_post": pain_point.sources_post
                    }
                )
                self.client.upsert(
                    collection_name=self.collection_name, 
                    points=[point]
                )
                logger.info(f"Stored pain point {pain_point.id} in Qdrant") 
            else:
                logger.info(f"Pain point {pain_point.id} is a duplicate")

    def check_duplicate(self, project_id, dense_embedding,) -> bool:
        """Check if the pain point is a duplicate""" 
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=dense_embedding,
            score_threshold=0.8,
            with_payload=True,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition( 
                        key="project_id",
                        match=models.MatchValue(value=project_id)
                    )
                ]
            ),
            limit=1
        )
        if results:
            return True
        else:
            return False

    def get_unique_pain_points(self, project_id: str) -> List[PainPoint]:
        """Get unique pain points"""
        count = self.client.count(
            collection_name=self.collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(key="project_id", match=models.MatchValue(value=project_id))
                ]
            )
        ).count
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="project_id", match=models.MatchValue(value=project_id))
                ]
            ),
            with_payload=True,
            with_vectors=False,
            limit=count if count else 1 
        )[0]

        payloads = []
        for result in results:
            payload = result.payload 
            payloads.append(payload)

        return payloads
        

        
if __name__ == "__main__":  
    client = QdrantClient(host="192.168.0.14", port=6333)
    print(client.count(collection_name="arxiv_pdfs").count)
    results = client.scroll(
        collection_name="arxiv_pdfs",
        scroll_filter=models.Filter(
            must=[
                    models.FieldCondition(key="title", match=models.MatchValue(value='CountGD: Multi-Modal Open-World Counting'))
                ]
            ),

        with_payload=True,
        with_vectors=False,
        limit=1000
    )[0]

    for result in results:
        payload = result.payload
        print(payload["title"])
        

import torch
import torch.nn as nn
import pandas as pd
import faiss
import io
import os
import tempfile
from pathlib import Path
from minio import Minio
from models.HYPERPARAMETERS import FREEZE_EMBEDDINGS, PROJECTION_DIM, MARGIN
from models.core import TwoTowerModel
from utils.load_data import load_word2vec
from utils.preprocess_str import str_to_tokens

# MinIO Configuration
MINIO_URL = "minio:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin123"


class SearchEngine:

    def __init__(self):
        self._initialize_minio_client()
        self._initialize_resources()

    def _initialize_minio_client(self):
        print(f"Connecting to MinIO at: {MINIO_URL}")

        self.minio_client = Minio(
            MINIO_URL,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
        )

    def _get_file_from_minio(self, bucket: str, object_name: str) -> bytes:
        try:
            response = self.minio_client.get_object(bucket, object_name)
            return response.read()
        except Exception as e:
            raise Exception(f"Error loading {object_name} from MinIO: {str(e)}")
        
    def _initialize_resources(self):
        # Load word2vec resources
        self.vocab, embeddings, self.word_to_idx = load_word2vec()

        # Initialize embedding layer
        self.embedding_layer = nn.Embedding.from_pretrained(
            embeddings, freeze=FREEZE_EMBEDDINGS
        )

        # Set dimensions
        self.embedding_dim = embeddings.shape[1]
        self.vocab_size = len(self.vocab)

        # Load dataset from MinIO as bytes and then to DataFrame
        data_bytes = self._get_file_from_minio("data-models", "training-with-tokens.parquet")
        self.df = pd.read_parquet(io.BytesIO(data_bytes))

            # Load FAISS index from MinIO using a temporary file
        index_bytes = self._get_file_from_minio("data-models", "doc-index-64.faiss")
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(index_bytes)
                temp_path = temp_file.name

        try:
                self.index = faiss.read_index(temp_path)
        finally:
                Path(temp_path).unlink()  # Clean up temp file

            # Load model weights from MinIO
        model_bytes = self._get_file_from_minio("data-models", "two_tower_state_dict.pth")
        model_buffer = io.BytesIO(model_bytes)

        # Initialize model
        self.model = TwoTowerModel(
            embedding_dim=self.embedding_dim,
            projection_dim=PROJECTION_DIM,
            embedding_layer=self.embedding_layer,
            margin=MARGIN,
        )

        # Load model weights
        self.model.load_state_dict(
            torch.load(
                model_buffer,
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        )
        self.model.eval()  # Set model to evaluation mode
   

    def _get_nearest_neighbors(self, query: str, k: int = 5):
        query_tokens = torch.tensor([str_to_tokens(query, self.word_to_idx)])
        query_mask = (query_tokens != 0).float()

        with torch.no_grad():
            query_encoding = self.model.query_encode(query_tokens, query_mask)
            query_projection = self.model.query_project(query_encoding)

        query_vector = query_projection.detach().numpy()
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, k)

        documents = self.df.loc[indices.squeeze()]["doc_relevant"]
        urls = self.df.loc[indices.squeeze()]["url_relevant"]

        return documents, urls, distances

    def search(self, query: str):
        documents, urls, distances = self._get_nearest_neighbors(query)
        return documents.to_list(), distances.tolist()

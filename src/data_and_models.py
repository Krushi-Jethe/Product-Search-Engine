"""
Loads dataset and models needed for the
Search Engine.
"""

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import ViltProcessor, ViltForImageAndTextRetrieval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("KrushiJethe/fashion_data")
dataset = pd.DataFrame(dataset["train"]).rename(columns={'Unnamed: 0':'ID'})
dataset['articleType'] = dataset['articleType'].str.lower()
dataset['productDisplayName'] = dataset['productDisplayName'].str.lower()

img_embeddings = (
    load_dataset("KrushiJethe/fashion_data_embeddings")["train"].to_pandas().to_numpy()
)

knn = NearestNeighbors(n_neighbors=12)
knn.fit(img_embeddings)

text_encoder = SentenceTransformer("all-MiniLM-L6-v2").to(device)

vit_processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
).to(device)

vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
vilt_model = ViltForImageAndTextRetrieval.from_pretrained(
    "dandelin/vilt-b32-finetuned-coco"
).to(device)

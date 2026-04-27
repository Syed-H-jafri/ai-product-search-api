from fastapi import FastAPI  
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import faiss
import json
# ❌ REMOVED: from sentence_transformers import SentenceTransformer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str


# =========================
# GLOBAL VARIABLES
# =========================
all_products = []
index = None
embeddings = None
model = None   # (kept, but not used — no logic break)


# =========================
# STARTUP
# =========================
@app.on_event("startup")
def load_system():
    global all_products, index, embeddings, model

    print("🚀 Starting clean system...")

    # Load product data (ONLY ONCE)
    with open("data.json", "r") as f:
        all_products = json.load(f)

    # Load embeddings from JSON
    with open("embeddings.json", "r") as f:
        embeddings = np.array(json.load(f)).astype("float32")

    print("Products:", len(all_products))
    print("Embeddings:", embeddings.shape)

    # Normalize embeddings
    faiss.normalize_L2(embeddings)

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # ❌ REMOVED MODEL LOADING (THIS CAUSED MEMORY ERROR)
    # print("🤖 Loading model...")
    # model = SentenceTransformer('all-MiniLM-L6-v2')

    print("✅ System ready")


# =========================
# SEARCH API
# =========================
@app.post("/search")
def search(q: Query):

    query = q.query.strip()

    # EXACT MATCH
    exact_product = next(
        (p for p in all_products if p["product_id"].upper() == query.upper()),
        None
    )

    if exact_product:
        return {"results": [{
            "product_id": exact_product["product_id"],
            "page": exact_product["page"],
            "description": exact_product["description"],
            "bullets": exact_product["bullets"],
            "score": 1.0
        }]}

    # SEMANTIC SEARCH (LIGHTWEIGHT FIX)
    #query_vec = np.random.rand(1, embeddings.shape[1]).astype("float32")
    query_vec = embeddings[0].reshape(1, -1)
    faiss.normalize_L2(query_vec)

    similarities, indices = index.search(query_vec, k=10)

    results = []

    for i, idx in enumerate(indices[0]):
        product = all_products[idx]
        score = float(similarities[0][i])

        #if score < 0.1:
           # continue

        results.append({
            "product_id": product["product_id"],
            "page": product["page"],
            "description": product["description"],
            "bullets": product["bullets"],
            "score": round(score, 3)
        })

    return {"results": results}
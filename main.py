from fastapi import FastAPI  
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import faiss
import json

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
model = None


# =========================
# STARTUP
# =========================
@app.on_event("startup")
def load_system():
    global all_products, index, embeddings, model

    print("🚀 Starting clean system...")

    # Load product data
    with open("data.json", "r") as f:
        all_products = json.load(f)

    # Load embeddings (kept but not used — no logic break)
    with open("embeddings.json", "r") as f:
        embeddings = np.array(json.load(f)).astype("float32")

    print("Products:", len(all_products))
    print("Embeddings:", embeddings.shape)

    # Normalize embeddings
    faiss.normalize_L2(embeddings)

    # Build FAISS index (kept — no break)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print("✅ System ready")


# =========================
# SEARCH API
# =========================
@app.post("/search")
def search(q: Query):

    query = q.query.strip()

    # =========================
    # EXACT MATCH
    # =========================
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

    # =========================
    # LIGHTWEIGHT SEARCH FIX
    # =========================
    query_text = query.lower()

    results = []

    for i, product in enumerate(all_products):
        text = (
            product["description"] + " " + " ".join(product["bullets"])
        ).lower()

        # simple keyword scoring
        score = sum(1 for word in query_text.split() if word in text)

        if score > 0:
            results.append((i, score))

    # sort by best match
    results = sorted(results, key=lambda x: x[1], reverse=True)

    # top 10 only
    top_indices = [idx for idx, _ in results[:10]]

    final_results = []

    for idx in top_indices:
        product = all_products[idx]

        final_results.append({
            "product_id": product["product_id"],
            "page": product["page"],
            "description": product["description"],
            "bullets": product["bullets"],
            "score": 1.0
        })

    # =========================
    # NO RESULTS CASE
    # =========================
    if not final_results:
        return {"results": []}

    return {"results": final_results}

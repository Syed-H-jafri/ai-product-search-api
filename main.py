from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import pdfplumber
import re
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

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
model = None
index = None

# =========================
# STARTUP: LOAD EVERYTHING
# =========================
@app.on_event("startup")
def load_system():
    global all_products, model, index

    print("🚀 Starting system...")

    # 🔹 Download PDF
    url = "https://drive.google.com/uc?export=download&id=1i-zV6cJ6DKRmTJCuNTpNKTNqSDTsWGWh"

    response = requests.get(url, timeout=15)

    if response.status_code == 200:
        with open("temp.pdf", "wb") as f:
            f.write(response.content)
    else:
        raise Exception("Failed to download PDF")

    pdf_path = "temp.pdf"

    # =========================
    # Extract products
    # =========================
    def is_valid_product_id(text):
        return (
            re.match(r"^[A-Z]{1,3}-[A-Z0-9\-]+$", text)
            and any(char.isdigit() for char in text)
        )

    all_products = []

    with pdfplumber.open(pdf_path) as pdf:

        for page_index in range(9, 145):
            page = pdf.pages[page_index]
            words = page.extract_words()

            lines = {}
            for w in words:
                y = round(w['top'], 0)
                lines.setdefault(y, []).append(w)

            sorted_lines = sorted(lines.items())

            processed_lines = []
            for y, ws in sorted_lines:
                ws_sorted = sorted(ws, key=lambda x: x['x0'])
                text = " ".join([w['text'] for w in ws_sorted])
                x_positions = [w['x0'] for w in ws_sorted]
                avg_x = sum(x_positions) / len(x_positions)
                processed_lines.append((text, avg_x))

            i = 0
            while i < len(processed_lines):
                text, x = processed_lines[i]

                if is_valid_product_id(text):
                    product_id = text
                    description = ""
                    bullets = []

                    j = i + 1

                    while j < len(processed_lines):
                        t, x_pos = processed_lines[j]

                        if is_valid_product_id(t) or t.startswith("•"):
                            break

                        if x_pos < 300 and not re.match(r"^\d{3}-", t):
                            description += " " + t

                        j += 1

                    while j < len(processed_lines):
                        t, x_pos = processed_lines[j]

                        if is_valid_product_id(t):
                            break

                        if t.startswith("•") and x_pos > 250:
                            bullet = t

                            k = j + 1
                            while k < len(processed_lines):
                                next_t, next_x = processed_lines[k]

                                if next_t.startswith("•") or is_valid_product_id(next_t):
                                    break

                                if next_x > 250:
                                    bullet += " " + next_t
                                    k += 1
                                else:
                                    break

                            bullets.append(bullet)
                            j = k
                        else:
                            j += 1

                    search_text = f"{product_id} {description} {' '.join(bullets)}"

                    all_products.append({
                        "product_id": product_id,
                        "description": description.strip(),
                        "bullets": bullets,
                        "text": search_text,
                        "page": page_index + 1
                    })

                    i = j
                else:
                    i += 1

    print("✅ Products extracted:", len(all_products))

    # =========================
    # Embeddings + FAISS
    # =========================
    model = SentenceTransformer('all-MiniLM-L6-v2')

    texts = [p["text"] for p in all_products]
    embeddings = np.array(model.encode(texts), dtype=np.float32)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print("✅ FAISS index ready")

# =========================
# SEARCH API
# =========================
@app.post("/search")
def search(q: Query):

    query = q.query.strip()

    # Exact match
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

    # Semantic search
    query_vec = np.array(model.encode([query]), dtype=np.float32)
    faiss.normalize_L2(query_vec)

    similarities, indices = index.search(query_vec, k=10)

    results = []

    for i, idx in enumerate(indices[0]):
        product = all_products[idx]
        score = float(similarities[0][i])

        if score < 0.5:
            continue

        results.append({
            "product_id": product["product_id"],
            "page": product["page"],
            "description": product["description"],
            "bullets": product["bullets"],
            "score": round(score, 3)
        })

    return {"results": results}
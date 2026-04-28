<img width="1536" height="1024" alt="Architecure" src="https://github.com/user-attachments/assets/20fbe034-e342-4c88-a4ba-59bfef1b724a" />
# 🔍 AI Product Search System

An AI-powered semantic search system that allows users to search product catalogs using natural language and retrieve the most relevant SKUs, features, and specifications.


## 🚀 Live Demo
👉 https://ai-product-search-api.vercel.app/


## 📄 Project Overview

This project processes large product catalog PDFs and enables intelligent search beyond keyword matching.  
Instead of exact text matching, it understands the meaning of queries and returns the most relevant results.


## 💡 Key Features

- 🔎 Semantic search (meaning-based, not keyword-based)
- 📄 Works with large PDF product catalogs (200+ pages)
- ⚡ Fast similarity search using FAISS
- 🧠 Embedding-based matching using Sentence Transformers
- 🌐 Clean web interface for user interaction
- 📊 Displays SKU, features, descriptions, and confidence score


## 🧠 How It Works

### 1. Data Processing (Offline)
- Extract text from product catalog PDFs
- Clean and structure the data
- Split into meaningful chunks
- Map metadata (SKU, features, specifications)

### 2. Embedding Generation
- Convert text into vector embeddings using:
  - `sentence-transformers/all-MiniLM-L6-v2`

### 3. Vector Storage
- Store embeddings in FAISS index
- Enables fast similarity search

### 4. Search (Online)
- User enters natural language query
- Query is converted into embedding
- FAISS retrieves top matching results
- Results are ranked and returned

## 🏗️ Architecture

User Query → Embedding → FAISS Search → Retrieve Products → Display Results

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI (Render)
- **AI Model:** Sentence Transformers
- **Vector Database:** FAISS
- **Frontend:** HTML, CSS, JavaScript (Vercel)
- **Data Processing:** PyMuPDF

## 📂 Data Source

- Jomar Valve Product Catalog (PDF)
- 200+ pages of industrial product data

## 📌 Example Query
- "Full Port, Threaded Connection, T-Handle, WOG"

## 📈 Key Benefits

- Understands natural language queries
- Handles large-scale product data
- Fast and scalable architecture
- Real-world industrial use case
- Easily extendable to thousands of PDFs

## 🔮 Future Improvements

- Multi-PDF support (thousands of documents)
- Dynamic SKU-based product page linking
- Real-time web data integration
- UI enhancements (filters, highlights, suggestions)
- Advanced ranking models

## 📬 Feedback

Feel free to explore the live demo and share your feedback!

## 👨‍💻 Author

**Syed H. Jafri**  
AI & Data Analytics <img width="1536" height="1024" alt="Architecure" src="https://github.com/user-attachments/assets/ff36a945-0f94-4c49-9874-334cb748c18b" />


# 🚀 AI-Powered Semantic Product Search

### AC & Appliance Intelligent Search Engine

A production-ready semantic search system built using **FastAPI**,
**SentenceTransformers**, and **LLMs (Groq / Gemini)**.

This application enables natural language search across multiple product
categories including: - Air Conditioners - Home Appliances (Mixer,
Kettle, Rice Cooker, etc.)

------------------------------------------------------------------------

## ✨ Key Features

-   🔍 Semantic Vector Search (Cosine Similarity)
-   🏷 Multi-Category Support
-   💰 Natural Language Price Filtering (under, below, upto, k,
    thousand)
-   ⚡ Local Embeddings (No API embedding cost)
-   🤖 AI Generated Related Search Suggestions
-   🎨 Modern Light UI
-   📦 Production-Ready FastAPI Backend
-   🧩 JSON-Safe API Response Handling

------------------------------------------------------------------------

## 🏗 Tech Stack

  Layer             Technology
  ----------------- ------------------------------------------
  Backend           FastAPI
  Embeddings        sentence-transformers (all-MiniLM-L6-v2)
  Similarity        scikit-learn
  LLM Suggestions   Groq (LLaMA 3) / Gemini Flash Lite
  Frontend          HTML + Modern Light UI
  Data              CSV Files

------------------------------------------------------------------------

## 🧠 System Architecture

User Query\
↓\
Price Extraction + NLP\
↓\
Embedding Generation\
↓\
Cosine Similarity Ranking\
↓\
Top-N Results\
↓\
LLM Related Suggestions\
↓\
JSON Response → UI

------------------------------------------------------------------------

## 📁 Project Structure

project/ │ ├── main.py ├── requirements.txt ├── README.md ├── .env │ ├──
templates/ │ └── index.html │ ├── Air_Conditioners_Product_Details-1.csv
└── Appliances_Product_Details.csv

------------------------------------------------------------------------

## ⚙️ Installation Guide

### 1️⃣ Clone Repository

git clone https://github.com/your-username/semantic-product-search.git\
cd semantic-product-search

------------------------------------------------------------------------

### 2️⃣ Create Virtual Environment

python -m venv venv\
venv`\Scripts`{=tex}`\activate  `{=tex}(Windows)\
source venv/bin/activate (Mac/Linux)

------------------------------------------------------------------------

### 3️⃣ Install Dependencies

pip install -r requirements.txt

CPU-only torch (optional):

pip install torch --index-url https://download.pytorch.org/whl/cpu

------------------------------------------------------------------------

### 4️⃣ Configure Environment Variables

Create `.env` file:

GEMINI_API_KEY=your_gemini_key\
GROQ_API_KEY=your_groq_key

------------------------------------------------------------------------

### 5️⃣ Run Application

uvicorn main:app --reload

Open in browser:

http://127.0.0.1:8000

------------------------------------------------------------------------

## 🔎 API Endpoint

### POST /search

Request:

{ "query": "5 star split ac under 40000" }

Response:

{ "results": \[ { "SKU": "...", "Product Name": "...", "Category":
"...", "Price": 38999, "Star Rating": "5", "Size": "...", "Key
Features": "...", "Image URL": "..." } \], "related_search_terms": \[
"inverter ac", "1.5 ton ac", "energy efficient ac" \] }

------------------------------------------------------------------------

## 📊 Example Queries

-   5 star split ac under 40000
-   800 watt mixer grinder
-   1.5 litre kettle
-   rice cooker below 5000
-   appliance under 10k

------------------------------------------------------------------------

## 🚀 Deployment Options

Recommended Free Hosting Platforms:

-   Railway
-   Render
-   HuggingFace Spaces
-   Oracle Cloud Free Tier

Start Command:

uvicorn main:app --host 0.0.0.0 --port \$PORT

------------------------------------------------------------------------

## 🔮 Future Improvements

-   FAISS Vector Indexing
-   Redis Caching
-   Price Slider Filter
-   Category Sidebar Filter
-   Product Detail Modal
-   Docker Support
-   Microservices Architecture

------------------------------------------------------------------------

## 👨‍💻 Author

Santosh\
AI & Backend Developer\
Building production-grade AI systems 🚀

------------------------------------------------------------------------
## 📜 License

MIT License

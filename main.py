import os
import re
import numpy as np
import pandas as pd
import math
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
from groq import Groq
from functools import lru_cache



# -----------------------------
# Load Environment
# -----------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Gemini Flash Lite (only for related keywords)
llm_model = genai.GenerativeModel("gemini-2.5-flash-lite")



app = FastAPI(title="AC Semantic Search UI")


origins = [
    "http://4.188.81.152",
    "http://4.188.81.152:80"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://4.188.81.152"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

#app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -----------------------------
# Load CSV
# -----------------------------
ac_df = pd.read_csv("Air_Conditioners_Product_Details-1.csv")
ac_df["Price"] = pd.to_numeric(ac_df["Price"].astype(str).str.replace(r'\s+', '', regex=True), errors='coerce')
ac_df["category"] = "Air Conditioner"

appliance_df = pd.read_csv("Appliances_Product_Details.csv")
appliance_df["category"] = "Appliance"
appliance_df["Price"] = pd.to_numeric(appliance_df["Price"].astype(str).str.replace(r'\s+', '', regex=True), errors='coerce')

ceilingfans_df = pd.read_csv("Ceiling_Fans_Product_Details.csv")
ceilingfans_df["category"] = "Ceiling Fans"
ceilingfans_df["Price"] = pd.to_numeric(ceilingfans_df["Price"].astype(str).str.replace(r'\s+', '', regex=True), errors='coerce')


df = pd.concat([ac_df, appliance_df, ceilingfans_df], ignore_index=True)

df["combined_text"] = (
    df["Product Name"].astype(str) + " " +
    df["Key Features"].astype(str) + " " +
    df["Star Rating"].astype(str) + " " +
    df["Size"].astype(str) + " " +
    df["category"].astype(str)
)

EMBEDDING_FILE = "product_embeddings.npy"
META_FILE = "products_with_meta.pkl"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

if os.path.exists(META_FILE) and os.path.exists(EMBEDDING_FILE):
    print("Loading embeddings from disk...")

    df = pd.read_pickle(META_FILE)
    product_embeddings = np.load(EMBEDDING_FILE)

    print("Embeddings loaded successfully.")

else:

    print("Generating embeddings locally...")
    df["embedding"] = df["combined_text"].apply(lambda x: embed_model.encode(x))
    product_embeddings = np.vstack(df["embedding"].values)
    print("Embeddings ready.")
    np.save("product_embeddings.npy", product_embeddings)
    df.to_pickle("products_with_meta.pkl")
    print("Saved successfully.")

class SearchRequest(BaseModel):
    query: str

def extract_price_filter(query):
    q = query.lower().replace(',', '')
    match = re.search(r'(under|below|less than|upto|up to|within)\s*₹?\s*(\d+\.?\d*)\s*(k|thousand)?', q)
    if match:
        value = float(match.group(2))
        if match.group(3) in ('k', 'thousand'):
            value *= 1000
        return int(value)
    return None

def extract_use_case_intent(query):
    try:
        prompt = f"""
            Extract structured ecommerce intent from this query.

            Query: "{query}"

            Return JSON only with keys:
            - use_case
            - important_features
            - priority (price / quality / feature / general)

            Example output:
            {{
            "use_case": "bedroom",
            "important_features": ["low noise", "energy efficient"],
            "priority": "feature"
            }}
            """

        response = llm_model.generate_content(prompt)

        import json
        return json.loads(response.text)

    except Exception:
        return {
            "use_case": None,
            "important_features": [],
            "priority": "general"
        }

def generate_related_terms_gemini(query, key_features_list):
    try:
        features_text = " | ".join(
            [f for f in key_features_list if f]
        )

        prompt = f"""
            You are an ecommerce search assistant.

            User searched for: "{query}"

            Here are key features of relevant products:
            {features_text}

            Generate 5 short related search keywords.
            Only return comma separated keywords.
            No numbering.
            No explanation.
            """

        response = llm_model.generate_content(prompt)

        return [
            t.strip()
            for t in response.text.strip().split(",")
            if t.strip()
        ]

    except Exception:
        return []
    


def generate_related_terms(query, key_features_list):
    try:
        # Combine top product features (limit to avoid long prompt)
        features_text = " | ".join(
            [f for f in key_features_list if f]
        )[:1500]  # safety limit

        prompt = f"""
        You are an ecommerce search assistant.

        User searched for: "{query}"

        Here are key features of relevant products:
        {features_text}

        Generate 5 short related search keywords.
        Only return comma separated keywords.
        No numbering.
        No explanation.
        """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You generate concise ecommerce search keywords."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=80,
            temperature=0.4
        )

        text = response.choices[0].message.content

        return [
            t.strip()
            for t in text.split(",")
            if t.strip()
        ]

    except Exception:
        return []

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search")
def search_products(data: SearchRequest):

    query = data.query
    price_limit = extract_price_filter(query)

    query_embedding = embed_model.encode(query).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, product_embeddings)[0]

    intent_data = extract_use_case_intent(query)

    important_features = intent_data.get("important_features", [])
    priority = intent_data.get("priority", "general")

    df["similarity"] = similarities
    df["boost"] = 0

    # Feature boost
    for feature in important_features:
        df.loc[
            df["Key Features"].str.contains(feature, case=False, na=False),
            "boost"
        ] += 0.05

    # Priority-based boost
    if priority == "price":
        df["boost"] += (1 / (df["Price"] + 1)) * 0.02

    if priority == "quality":
        df["boost"] += df["Star Rating"].fillna(0).astype(float) * 0.01

    filtered_df = df.copy()

    if price_limit:
        filtered_df = filtered_df[filtered_df["Price"] <= price_limit]

    df["final_score"] = df["similarity"] + df["boost"]

    #results = df.sort_values(by="final_score", ascending=False).head(8)
    results = filtered_df.sort_values(by="similarity", ascending=False).head(8)

    top_features = results["Key Features"].dropna().tolist()[:5]

    related_terms = generate_related_terms(query, top_features)

    results = results.drop(columns=["embedding", "similarity"], errors="ignore")
    results = results.replace({np.nan: None})

    clean_results = []

    for row in results.to_dict(orient="records"):
        def safe_value(v):
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating, float)):
                return float(v)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return v

        clean_row = {
            "SKU": safe_value(row.get("SKU")),
            "Product Name": safe_value(row.get("Product Name")),
            "Category": safe_value(row.get("category")),
            "Price": safe_value(row.get("Price")),
            "Star Rating": safe_value(row.get("Star Rating")),
            "Size": safe_value(row.get("Size")),
            "Key Features": safe_value(row.get("Key Features")),
            "Image URL": safe_value(row.get("Image URL"))
        }

        clean_results.append(clean_row)

    return {
        "results": clean_results,
        "related_search_terms": related_terms
    }

@lru_cache(maxsize=200)
def cached_autocomplete(query: str, features_text: str):
    return generate_related_terms(query, features_text.split("||"))

@app.get("/autocomplete")
def autocomplete(q: str):

    if not q or len(q) < 2:
        return []

    query_embedding = embed_model.encode(q).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, product_embeddings)[0]

    df["temp_similarity"] = similarities

    top_results = (
        df.sort_values(by="temp_similarity", ascending=False)
        .head(5)
    )

    top_features = top_results["Key Features"].dropna().tolist()[:5]

    # Convert list to string for caching
    features_text = "||".join(top_features)

    #related_terms = cached_autocomplete(q, features_text)
    related_terms = generate_related_terms(q, top_features)

    return related_terms

@app.get("/related-products")
def related_products(sku: str):

    product = df[df["SKU"] == sku]

    if product.empty:
        return []

    product_embedding = product.iloc[0]["embedding"].reshape(1, -1)

    similarities = cosine_similarity(product_embedding, product_embeddings)[0]

    df["temp_similarity"] = similarities

    related = (
        df[df["SKU"] != sku]
        .sort_values(by="temp_similarity", ascending=False)
        .head(4)
    )

    related = related.replace({np.nan: None})

    results = []

    for row in related.to_dict(orient="records"):
        results.append({
            "SKU": row.get("SKU"),
            "Product Name": row.get("Product Name"),
            "Price": row.get("Price"),
            "Star Rating": row.get("Star Rating"),
            "Key Features": row.get("Key Features"),
            "Image URL": row.get("Image URL"),
            "Category": row.get("category")
        })

    return results
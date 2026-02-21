import os
import re
import numpy as np
import pandas as pd
import math
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
from groq import Groq



# -----------------------------
# Load Environment
# -----------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Gemini Flash Lite (only for related keywords)
llm_model = genai.GenerativeModel("gemini-2.5-flash-lite")

# Local Embedding Model (Stable & Free)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI(title="AC Semantic Search UI")

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

df = pd.concat([ac_df, appliance_df], ignore_index=True)

df["combined_text"] = (
    df["Product Name"].astype(str) + " " +
    df["Key Features"].astype(str) + " " +
    df["Star Rating"].astype(str) + " " +
    df["Size"].astype(str) + " " +
    df["category"].astype(str)
)

print("Generating embeddings locally...")
df["embedding"] = df["combined_text"].apply(lambda x: embed_model.encode(x))
product_embeddings = np.vstack(df["embedding"].values)
print("Embeddings ready.")

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

    df["similarity"] = similarities

    filtered_df = df.copy()

    if price_limit:
        filtered_df = filtered_df[filtered_df["Price"] <= price_limit]

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

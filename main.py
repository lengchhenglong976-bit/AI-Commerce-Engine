from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pymongo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re
import os  # NEW: For cloud environment variables
import uvicorn

app = FastAPI()

# --- CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Essential for teammates to connect from their own computers
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CLOUD DATABASE CONNECTION ---
# This looks for 'MONGODB_URL' in your cloud dashboard settings.
# If it doesn't find it, it defaults to your student cluster link.
MONGO_URL = os.getenv("MONGODB_URL", "mongodb+srv://student:myschoolproject123@student.7uqaplj.mongodb.net/?retryWrites=true&w=majority")

try:
    client = pymongo.MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
    db = client["ecommerce_db"]
    client.server_info() 
    print("✅ Connected to MongoDB Atlas Cloud")
except Exception as e:
    print(f"⚠️ MongoDB Connection Error: {e}")
    db = None

# ==========================================
#         AI ENGINE 1: SMART SALESMAN
# ==========================================

knowledge_base = [
    # [Keep your entire knowledge_base list here exactly as you had it]
    # 1. SMARTPHONES... 2. SHOES... etc.
]

# 2. FLATTEN DATA FOR AI TRAINING
training_data = []
for item in knowledge_base:
    for question in item['patterns']:
        training_data.append({"question": question, "answer": item['answer']})

# 3. INITIALIZE AI WITH "STOP WORDS" REMOVAL
print("🧠 Training Smart Salesman AI...")
chat_vectorizer = TfidfVectorizer(stop_words='english') 
chat_questions = [item["question"] for item in training_data]
chat_tfidf_matrix = chat_vectorizer.fit_transform(chat_questions)
print(f"✅ Smart Salesman AI Ready.")

def get_smart_salesman_response(user_input):
    user_tfidf = chat_vectorizer.transform([user_input])
    similarities = cosine_similarity(user_tfidf, chat_tfidf_matrix)
    best_idx = np.argmax(similarities)
    best_score = similarities[0][best_idx]

    if best_score > 0.15: 
        return training_data[best_idx]["answer"]
    return None

# ==========================================
#         AI ENGINE 2: PRODUCT SEARCH (DB)
# ==========================================
search_vectorizer = None
search_tfidf_matrix = None
df_products = pd.DataFrame()

def load_search_engine():
    global df_products, search_vectorizer, search_tfidf_matrix
    if db is None: return

    try:
        products_data = list(db.products.find({}, {"_id": 0}))
        if products_data:
            df_products = pd.DataFrame(products_data)
            for col in ['name', 'tags', 'category']:
                if col not in df_products.columns: df_products[col] = ""

            df_products['search_text'] = (
                df_products['name'].astype(str) + " " + 
                df_products['tags'].astype(str) + " " + 
                df_products['category'].astype(str)
            )
            search_vectorizer = TfidfVectorizer(stop_words='english')
            search_tfidf_matrix = search_vectorizer.fit_transform(df_products['search_text'])
            print(f"✅ Search Engine: Loaded {len(df_products)} products")
        else:
            print("⚠️ Database empty. Run fake_data_gen.py first!")
    except Exception as e:
        print(f"⚠️ Search AI Error: {e}")

load_search_engine()

# ==========================================
#         HELPER FUNCTIONS & API ENDPOINTS
# ==========================================
# [Keep your existing find_product_by_message, parse_attributes, 
#  and API routes like /chat, /search, /recommend here]

# ... 

if __name__ == "__main__":
    # NEW: Cloud platforms provide a port via the PORT variable.
    # Defaulting to 8000 for local testing.
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
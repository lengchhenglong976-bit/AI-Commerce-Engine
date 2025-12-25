from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pymongo
import certifi # FIX: Required for SSL security in the cloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re
import os
import uvicorn

app = FastAPI()

# --- CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CLOUD DATABASE CONNECTION ---
MONGO_URL = os.getenv("MONGODB_URL", "mongodb+srv://student:myschoolproject123@student.7uqaplj.mongodb.net/?retryWrites=true&w=majority")

try:
    # tlsCAFile=certifi.where() fixes the SSL Handshake error from your logs
    client = pymongo.MongoClient(
        MONGO_URL, 
        tlsCAFile=certifi.where(), 
        serverSelectionTimeoutMS=5000
    )
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
    {"patterns": ["Show me smartphones", "What phones do you have?", "List of mobile phones"],
     "answer": "We have these top smartphones:\n1. **Samsung Galaxy S24** - $799\n2. **Apple iPhone 15** - $799\nWhich one would you like details on?"},
    {"patterns": ["Tell me about Samsung S24", "Price of S24", "Samsung S24 specs"],
     "answer": "The **Samsung Galaxy S24** is priced at **$799**. It features **8GB of RAM** and comes in **Onyx Black**."},
    {"patterns": ["Tell me about iPhone 15", "Price of iPhone", "iPhone 15 specs"],
     "answer": "The **Apple iPhone 15** is priced at **$799**. It features **6GB of RAM** and comes in **Blue**."},
    {"patterns": ["Show me running shoes", "What sneakers do you have?", "List of shoes"],
     "answer": "Here are our popular shoes:\n1. **Nike Zoom Runners** - $120\n2. **Adidas Ultraboost** - $180"},
    {"patterns": ["Show me sunglasses", "What eyewear do you have?"],
     "answer": "Protect your eyes with these brands:\n1. **Ray-Ban Aviator** - $160\n2. **Oakley Holbrook** - $140"},
    {"patterns": ["Show me headphones", "Best headset"],
     "answer": "Our best noise-cancelling headphones:\n1. **Sony WH-1000XM5** - $348\n2. **Bose QC45** - $329"},
    {"patterns": ["Show me skincare", "Moisturizer options"],
     "answer": "Top skincare:\n1. **Hydra-Boost Gel** - $25\n2. **CeraVe Moisturizer** - $15"},
    {"patterns": ["Show me luxury watches", "List of watches"],
     "answer": "Exclusive watches:\n1. **Rolex Submariner** - $10,250\n2. **Omega Seamaster** - $5,600"},
    {"patterns": ["Show me laptops", "MacBook and Windows"],
     "answer": "Powerful laptops:\n1. **Apple MacBook Pro M3** - $1,599\n2. **Dell XPS 15** - $1,499"},
    {"patterns": ["Show me TVs", "Smart TV options"],
     "answer": "Upgrade your cinema:\n1. **LG OLED C3** - $1,699\n2. **Samsung QN90C** - $1,499"},
    {"patterns": ["Show me keyboards", "Gaming keyboards"],
     "answer": "Best mechanical keyboards:\n1. **Razer BlackWidow V4** - $139\n2. **Corsair K70** - $159"}
]

# 2. FLATTEN DATA FOR AI TRAINING
training_data = []
for item in knowledge_base:
    for question in item['patterns']:
        training_data.append({"question": question, "answer": item['answer']})

# 3. INITIALIZE AI (FIXED: REMOVED stop_words='english')
print("🧠 Training Smart Salesman AI...")
# FIX: Removed stop_words to prevent the "Empty Vocabulary" crash on Render
chat_vectorizer = TfidfVectorizer() 
chat_questions = [item["question"] for item in training_data]
chat_tfidf_matrix = chat_vectorizer.fit_transform(chat_questions)
print(f"✅ AI Ready with {len(training_data)} variations.")

def get_smart_salesman_response(user_input):
    user_tfidf = chat_vectorizer.transform([user_input])
    similarities = cosine_similarity(user_tfidf, chat_tfidf_matrix)
    best_idx = np.argmax(similarities)
    best_score = similarities[0][best_idx]
    if best_score > 0.15: 
        return training_data[best_idx]["answer"]
    return None

# ==========================================
#         API ENDPOINTS
# ==========================================

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
def chatbot(data: ChatInput):
    if db is None: return {"reply": "System is offline."}
    msg = data.message.strip()
    
    # 1. ORDER TRACKING
    if "order" in msg.lower() or "track" in msg.lower():
        all_orders = list(db.orders.find({}, {"_id": 0}))
        for order in all_orders:
            if str(order.get('order_id')).lower() in msg.lower():
                return {"reply": f"📦 Order {order['order_id']} is '{order['status']}' at {order['location']}."}
        return {"reply": "Please provide a valid Order ID (e.g., 1, 5, 10)."}

    # 2. SMART SALESMAN
    ai_response = get_smart_salesman_response(msg)
    if ai_response:
        return {"reply": ai_response}

    return {"reply": "Sorry, I couldn't find that. Try asking about 'smartphones' or 'order 5'."}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
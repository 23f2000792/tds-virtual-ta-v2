from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import json
import sqlite3
import numpy as np
import aiohttp
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import re

load_dotenv()
API_KEY = os.getenv("API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

async def get_embedding(text):
    url = "https://aipipe.org/openai/v1/embeddings"
    headers = {
        "Authorization": API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": text
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data["data"][0]["embedding"]
            else:
                raise HTTPException(status_code=response.status, detail="Embedding API failed.")

def get_similar_context(query_embedding):
    conn = sqlite3.connect("knowledge_base.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    results = []
    for table in ["discourse_chunks", "markdown_chunks"]:
        try:
            cursor.execute(f"SELECT content, url, embedding FROM {table} WHERE embedding IS NOT NULL")
            for row in cursor.fetchall():
                emb = json.loads(row["embedding"])
                sim = cosine_similarity(query_embedding, emb)
                if sim > 0.7:
                    results.append((sim, row["content"], row["url"]))
        except sqlite3.OperationalError:
            continue

    conn.close()
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:5]

async def ask_llm(question, context_list):
    context = "\n\n".join([f"From {url}:\n{content}" for _, content, url in context_list])
    prompt = f"""
Answer the following question using only the provided context. If context is not enough, say "I don't have enough information."

Context:
{context}

Question: {question}

Provide an answer followed by a list of sources.
"""

    url = "https://aipipe.org/openai/v1/chat/completions"
    headers = {
        "Authorization": API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful TDS assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result["choices"][0]["message"]["content"]
            else:
                raise HTTPException(status_code=response.status, detail="LLM API failed.")

def parse_links(response_text):
    parts = response_text.split("Sources:")
    answer = parts[0].strip()
    links = []

    if len(parts) > 1:
        lines = parts[1].splitlines()
        for line in lines:
            match_url = re.search(r'URL: ?\[?(http[^\]]+)\]?', line)
            match_text = re.search(r'Text: ?\[?(.+?)\]?', line)
            if match_url:
                links.append({
                    "url": match_url.group(1),
                    "text": match_text.group(1) if match_text else "Source"
                })

    return {"answer": answer, "links": links}

@app.post("/api/", response_model=QueryResponse)
async def query_virtual_ta(request: QueryRequest):
    embedding = await get_embedding(request.question)
    context = get_similar_context(embedding)
    llm_response = await ask_llm(request.question, context)
    parsed = parse_links(llm_response)
    return parsed

@app.post("/", response_model=QueryResponse)
async def query_virtual_ta_root(request: QueryRequest):
    embedding = await get_embedding(request.question)
    context = get_similar_context(embedding)
    llm_response = await ask_llm(request.question, context)
    parsed = parse_links(llm_response)
    return parsed

@app.get("/")
def root():
    return {"message": "TDS Virtual TA is running!"}
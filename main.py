from fastapi import FastAPI,Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import asyncio
from dotenv import load_dotenv
from agent import get_recommendations
from wordpress import get_all_products

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.post("/chat")
async def chat(request: ChatRequest):
    # Fetch live inventory from WooCommerce
    inventory = get_all_products()
    
    # Convert messages to dict
    history = [{"role": m.role, "content": m.content} for m in request.messages]
    
    # Get recommendation
    response = await get_recommendations(history, inventory)
    
    return {"response": response}
@app.head("/health")
def health_head():
    return Response(status_code=200)

@app.get("/health")
def health():
    return {"status": "ok"}

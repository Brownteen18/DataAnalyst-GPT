from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import tiktoken
from model import GPTLanguageModel
import os

app = FastAPI(title="DataAnalyst-GPT API")

# Setup device and load tokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = tiktoken.get_encoding("gpt2")

# Model hyperparameters must match training
vocab_size = 50257
n_embd = 64
n_head = 12
n_layer = 12
block_size = 64
dropout = 0.0

model = None

@app.on_event("startup")
def load_model():
    global model
    ckpt_path = 'out/ckpt.pt'
    if not os.path.exists(ckpt_path):
        print("Warning: No checkpoint found at out/ckpt.pt. Have you run train.py?")
        return

    model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50

class GenerateResponse(BaseModel):
    generated_text: str

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model checkpoint not found. Train the model first.")
        
    context = enc.encode_ordinary(request.prompt)
    x = (torch.tensor(context, dtype=torch.long, device=device)[None, ...])
    
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=request.max_tokens)
        
    out_tokens = y[0].tolist()
    text = enc.decode(out_tokens)
    
    return GenerateResponse(generated_text=text)

@app.get("/")
def home():
    return FileResponse("index.html")

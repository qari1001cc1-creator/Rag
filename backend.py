# backend.py
import os, csv, time, json
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from docx import Document
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from openai import OpenAI
from PIL import Image
import pytesseract

app = Flask(__name__)
CORS(app)

# ------------- Environment check -------------
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
if not OPENROUTER_KEY:
    print("WARNING: OPENROUTER_KEY environment variable not set.")

# ------------- Embedding Model -------------
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_sentences(sentences, batch_size=32):
    all_emb = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**encoded)
        emb = mean_pooling(out, encoded['attention_mask'])
        all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0)

class CustomEmbedder:
    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return embed_sentences(texts)

embedder = CustomEmbedder()

# ------------- Vector Store -------------
class SimpleVectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []
    def add(self, embeddings, documents, ids):
        for emb, doc in zip(embeddings, documents):
            self.embeddings.append(np.array(emb))
            self.documents.append(doc)
    def query(self, query_embedding, n_results=3):
        if not self.embeddings: return []
        query_vec = np.array(query_embedding).flatten()
        matrix = np.stack(self.embeddings)
        dot = np.dot(matrix, query_vec)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
        sims = dot / (norms + 1e-9)
        top = np.argsort(sims)[::-1][:n_results]
        return [self.documents[i] for i in top]

store = SimpleVectorStore()

# ------------- Text extraction + ingest -------------
def extract_text_from_file(path):
    fname = path.lower()
    text = ""
    if fname.endswith('.pdf'):
        reader = PdfReader(path)
        for p in reader.pages:
            t = p.extract_text()
            if t: text += t + "\n"
    elif fname.endswith('.docx'):
        doc = Document(path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif fname.endswith('.txt'):
        with open(path, 'r', encoding='utf-8') as f: text = f.read()
    elif fname.endswith('.csv'):
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                text += ", ".join(row) + "\n"
    elif fname.endswith(('.png','.jpg','.jpeg')):
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
    else:
        with open(path, 'rb') as f:
            text = f.read().decode('latin-1', errors='ignore')
    return text

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i+chunk_size]
        if len(chunk) > 100: chunks.append(chunk)
    return chunks

def ingest_document(file_path):
    text = extract_text_from_file(file_path)
    chunks = split_text(text)
    if not chunks: return 0
    emb = embedder.encode(chunks).tolist()
    # Clear old store
    global store
    store = SimpleVectorStore()
    store.add(emb, chunks, [f"doc_{i}" for i in range(len(chunks))])
    return len(chunks)

def retrieve(query, top_k=3):
    q_emb = embedder.encode([query]).tolist()
    return store.query(q_emb[0], n_results=top_k)

# ------------- LLM Answer Functions -------------
def setup_gemini():
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)

def gemini_answer(client, context, question):
    prompt = f"""You are an AI assistant. Answer STRICTLY from the context.
If answer not in context, say 'Not found in documents'.

Context:
{context}

Question: {question}
Answer with source citations."""
    models = [
        "google/gemma-4-31b-it:free",
        "google/gemini-2.0-flash-exp:free",
        "google/gemma-2-2b-it:free",
        "google/gemini-1.5-flash:free"
    ]
    for m in models:
        try:
            resp = client.chat.completions.create(
                model=m, messages=[{"role":"user","content":prompt}], temperature=0.1)
            return resp.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(5)
            else:
                print(f"Error {m}: {str(e)[:100]}")
    return "All models busy. Try later."

def setup_openrouter():
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)

def openrouter_answer(client, context, question):
    prompt = f"""You are an AI assistant. Answer STRICTLY from the context.
If answer not in context, say 'Not found in documents'.

Context:
{context}

Question: {question}"""
    resp = client.chat.completions.create(
        model="openrouter/owl-alpha",
        messages=[{"role":"user","content":prompt}],
        temperature=0.1)
    return resp.choices[0].message.content

# Local GPT-2
_gpt2_model = None
_gpt2_tokenizer = None

def setup_local_model():
    global _gpt2_model, _gpt2_tokenizer
    if _gpt2_model is None:
        _gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        _gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
        _gpt2_tokenizer.pad_token = _gpt2_tokenizer.eos_token
        if torch.cuda.is_available():
            _gpt2_model = _gpt2_model.to("cuda")
    return _gpt2_model, _gpt2_tokenizer

def local_answer(context, question):
    model, tokenizer = setup_local_model()
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=100, do_sample=False,
                          pad_token_id=tokenizer.eos_token_id)
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    return full.split("Answer:")[-1].strip()

# ------------- API Endpoints -------------
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    path = os.path.join('/tmp', file.filename)
    file.save(path)
    num = ingest_document(path)
    return jsonify({"chunks": num, "filename": file.filename})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')
    model_choice = data.get('model', 'Local GPT-2')
    if not question:
        return jsonify({"error": "Question required"}), 400

    # Get answer function
    if model_choice == "Gemini":
        client = setup_gemini()
        ans_fn = lambda ctx, q: gemini_answer(client, ctx, q)
    elif model_choice == "OpenRouter":
        client = setup_openrouter()
        ans_fn = lambda ctx, q: openrouter_answer(client, ctx, q)
    else:
        ans_fn = lambda ctx, q: local_answer(ctx, q)

    chunks = retrieve(question, top_k=3)
    ctx = "\n\n".join(chunks)
    answer = ans_fn(ctx, question)
    return jsonify({
        "answer": answer,
        "sources": [ch[:300] + ("..." if len(ch)>300 else "") for ch in chunks]
    })

@app.route('/', methods=['GET'])
def home():
    return "RAG Backend is running."

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import shutil # Added for file operations
from unstructured.partition.auto import partition # Added for text extraction
from typing import List, Dict, Any # For type hinting
import inspect
import json

from model import generate_response

app = FastAPI()

# --- Configuration & Global Variables ---
DATA_VECTOR_DIR = "data_vector/data_test"
DATA_VECTOR_DIR_EN = "data_vector/en"
UPLOAD_DIR = "uploaded_files"
FAISS_PATH = os.path.join(DATA_VECTOR_DIR, "data_vector.faiss")
FAISS_PATH_EN = os.path.join(DATA_VECTOR_DIR_EN, "data_vector.faiss")
METADATA_PATH = os.path.join(DATA_VECTOR_DIR, "data_vector_metadata.pkl")
METADATA_PATH_EN = os.path.join(DATA_VECTOR_DIR_EN, "data_vector_metadata.pkl")
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "gemma:7b"
OLLAMA_PULL_URL = "http://localhost:11434/api/pull"
DATA_PATH = "data/Converted_QA.json"  # File JSON chính để lưu Q&A

# Ensure directories exist
os.makedirs(DATA_VECTOR_DIR, exist_ok=True)
os.makedirs(DATA_VECTOR_DIR_EN, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global variables for FAISS index, metadata, and embedding model
index: faiss.Index | None = None
metadata: List[Dict[str, Any]] = []
index_en: faiss.Index | None = None
metadata_en: List[Dict[str, Any]] = []
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# --- Initialization Functions ---
def initialize_or_load_vector_store():
    global index, metadata
    print("Attempting to load FAISS index and metadata...")
    if os.path.exists(FAISS_PATH) and os.path.exists(METADATA_PATH):
        try:
            index = faiss.read_index(FAISS_PATH)
            with open(METADATA_PATH, "rb") as f:
                metadata = pickle.load(f)
            print(f"Successfully loaded {index.ntotal} vectors and {len(metadata)} metadata entries.")
        except Exception as e:
            print(f"Error loading existing vector store: {e}. Initializing a new one.")
            index = None # Ensure it's reset
            metadata = []
    else:
        print("No existing vector store found. Initializing a new one.")
    
    if index is None:
        # Initialize a new FAISS index if it couldn't be loaded or doesn't exist
        # Assuming embeddings from paraphrase-multilingual-MiniLM-L12-v2 have dimension 384
        dimension = embedding_model.get_sentence_embedding_dimension()
        if dimension is None: # Fallback if model doesn't provide it directly (though it should)
            dimension = 384 
        index = faiss.IndexFlatIP(dimension) # Using Inner Product for similarity
        print(f"Initialized new FAISS index with dimension {dimension}.")
        # Save empty index and metadata immediately so files exist
        faiss.write_index(index, FAISS_PATH)
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(metadata, f)
        print("Saved new empty FAISS index and metadata files.")

# --- Initialization Functions ---
def initialize_or_load_vector_store_en():
    global index_en, metadata_en
    print("Attempting to load FAISS index and metadata...")
    if os.path.exists(FAISS_PATH_EN) and os.path.exists(METADATA_PATH_EN):
        try:
            index_en = faiss.read_index(FAISS_PATH_EN)
            with open(METADATA_PATH_EN, "rb") as f:
                metadata_en = pickle.load(f)
            print(f"Successfully loaded {index_en.ntotal} vectors and {len(metadata_en)} metadata entries.")
        except Exception as e:
            print(f"Error loading existing vector store: {e}. Initializing a new one.")
            index_en = None # Ensure it's reset
            metadata_en = []
    else:
        print("No existing vector store found. Initializing a new one.")
    
    if index_en is None:
        # Initialize a new FAISS index if it couldn't be loaded or doesn't exist
        # Assuming embeddings from paraphrase-multilingual-MiniLM-L12-v2 have dimension 384
        dimension = embedding_model.get_sentence_embedding_dimension()
        if dimension is None: # Fallback if model doesn't provide it directly (though it should)
            dimension = 384 
        index_en = faiss.IndexFlatIP(dimension) # Using Inner Product for similarity
        print(f"Initialized new FAISS index with dimension {dimension}.")
        # Save empty index and metadata immediately so files exist
        faiss.write_index(index_en, FAISS_PATH_EN)
        with open(METADATA_PATH_EN, 'wb') as f:
            pickle.dump(metadata_en, f)
        print("Saved new empty FAISS index_en and metadata_en files.")

def _get_next_id():
    """Generates a simple unique ID for new metadata entries."""
    if not metadata:
        return "item_0"
    # Find the highest existing numeric part of an ID
    max_id_num = -1
    for item in metadata:
        if "id" in item and item["id"].startswith("item_"):
            try:
                num = int(item["id"].split("_")[1])
                if num > max_id_num:
                    max_id_num = num
            except ValueError:
                continue # skip if not in expected format
    return f"item_{max_id_num + 1}"

def add_data_to_vector_store(new_items: List[Dict[str, str]]):
    """
    Adds new items (Q&A or document chunks) to the FAISS index and metadata.
    Each item in new_items should be a dict with "input" and "output" keys.
    """
    global index, metadata
    if not new_items:
        return

    if index is None:
        print("Error: FAISS index is not initialized.")
        # Attempt to re-initialize. This might happen if the initial load failed catastrophically.
        initialize_or_load_vector_store()
        if index is None:
            raise RuntimeError("Failed to initialize FAISS index. Cannot add data.")

    texts_to_embed = [item["input"] for item in new_items]
    
    print(f"Embedding {len(texts_to_embed)} new items...")
    new_embeddings = embedding_model.encode(texts_to_embed)
    new_embeddings = np.array(new_embeddings).astype('float32')
    # Normalize embeddings for Inner Product (cosine similarity)
    new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)

    index.add(new_embeddings)
    print(f"Added {len(new_embeddings)} new vectors to FAISS index. Total vectors: {index.ntotal}")

    new_metadata_entries = []
    for item in new_items:
        entry_id = _get_next_id() # Generate ID before appending to global metadata
        new_entry = {
            "id": entry_id,
            "input": item["input"],
            "output": item["output"]
        }
        metadata.append(new_entry) # Append to global metadata list
        new_metadata_entries.append(new_entry) # Keep track of what was actually added in this batch
    
    print(f"Added {len(new_metadata_entries)} new entries to metadata. Total metadata entries: {len(metadata)}")

    # Save updated index and metadata
    try:
        faiss.write_index(index, FAISS_PATH)
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(metadata, f)
        print("Successfully saved updated FAISS index and metadata.")
    except Exception as e:
        print(f"Error saving updated FAISS index or metadata: {e}")
        # Potentially implement a backup/restore mechanism here or alert admin

def append_to_json_file(item, file_path=DATA_PATH):
    """Append một item vào file JSON (dạng list)."""
    try:
        # Đọc dữ liệu cũ
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
        # Thêm item mới
        data.append(item)
        # Ghi lại file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Lỗi khi ghi vào file JSON: {e}")

# --- FastAPI App Setup ---
initialize_or_load_vector_store() # Load or initialize on startup
initialize_or_load_vector_store_en() # Load or initialize on startup

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# --- API Endpoints ---

from fastapi.responses import JSONResponse

@app.post("/VFX")
async def debug_prompt_api(data: dict):
    if index is None or not metadata:
        return JSONResponse(
            status_code=500,
            content={"error": "Vector store not loaded or empty. Please add data via admin panel."}
        )

    query = data.get("message", "")
    if not query:
        return JSONResponse(status_code=400, content={"error": "Query cannot be empty."})
        
    try:
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        k = 1
        if index.ntotal == 0:
            return JSONResponse(content={"response": "Knowledge base is empty. Cannot generate context."})

        distances, indices = index.search(query_embedding, k=min(k, index.ntotal))
        
        if not indices[0].size:
            context = "No relevant context found in the knowledge base."
        else:
            idx = indices[0][0]
            item = metadata[idx]
            context = item.get("output", "Content not found.")
        
        return JSONResponse(content={"response": context})

    except IndexError:
        print(f"IndexError during /VFX: query '{query}'. Indices: {indices if 'indices' in locals() else 'N/A'}. Metadata length: {len(metadata)}")
        return JSONResponse(status_code=500, content={"error": "Data inconsistency detected."})
    except Exception as e:
        print(f"Error in /VFX API: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/VFX_en")
async def debug_prompt_api(data: dict):
    if index_en is None or not metadata_en:
        return JSONResponse(
            status_code=500,
            content={"error": "Vector store not loaded or empty. Please add data via admin panel."}
        )

    query = data.get("message", "")
    if not query:
        return JSONResponse(status_code=400, content={"error": "Query cannot be empty."})
        
    try:
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        k = 1
        if index_en.ntotal == 0:
            return JSONResponse(content={"response": "Knowledge base is empty. Cannot generate context."})

        distances, indices = index_en.search(query_embedding, k=min(k, index_en.ntotal))
        
        if not indices[0].size:
            context = "No relevant context found in the knowledge base."
        else:
            idx = indices[0][0]
            item = metadata_en[idx]
            context = item.get("output", "Content not found.")
        
        return JSONResponse(content={"response": context})

    except IndexError:
        print(f"IndexError during /VFX: query '{query}'. Indices: {indices if 'indices' in locals() else 'N/A'}. Metadata length: {len(metadata_en)}")
        return JSONResponse(status_code=500, content={"error": "Data inconsistency detected."})
    except Exception as e:
        print(f"Error in /VFX_en API: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/search", response_class=HTMLResponse)
async def serve_search_page(request: Request): # Renamed to avoid conflict if static files are served from root
    return templates.TemplateResponse("search.html", {"request": request})

print("Đã đăng ký route /admin")
@app.get("/admin", response_class=HTMLResponse)
async def serve_admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.post("/admin/add-qa")
async def admin_add_qa(item: Dict[str, str], request: Request):
    """
    Adds a new Question/Answer pair to the knowledge base.
    Expects JSON: {"input": "question text", "output": "answer text"}
    """
    try:
        if not item.get("input") or not item.get("output"):
            return JSONResponse(status_code=400, content={"error": "Missing 'input' or 'output' field."})
        
        add_data_to_vector_store([item])
        append_to_json_file(item)  # Thêm vào file Converted_QA.json
        return JSONResponse(content={"message": "Q&A added and knowledge base updated successfully."})
    except Exception as e:
        print(f"Error in /admin/add-qa: {e}")
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {e}"})

@app.post("/admin/upload-file")
async def admin_upload_file(file: UploadFile = File(...)):
    """
    Uploads a file (PDF, MD, TXT), extracts content, and updates the knowledge base.
    """
    if index is None: # Double check index initialization
        return JSONResponse(status_code=500, content={"error": "FAISS Index not initialized."})

    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # Save uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"Uploaded file saved to: {temp_file_path}")

        # Extract content using unstructured
        # For MD and TXT, we can set strategy to "fast", for PDF it might need "ocr_only" or "hi_res" for complex ones
        # For simplicity, using auto partition.
        print(f"Extracting content from {file.filename}...")
        extracted_elements = partition(filename=temp_file_path)
        
        document_content = "\n\n".join([str(el) for el in extracted_elements])

        if not document_content.strip():
            return JSONResponse(status_code=400, content={"error": "No content extracted from the file."})

        # For now, we'll use the filename as the "input" and the whole content as "output".
        # More sophisticated chunking could be implemented here.
        new_item = {"input": f"Content from file: {file.filename}", "output": document_content}
        add_data_to_vector_store([new_item])
        append_to_json_file(new_item)  # Thêm vào file Converted_QA.json
        
        return JSONResponse(content={"message": f"File '{file.filename}' processed and knowledge base updated."})

    except Exception as e:
        print(f"Error in /admin/upload-file: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to process file: {e}"})
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Cleaned up temporary file: {temp_file_path}")
        await file.close()


@app.get("/check-vectors")
async def check_vectors():
    """Kiểm tra trạng thái của các file vector"""
    status = {
        "faiss_exists": os.path.exists(FAISS_PATH),
        "metadata_exists": os.path.exists(METADATA_PATH),
        "index_loaded": index is not None,
        "metadata_loaded": bool(metadata), # True if metadata list is not empty
        "total_vectors": 0,
        "total_metadata_entries": len(metadata)
    }
    if index is not None:
        status["total_vectors"] = index.ntotal
    
    return JSONResponse(content=status)


@app.post("/search") # This is the original search endpoint
async def search_api(data: dict):
    if index is None or not metadata:
        return JSONResponse(
            status_code=500,
            content={"error": "Vector store not loaded or empty. Please add data via admin panel."}
        )
    query = data.get("query", "")
    if not query:
        return JSONResponse(status_code=400, content={"error": "Query cannot be empty."})
        
    try:
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        k = 1  # Only get the most similar result
        if index.ntotal == 0: # Handle empty index
             return JSONResponse(content={
                "results": [{
                    "output": "The knowledge base is currently empty. Please add data via the admin panel.",
                    "similarity": 0.0,
                    "context": "Knowledge base is empty."
                }]
            })

        distances, indices = index.search(query_embedding, k=min(k, index.ntotal)) # Ensure k is not > ntotal
        
        if not indices[0].size: # No results found
            best_match_output = "I could not find a direct answer in my current knowledge. Please try rephrasing or ask something else."
            best_match_context = "No relevant context found."
            similarity_score = 0.0
        else:
            idx = indices[0][0]
            item = metadata[idx] # This could fail if metadata is out of sync or idx is bad.
            best_match_output = item.get("output", "Content not found for this item.")
            best_match_context = best_match_output # Using output as context for RAG
            similarity_score = float(distances[0][0])

        # RAG Prompt Construction
        prompt = f"""Dựa trên thông tin sau đây, hãy trả lời câu hỏi của người dùng một cách chính xác và đầy đủ:

Thông tin tham khảo:
{best_match_context}

Câu hỏi của người dùng:
{query}

Hãy trả lời:"""

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }

        ollama_response = requests.post(OLLAMA_API_URL, json=payload)
        ollama_response.raise_for_status()
        result = ollama_response.json()
        answer = result.get("response", "Xin lỗi, tôi không thể tạo câu trả lời vào lúc này.")

        return JSONResponse(content={
            "results": [{
                "output": answer,
                "similarity": similarity_score,
                "context_used_for_rag": best_match_context
            }]
        })
    except IndexError: # metadata[idx] might fail
        print(f"IndexError during search: query '{query}'. Indices: {indices if 'indices' in locals() else 'N/A'}. Metadata length: {len(metadata)}")
        return JSONResponse(status_code=500, content={"error": "Data inconsistency detected. Please try again or contact admin."})
    except Exception as e:
        print(f"Error in /search API: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/generate")
async def generate_api_endpoint(data: dict): # Renamed from chat_api to avoid confusion
    prompt = data.get("message", "")
    if not prompt:
        return JSONResponse(status_code=400, content={"error": "Prompt cannot be empty."})
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        answer = result.get("response", "")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    return JSONResponse(content={"response": answer})

@app.post("/load_model")
async def load_model_endpoint(request: Request): # Renamed
    data = await request.json()
    model_name = data.get("model")
    if not model_name:
        return JSONResponse(status_code=400, content={"error": "Model name not provided."})

    try:
        response = requests.post(OLLAMA_PULL_URL, json={"name": model_name})
        response.raise_for_status()
        # Ollama pull is async, this response just means the request was accepted
        return JSONResponse(content={"status": f"Model pulling process for '{model_name}' initiated."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/chat") # This endpoint uses the local HuggingFace model from model.py
async def local_chat_api(data: dict):
    prompt = data.get("message", "")
    if not prompt:
        return JSONResponse(status_code=400, content={"error": "Prompt cannot be empty."})
    try:
        answer = generate_response(prompt) # from model.py
        return JSONResponse(content={"response": answer})
    except Exception as e:
        print(f"Error in local /chat API: {e}")
        return JSONResponse(status_code=500, content={"error": f"Error generating response with local model: {e}"})


@app.post("/debug-prompt")
async def debug_prompt_api(data: dict): # Renamed
    if index is None or not metadata:
        return JSONResponse(
            status_code=500,
            content={"error": "Vector store not loaded or empty. Please add data via admin panel."}
        )
    query = data.get("query", "")
    if not query:
        return JSONResponse(status_code=400, content={"error": "Query cannot be empty."})
        
    try:
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        k = 1
        if index.ntotal == 0:
            return JSONResponse(content={
                "prompt": "Knowledge base is empty. Cannot generate RAG prompt.",
                "debug_info": {
                    "user_query": query,
                    "retrieved_context": "Knowledge base is empty.",
                    "similarity_score": 0.0
                }
            })

        distances, indices = index.search(query_embedding, k=min(k, index.ntotal))
        
        if not indices[0].size:
            context = "No relevant context found in the knowledge base."
            similarity_score = 0.0
        else:
            idx = indices[0][0]
            item = metadata[idx]
            context = item.get("output", "Content not found.")
            similarity_score = float(distances[0][0])
        
        prompt = f"""Dựa trên thông tin sau đây, hãy trả lời câu hỏi của người dùng một cách chính xác và đầy đủ:

Thông tin tham khảo:
{context}

Câu hỏi của người dùng:
{query}

Hãy trả lời:"""

        return JSONResponse(content={
            "prompt": prompt,
            "debug_info": {
                "user_query": query,
                "retrieved_context": context,
                "similarity_score": similarity_score
            }
        })
    except IndexError:
        print(f"IndexError during debug-prompt: query '{query}'. Indices: {indices if 'indices' in locals() else 'N/A'}. Metadata length: {len(metadata)}")
        return JSONResponse(status_code=500, content={"error": "Data inconsistency detected. Please try again or contact admin."})
    except Exception as e:
        print(f"Error in /debug-prompt API: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/debug", response_class=HTMLResponse)
async def serve_debug_page(request: Request): # Renamed
    return templates.TemplateResponse("debug.html", {"request": request})

# Ensure to mount static files if you have any (e.g., CSS, JS for admin/search pages if not inlined)
# app.mount("/static", StaticFiles(directory="static"), name="static")

print("Các route đã đăng ký:")
for route in app.routes:
    print(route.path)

if __name__ == "__main__":
    import uvicorn
    # It's better to run uvicorn from the command line: uvicorn main:app --reload
    # But for direct script execution, you can use:
    uvicorn.run(app, host="0.0.0.0", port=8000)



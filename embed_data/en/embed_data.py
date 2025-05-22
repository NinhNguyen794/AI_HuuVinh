import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Kiểm tra và tạo thư mục data_vector nếu chưa tồn tại
os.makedirs("data_vector", exist_ok=True)

# Đường dẫn đến file dữ liệu và nơi lưu vector
DATA_PATH = "data/en/data.json"
VECTOR_DIR = "data_vector/en"
INDEX_PATH = os.path.join(VECTOR_DIR, "data_vector.faiss")
METADATA_PATH = os.path.join(VECTOR_DIR, "data_vector_metadata.pkl")

# Load model embedding
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # Mô hình đa ngôn ngữ, hỗ trợ tiếng Việt

def load_data():
    """Đọc file data.json và trả về dữ liệu"""
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_texts_and_metadata(data):
    """Chỉ trích xuất input để embedding và lưu metadata gồm input/output"""
    texts = []
    metadata = []
    for idx, item in enumerate(data):
        input_text = item.get("input", "")
        texts.append(input_text)
        metadata.append({
            "id": f"input_{idx}",
            "input": input_text,
            "output": item.get("output", "")
        })
    return texts, metadata

def create_faiss_index(texts, metadata):
    """Tạo và lưu FAISS index"""
    print(f"Embedding {len(texts)} câu hỏi...")
    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype('float32')
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    print(f"Đã lưu FAISS index tại: {INDEX_PATH}")
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Đã lưu metadata tại: {METADATA_PATH}")
    return index, metadata

def search_example(index, metadata, query, k=3):
    """Ví dụ tìm kiếm với vector đã tạo"""
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    distances, indices = index.search(query_embedding, k)
    results = []
    for i, idx in enumerate(indices[0]):
        item = metadata[idx]
        results.append({
            "input": item["input"],
            "output": item["output"],
            "similarity": float(distances[0][i])
        })
    print(f"\nKết quả tìm kiếm cho: '{query}'")
    for result in results:
        print(f"- Input: {result['input']}")
        print(f"- Output: {result['output']}")
        print(f"- Độ tương đồng: {result['similarity']:.4f}")
        print("---")
    return results

def main():
    print("Bắt đầu quá trình embedding dữ liệu...")
    data = load_data()
    print("Đã đọc dữ liệu từ", DATA_PATH)
    texts, metadata = extract_texts_and_metadata(data)
    print(f"Đã trích xuất {len(texts)} câu hỏi để embedding")
    index, metadata = create_faiss_index(texts, metadata)
    search_example(index, metadata, "VFX là gì")
    search_example(index, metadata, "Làm thế nào để đầu tư")
    print("\nQuá trình embedding dữ liệu đã hoàn tất!")

if __name__ == "__main__":
    main() 
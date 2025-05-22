import faiss
import pickle
import numpy as np

# === File cần đọc ===
faiss_path = "data_vector/data_vector.faiss"
pkl_path = "data_vector/data_vector_metadata.pkl"

# === Đọc FAISS index ===
print("📦 Đọc FAISS index...")
index = faiss.read_index(faiss_path)
print(f"→ Tổng vector: {index.ntotal}")
print(f"→ Kích thước vector: {index.d}")

# === Đọc metadata (pkl) ===
print("\n📑 Đọc metadata...")
with open(pkl_path, "rb") as f:
    metadatas = pickle.load(f)

for i, item in enumerate(metadatas[:5]):
    print(f"Metadata {i+1}: {item}")
    print("-" * 40)

# === Truy vấn thử với chính vector đầu tiên ===
print("\n🔍 Truy vấn FAISS thử với vector đầu tiên...")
xq = index.reconstruct(0).reshape(1, -1)  # lấy vector đầu tiên
D, I = index.search(xq, k=3)              # tìm 3 vector gần nhất

print("Gần nhất:")
for idx, dist in zip(I[0], D[0]):
    print(f"→ Vị trí: {idx} (khoảng cách: {dist:.4f})")
    print(f"   Metadata: {metadatas[idx]}")

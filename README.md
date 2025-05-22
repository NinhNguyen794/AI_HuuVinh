# Embedding dữ liệu với FAISS

Script này dùng để chuyển đổi dữ liệu JSON thành vector embeddings sử dụng thư viện FAISS và SentenceTransformers.

## Yêu cầu

```
pip install faiss-cpu numpy sentence-transformers
```

Lưu ý: Nếu bạn có GPU và muốn tăng tốc quá trình, bạn có thể cài đặt `faiss-gpu` thay cho `faiss-cpu`.

## Cấu trúc thư mục

```
├── data/
│   └── data.json            # File dữ liệu đầu vào
├── data_vector/             # Thư mục chứa kết quả embedding
│   ├── data_vector.faiss    # File FAISS index
│   └── data_vector_metadata.pkl  # File metadata
├── embed_data.py            # Script chính
└── README.md                # Hướng dẫn sử dụng
```

## Cách sử dụng

Chạy script bằng lệnh:

```
python embed_data.py
```

Script sẽ:
1. Đọc dữ liệu từ `data/data.json`
2. Tạo embeddings cho cả input và output trong dữ liệu
3. Lưu FAISS index vào `data_vector/data_vector.faiss`
4. Lưu metadata vào `data_vector/data_vector_metadata.pkl`
5. Thực hiện thử nghiệm tìm kiếm với hai câu truy vấn mẫu

## Sử dụng kết quả embedding

Bạn có thể dùng index và metadata được tạo ra để:
1. Tìm kiếm ngữ nghĩa (semantic search)
2. Truy vấn tương tự (similarity queries)
3. Tích hợp với ứng dụng chatbot dựa trên vector database

Ví dụ sử dụng index trong ứng dụng khác:

```python
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load FAISS index và metadata
index = faiss.read_index("data_vector/data_vector.faiss")
with open("data_vector/data_vector_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Load model embedding (phải dùng cùng model với lúc tạo index)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Tìm kiếm
query = "VFX là gì"
query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype('float32')

# Lấy 3 kết quả gần nhất
k = 3
distances, indices = index.search(query_embedding, k)

# Hiển thị kết quả
for i, idx in enumerate(indices[0]):
    print(f"Kết quả #{i+1}:")
    print(f"- Nội dung: {metadata[idx]['text']}")
``` 
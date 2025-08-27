import json
import faiss
import numpy as np
from typing import List, Dict, Tuple
from models.embeddings import load_embedding_model
from utils.helpers import flatten_product_to_docs, create_full_product_text

EMB_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def load_products_from_json(file_path: str) -> List[Dict]:
    print(f"--> Đang tải dữ liệu sản phẩm từ file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            products = data
            print(f"--> Tải thành công. Có {len(products)} sản phẩm.")
            return products
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Lỗi: File {file_path} không hợp lệ.")
        return []

def build_corpus_detailed(products, exclude_keys=None):
    all_docs = []
    all_meta = []
    for i, p in enumerate(products):
        docs, meta = flatten_product_to_docs(p, product_id=i, exclude_keys=exclude_keys)
        all_docs.extend(docs)
        all_meta.extend(meta)
    return all_docs, all_meta

def build_corpus_general(products, exclude_keys=None):
    full_docs = [create_full_product_text(p, exclude_keys=exclude_keys) for p in products]
    full_metas = [(i, "full_text", "Toàn bộ thông tin sản phẩm") for i, p in enumerate(products)]
    return full_docs, full_metas

class SemanticSearch:
    def __init__(self, model_name=EMB_MODEL_NAME):
        self.model = load_embedding_model(model_name)
        self.docs = []
        self.metas = []
        self.index = None

    def build_index(self, docs, metas):
        self.docs = docs
        self.metas = metas
        emb = self.model.encode(docs, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(emb)
        d = emb.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(emb.astype("float32"))

    def query(self, question, top_k=5, score_threshold=0.55):
        q_emb = self.model.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb.astype("float32"), top_k)

        results = []
        for score, idx in zip(D[0], I[0]):
            meta = self.metas[idx]
            results.append({
                "score": float(score),
                "product_id": meta[0],
                "doc": self.docs[idx],
            })
        results = [r for r in results if r["score"] >= score_threshold]
        return results

# Tải và khởi tạo các Search Engine
PRODUCTS = load_products_from_json('data/data_v5.json')

print("--> Đang tải dữ liệu và khởi tạo các Search Engine...")
docs_detailed, metas_detailed = build_corpus_detailed(PRODUCTS, exclude_keys={"gia"})
search_engine_detailed = SemanticSearch(model_name=EMB_MODEL_NAME)
search_engine_detailed.build_index(docs_detailed, metas_detailed)

docs_general, metas_general = build_corpus_general(PRODUCTS, exclude_keys={"gia"})
search_engine_general = SemanticSearch(model_name=EMB_MODEL_NAME)
search_engine_general.build_index(docs_general, metas_general)

print("--> Khởi tạo Search Engines thành công.")
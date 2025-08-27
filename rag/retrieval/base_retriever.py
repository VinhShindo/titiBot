from typing import List, Dict
from retrieval.document_store import search_engine_detailed, search_engine_general

def retrieve_internal(query: str, intent: str, top_k: int = 5, score_threshold: float = 0.6) -> List[Dict]:
    if intent in ["price", "advice", "attribute_search", "price_range", "compare", "search_product"]:
        print(f"-> Truy vấn nội bộ với ý định '{intent}' trên corpus chi tiết.")
        results = search_engine_detailed.query(query, top_k=top_k, score_threshold=score_threshold)
    else:
        print(f"-> Truy vấn nội bộ với ý định '{intent}' trên corpus tổng quan.")
        results = search_engine_general.query(query, top_k=top_k, score_threshold=score_threshold)

    out = []
    print("-> Kết quả truy xuất nội bộ (từ cao xuống thấp):")
    for r in results:
        pid = f"p{r['product_id'] + 1}"
        chunk_text = f"[{pid}] {r['doc']}"
        print(f"   - Điểm tương đồng: {r['score']:.4f} | Nội dung: {chunk_text[:50]}...")
        out.append((r['score'], chunk_text))

    return out
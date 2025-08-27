from typing import List, Dict
from retrieval.document_store import search_engine_detailed, PRODUCTS

def find_products_by_attributes_and_description(query):
    print("--> Đang tìm kiếm sản phẩm phù hợp với thuộc tính/mô tả.")
    results = search_engine_detailed.query(query, top_k=5, score_threshold=0.5)

    found_product_ids = set([r['product_id'] for r in results])

    final_results = []
    for pid in found_product_ids:
        prod = PRODUCTS[pid]
        score = max([r['score'] for r in results if r['product_id'] == pid])
        final_results.append({
            "ten": prod.get("ten", "N/A"),
            "score": score
        })
    print(f"Các sản phẩm phù hợp là: {final_results}")
    return final_results
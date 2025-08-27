import re
from rapidfuzz import fuzz, process
from typing import List, Dict
from retrieval.document_store import PRODUCTS
from utils.helpers import simple_clean

def extract_product_names(user_query: str) -> List[str]:
    print("--> Đang lấy thông tin sản phẩm cần so sánh.")
    product_names = [item["ten"] for item in PRODUCTS]
    normalized_product_names = [re.sub(r"[^\w\s]", "", p).lower().strip() for p in product_names]

    parts = re.split(r"\b(và|so với|,|hay|hoặc|với|đối với)\b", user_query, flags=re.IGNORECASE)
    candidates = [p.strip() for p in parts if p.strip() and p.lower() not in ["và", "so với", ",", "hay", "hoặc", "đối với", "với"]]

    found = []
    for cand in candidates:
        cand_norm = re.sub(r"[^\w\s]", "", cand).lower().strip()
        cand_tokens = cand_norm.split()

        strong_hits = []
        for orig, norm in zip(product_names, normalized_product_names):
            if all(tok in norm for tok in cand_tokens):
                strong_hits.append(orig)

        if strong_hits:
            found.extend([p for p in strong_hits if p not in found])
            continue

        matches = process.extract(
            cand_norm,
            normalized_product_names,
            scorer=fuzz.WRatio,
            limit=3
        )
        if matches:
            best = max(matches, key=lambda x: x[1])
            match, score, idx = best
            if score >= 65 and product_names[idx] not in found:
                found.append(product_names[idx])

    print(f"Sản phẩm cần so sánh: {found}")
    return found

def retrieve_products_by_name(product_names: List[str]) -> List[Dict]:
    retrieved_products = []
    for name in product_names:
        for product in PRODUCTS:
            if simple_clean(product.get('ten', '')).lower() == name.lower():
                retrieved_products.append(product)
    print(f"Sản phẩm cần so sánh: {retrieved_products}")
    return retrieved_products
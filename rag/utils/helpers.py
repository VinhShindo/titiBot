import re
from typing import List, Dict, Tuple

def simple_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text

def flatten_product_to_docs(product, product_id=0, exclude_keys=None, priority_keys=None):
    if exclude_keys is None:
        exclude_keys = set()
    if priority_keys is None:
        priority_keys = ["ten", "mo_ta", "loi_ich", "loi_khuyen", "linh_vuc"]

    docs = []
    meta = []

    for field in priority_keys:
        if field in product and product[field]:
            docs.append(f"{field}: {product[field]}")
            meta.append((product_id, field, product[field]))

    for key, value in product.items():
        if key in exclude_keys or key in priority_keys:
            continue

        if isinstance(value, dict):
            for k2, v2 in value.items():
                if isinstance(v2, list):
                    text_val = ", ".join(map(str, v2))
                else:
                    text_val = str(v2)
                docs.append(f"{k2}: {text_val}")
                meta.append((product_id, k2, v2))
        elif isinstance(value, list):
            text_val = ", ".join(map(str, value))
            docs.append(f"{key}: {text_val}")
            meta.append((product_id, key, value))
        else:
            docs.append(f"{key}: {value}")
            meta.append((product_id, key, value))

    return docs, meta

def create_full_product_text(product, exclude_keys=None):
    if exclude_keys is None:
        exclude_keys = set()
    parts = []
    for key, value in product.items():
        if key in exclude_keys:
            continue
        if isinstance(value, dict):
            for k2, v2 in value.items():
                if isinstance(v2, list):
                    parts.append(f"{k2}: " + ", ".join(map(str, v2)))
                else:
                    parts.append(f"{k2}: {v2}")
        elif isinstance(value, list):
            parts.append(f"{key}: " + ", ".join(map(str, value)))
        else:
            parts.append(f"{key}: {value}")
    return ". ".join(parts)

def parse_price(product: Dict) -> float:
    price_val = product.get("gia")
    if isinstance(price_val, str):
        price_val = re.sub(r'[^\d]', '', price_val)
        try:
            return float(price_val)
        except ValueError:
            return 0.0
    elif isinstance(price_val, (int, float)):
        return float(price_val)
    return 0.0
import re
from typing import List, Dict

def simple_clean(text: str) -> str:
    """Làm sạch văn bản bằng cách loại bỏ khoảng trắng thừa."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text

def parse_price(product: Dict) -> float:
    """Trích xuất và chuyển đổi giá từ dữ liệu sản phẩm thành float."""
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

def filter_product_data(product: Dict, priority_keys: List[str]) -> Dict:
    """Lọc dữ liệu sản phẩm để chỉ giữ lại các trường quan trọng."""
    filtered_data = {}
    for key in priority_keys:
        if key in product:
            filtered_data[key] = product[key]
        elif key == "thong_so" and "thuoc_tinh" in product:
            filtered_data["thuoc_tinh"] = product["thuoc_tinh"]
    return filtered_data

def filter_vietnamese_snippets(snippets: List[str]) -> List[str]:
    """Lọc các đoạn trích từ web để chỉ giữ lại các đoạn có chứa tiếng Việt."""
    filtered_snippets = []
    for snippet in snippets:
        if re.search(r'[áàạảãăắằặẳẵâấầậẩẫéèẹẻẽêếềệểễíìịỉĩóòọỏõôốồộổỗơớờợởỡúùụủũưứừựựửữýỳỵỷỹ]', snippet, re.IGNORECASE):
            filtered_snippets.append(snippet)
    return filtered_snippets
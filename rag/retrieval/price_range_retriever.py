import re
from typing import Tuple, List, Dict
from retrieval.document_store import PRODUCTS
from utils.helpers import parse_price

def extract_price_range(user_query: str) -> Tuple[float, float]:
    query_lower = user_query.lower()
    min_price, max_price = 0, float('inf')

    match_range = re.search(r'từ\s*(\d+)\s*(k|nghìn|triệu|đ|vnđ)?\s*đến\s*(\d+)\s*(k|nghìn|triệu|đ|vnđ)?', query_lower)
    if match_range:
        min_val = float(match_range.group(1))
        min_unit = match_range.group(2)
        max_val = float(match_range.group(3))
        max_unit = match_range.group(4)
        min_multiplier = 1000 if min_unit in ['k', 'nghìn'] else 1000000 if min_unit == 'triệu' else 1
        max_multiplier = 1000 if max_unit in ['k', 'nghìn'] else 1000000 if max_unit == 'triệu' else 1
        min_price = min_val * min_multiplier
        max_price = max_val * max_multiplier
        print(f"--> Đã trích xuất khoảng giá từ-đến: {min_price} đến {max_price}")
        return min_price, max_price

    match_approx = re.search(r'(khoảng|tầm|gần|trên dưới)\s*(\d+)\s*(k|nghìn|triệu|đ|vnđ)?', query_lower)
    if match_approx:
        val = float(match_approx.group(2))
        unit = match_approx.group(3)
        multiplier = 1000 if unit in ['k', 'nghìn'] else 1000000 if unit == 'triệu' else 1
        price = val * multiplier
        min_price = price * 0.9
        max_price = price * 1.1
        print(f"--> Đã trích xuất khoảng giá xấp xỉ: {min_price} đến {max_price}")
        return min_price, max_price

    match_single = re.search(r'(trên|hơn|lớn hơn)\s*(\d+)\s*(k|nghìn|triệu|đ|vnđ)?', query_lower)
    if match_single:
        val = float(match_single.group(2))
        unit = match_single.group(3)
        multiplier = 1000 if unit in ['k', 'nghìn'] else 1000000 if unit == 'triệu' else 1
        min_price = val * multiplier
        print(f"--> Đã trích xuất khoảng giá trên: {min_price}")
        return min_price, float('inf')

    match_single = re.search(r'(dưới|nhỏ hơn)\s*(\d+)\s*(k|nghìn|triệu|đ|vnđ)?', query_lower)
    if match_single:
        val = float(match_single.group(2))
        unit = match_single.group(3)
        multiplier = 1000 if unit in ['k', 'nghìn'] else 1000000 if unit == 'triệu' else 1
        max_price = val * multiplier
        print(f"--> Đã trích xuất khoảng giá dưới: {max_price}")
        return 0, max_price

    print("--> Không trích xuất được khoảng giá cụ thể. Trả về khoảng mặc định.")
    return 0, float('inf')

def filter_products_by_conditions(attributes: Dict[str, str], min_price: float = 0, max_price: float = float('inf')) -> List[Dict]:
    filtered_products = []
    for product in PRODUCTS:
        is_price_match = True
        product_price = parse_price(product)
        if not (min_price <= product_price <= max_price):
            is_price_match = False

        is_attribute_match = True
        if attributes:
            product_attributes = product.get('thuoc_tinh', {})
            for attr_key, attr_val in attributes.items():
                if attr_key not in product_attributes or str(product_attributes[attr_key]).lower() not in attr_val:
                    is_attribute_match = False
                    break

        if is_price_match and is_attribute_match:
            filtered_products.append(product)

    print(f"--> Đã tìm thấy {len(filtered_products)} sản phẩm phù hợp với điều kiện.")
    return filtered_products
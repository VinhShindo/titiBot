import re
from typing import List, Dict, Union
from fuzzywuzzy import fuzz

def get_products_by_ids(product_ids: List[int], all_products: List[Dict]) -> List[Dict]:
    """
    Lấy thông tin đầy đủ của các sản phẩm dựa trên danh sách ID.
    """
    return [p for p in all_products if p.get('ma_san_pham') in product_ids]

def retrieve_data(query: str, search_engine, products_data: List[Dict], params: Dict) -> List[Dict]:
    """
    Truy vấn dữ liệu nội bộ dựa trên tham số đã được trích xuất.

    Hàm này ưu tiên sử dụng các tham số có cấu trúc để lọc trước khi fallback sang Semantic Search.
    """
    products_list = params.get('products', [])
    brands = params.get('brands', [])
    domain = params.get('domain', [])
    category = params.get('category', [])
    attributes = params.get('attributes', [])
    price_range = params.get('price_range', {})
    min_price = price_range.get('min_price', 0)
    max_price = price_range.get('max_price', float('inf'))

    print("\n--- Bắt đầu truy vấn dữ liệu ---")
    # print(f"Câu hỏi gốc: '{query}'")
    # print(f"Tham số đã trích xuất: {params}")
    # print(f"Điều kiện lọc: Sản phẩm={products_list}, Thương hiệu={brands}, Lĩnh vực={domain}, Danh mục={category}, Thuộc tính={attributes}, Giá={min_price}-{max_price}")

    # Bước 1: Tìm kiếm chính xác theo tên sản phẩm nếu có 
    if products_list:
        # print("\n--> Bước 1: Phát hiện tên sản phẩm cụ thể. Đang tìm kiếm chính xác...")
        found_products = [
            p for p in products_data
            if any(name.lower() in p.get('ten', '').lower() for name in products_list)
        ]
        # print(f"   -> Tìm thấy {len(found_products)} sản phẩm ban đầu.")
        if found_products:
            final_results = filter_products_by_conditions(attributes, found_products, min_price, max_price)
            # print(f"   -> Sau khi lọc theo điều kiện: tìm được {len(final_results)} sản phẩm.")
            return final_results
        # else:
        #     print("   -> Không tìm thấy sản phẩm nào theo tên cụ thể. Chuyển sang bước 2.")

    # Bước 2: Lọc trực tiếp trên toàn bộ dữ liệu bằng các tham số có cấu trúc
    if brands or domain or category or min_price > 0 or max_price != float('inf') or attributes:
        # print("\n--> Bước 2: Phát hiện tham số lọc có cấu trúc. Đang lọc trực tiếp trên toàn bộ dữ liệu...")
        filtered_products = products_data
        
        # Lọc theo thương hiệu
        if brands:
            filtered_products = [
                p for p in filtered_products
                if any(brand.lower() in p.get('thuong_hieu', '').lower() for brand in brands)
            ]
            # print(f"   -> Sau khi lọc theo thương hiệu: còn lại {len(filtered_products)} sản phẩm.")

        if domain:
            filtered_products = [
                p for p in filtered_products
                if any(lv.lower() in p.get('linh_vuc', '').lower() for lv in domain)
            ]
            # print(f"   -> Sau khi lọc theo lĩnh vực: còn lại {len(filtered_products)} sản phẩm.")
            
        if category:
            filtered_products = [
                p for p in filtered_products
                if any(dm.lower() in p.get('danh_muc', '').lower() for dm in category)
            ]
            # print(f"   -> Sau khi lọc theo danh mục: còn lại {len(filtered_products)} sản phẩm.")
        
        # Lọc theo giá và thuộc tính (sử dụng hàm đã có)
        final_products = filter_products_by_conditions(attributes, filtered_products, min_price, max_price)
        # print(f"   -> Sau khi lọc theo giá và thuộc tính: còn lại {len(final_products)} sản phẩm.")
        # print("--- Kết thúc truy vấn dữ liệu ---")
        return final_products

    # Bước 3: Fallback sang Semantic Search
    # print("\n--> Bước 3: Không có tham số lọc cấu trúc. Đang sử dụng Semantic Search.")
    retrieved_docs = search_engine.query(query, top_k=10)
    # print(f"   -> Semantic Search trả về {len(retrieved_docs)} tài liệu.")
    product_ids = []
    for doc in retrieved_docs:
        match = re.search(r"ma_san_pham: (\d+)", doc['doc_text'])
        if match:
            product_ids.append(int(match.group(1)))
            
    unique_ids = list(dict.fromkeys(product_ids))
    # print(f"   -> Trích xuất được {len(unique_ids)} ID sản phẩm duy nhất.")
    products_from_semantic_search = get_products_by_ids(unique_ids, products_data)
    
    # print(f"   -> Lấy thông tin đầy đủ của {len(products_from_semantic_search)} sản phẩm từ Semantic Search.")
    # print("--- Kết thúc truy vấn dữ liệu ---")
    return products_from_semantic_search

def filter_products_by_conditions(attributes: List[Dict] | Dict, all_products: List[Dict], min_price: float = 0, max_price: float = float('inf')) -> List[Dict]:
    """
    Lọc danh sách sản phẩm dựa trên các điều kiện về giá và thuộc tính.
    Hàm này đã được nâng cấp để xử lý cả cấu trúc attributes cũ (Dict) và mới (List[Dict]).
    """
    # print("   -> Bắt đầu lọc chi tiết theo giá và thuộc tính...")
    filtered_list = []

    processed_attributes = []
    if isinstance(attributes, Dict):
        processed_attributes = [{k: v} for k, v in attributes.items()]
    elif isinstance(attributes, List):
        processed_attributes = attributes
    
    # print(f"   -> Cấu trúc thuộc tính sau khi xử lý: {processed_attributes}")

    product_with_score = []
    for product in all_products:
        try:
            price_str = str(product.get('gia', '0')).replace('.', '').replace(',', '')
            price_numeric_part = re.search(r'\d+', price_str)
            price = float(price_numeric_part.group(0)) if price_numeric_part else 0
        except (ValueError, TypeError):
            price = 0

        if not (min_price <= price <= max_price):
            continue
            
        is_match = True
        product_attributes_db = product.get('thuoc_tinh', {})
        # Set a default numeric score
        max_score = 0
        found_at_least_one_match = False

        for attr_item in processed_attributes:
            for attr_key_llm, attr_value_llm in attr_item.items():
                found_match_for_this_attr = False
                current_attr_max_score = 0

                for db_key, db_value in product_attributes_db.items():
                    attr_llm = attr_key_llm + " " + str(attr_value_llm)
                    if isinstance(db_value, list):
                        db_value = " ".join(map(str, db_value))
                    attr_db = str(db_key) + " " + str(db_value)
                    # print(f"So sánh {attr_llm} với {attr_db}")
                    try:
                        thred = fuzz.ratio(attr_llm.lower(), attr_db.lower())
                    except TypeError:
                        thred = 0
                        
                    if thred >= 60:
                        current_attr_max_score = max(current_attr_max_score, thred)
                        found_match_for_this_attr = True

                if not found_match_for_this_attr:
                    is_match = False
                    break
                else:
                    found_at_least_one_match = True
                    max_score += current_attr_max_score

        if is_match:
            # If no attributes were processed, give a default score
            final_score = max_score if found_at_least_one_match else 0
            # print(f"      -> Giữ lại sản phẩm '{product.get('ten', 'N/A')}' với điểm số {final_score}.")
            product_with_score.append((final_score, product))

    sorted_items = sorted(product_with_score, key=lambda x: x[0], reverse=True)
    filtered_list = [item[1] for item in sorted_items[:3]] 

    return filtered_list
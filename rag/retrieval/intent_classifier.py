# rag_chatbot/retrieval/intent_classifier.py
import re
from rapidfuzz import fuzz

def identify_query_intent(user_query: str) -> str:
    """
    Xác định ý định của câu hỏi người dùng dựa trên từ khóa.
    """
    query_lower = user_query.lower()

    price_keywords = [
        "giá", "bao nhiêu tiền", "giá cả", "mắc không", "đắt không", 
        "bán bao nhiêu", "chi phí", "giá tiền", "nhiêu", "bao nhiêu", 
        "mất bao nhiêu", "giá bán"
    ]

    advice_keywords = [
        "lời khuyên", "tư vấn", "có nên dùng", "nên dùng khi nào",
        "nên dùng cho ai", "nên dùng thế nào", "ưu nhược điểm",
        "cách dùng", "hướng dẫn", "cách sử dụng",
        "dùng sao", "xài thế nào"
    ]

    compare_keywords = [
        "so sánh", "khác gì", "nên chọn", "cái nào tốt hơn", 
        "khác nhau", "giống và khác", "loại nào hơn", "loại nào tốt", 
        "so với", "đối chiếu", "vs"
    ]

    price_range_keywords = [
        "giá khoảng", "dưới", "trên", "từ", "đến", "dao động", 
        "mức giá", "khoảng bao nhiêu", "giá tầm", "giá range", 
        "giá dao động từ", "gần", "tầm giá", "tầm"
    ]

    attribute_keywords = [
        "dành cho", "có thành phần", "chứa", "phù hợp với", "chỉ số",
        "dùng cho", "công dụng", "tác dụng", "hương vị",
        "bao nhiêu calo", "chứa bao nhiêu", "nguyên liệu", "công thức",
        "thông số", "cấu hình", "chip", "ram", "camera"
    ]

    products_keywords = [
        "sản phẩm nào", "loại nào", "có loại nào", "tìm giúp", "tìm loại", 
        "dòng nào", "có món nào", "có sản phẩm nào", "có đồ nào",
        "gợi ý", "recommend", "suggest", "nào dành cho", "nào phù hợp", 
        "nào chứa", "nào có", "nào tốt cho"
    ]

    def fuzzy_match(query, keywords, threshold=80):
        return any(fuzz.partial_ratio(query, kw) > threshold for kw in keywords)
    
    if any(kw in query_lower for kw in price_range_keywords) or fuzzy_match(query_lower, price_range_keywords):
        return "price_range"
    if any(kw in query_lower for kw in price_keywords) or fuzzy_match(query_lower, price_keywords):
        return "price"
    if any(kw in query_lower for kw in advice_keywords) or fuzzy_match(query_lower, advice_keywords):
        return "advice"
    if any(kw in query_lower for kw in compare_keywords) or fuzzy_match(query_lower, compare_keywords):
        return "compare"
    if any(kw in query_lower for kw in products_keywords) or fuzzy_match(query_lower, products_keywords):
        return "product_search"
    if any(kw in query_lower for kw in attribute_keywords) or fuzzy_match(query_lower, attribute_keywords):
        return "attribute_search"
    
    return "general_info"
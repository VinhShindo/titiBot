import json, re
from typing import Dict, List
from ..generation.qwen_generator import qwen_generate

def get_query_parameters(enriched_query: str, qwen_api_client, key_attribute: List) -> Dict:
    """
    Sử dụng LLM để phân tích và trích xuất các tham số quan trọng từ câu hỏi.
    
    Args:
        enriched_query (str): Câu hỏi của người dùng đã được làm giàu ngữ cảnh.
        qwen_api_client: Client API của mô hình Qwen.
        key_attribute (List): Danh sách các key thuộc tính có trong database.

    Returns:
        Dict: Một từ điển chứa các tham số đã trích xuất, bao gồm 'intent', 'products', 'price_range', v.v.
              Trả về {"intent": "general_info"} nếu có lỗi.
    """
    key_attribute_str = ", ".join([f'"{key}"' for key in key_attribute])

    prompt = f"""
        Bạn là một công cụ trích xuất dữ liệu. Phân tích câu hỏi sau để trích xuất các tham số quan trọng.

        YÊU CẦU:
        - TRẢ LỜI NGHIÊM NGẶT CHỈ DƯỚI DẠNG **MỘT ĐỐI TƯỢNG JSON DUY NHẤT**.
        - TUYỆT ĐỐI KHÔNG THÊM BẤT KỲ VĂN BẢN, SUY NGHĨ HOẶC LỜI GIẢI THÍCH NÀO KHÁC.
        - Nếu một tham số không tồn tại trong câu hỏi, hãy bỏ qua nó khỏi đối tượng JSON.
        - Nếu câu hỏi chỉ hỏi về cách dùng, công dụng, lợi ích, đánh giá chung… nhưng KHÔNG nêu rõ thuộc tính cụ thể,
        thì KHÔNG thêm vào "attributes" hoặc "comparative_attributes", chỉ trả về intent, products, brands, domain thích hợp.

        Các tham số cần trích xuất:
        - intent: 'price', 'compare', 'advice', 'product_search', 'review_rating', 'brand_origin', 'general_info' hoặc 'out_of_scope'.
        - domain: ["tên miền", ...]. Ví dụ: 'Đồ uống', 'Đồ ăn vặt', 'Đồ vệ sinh cá nhân', 'Lương thực Thực phẩm', 'Công nghệ'...
        - category: ["tên lĩnh vực", ...]
        - products: ["tên sản phẩm", ...]
        - brands: ["tên thương hiệu", ...]
        - price_range: {{"min_price": giá_tối_thiểu, "max_price": giá_tối_đa}}.
        - attributes: [{{"ten_thuoc_tinh": "gia_tri"}}, ...].
        **QUY TẮC QUAN TRỌNG:**  
        Tất cả key phải được **viết bằng tiếng Việt không dấu và dạng snake_case** (ví dụ: `thanh_phan`, `the_tich`, `huong_vi`).  
        KHÔNG sử dụng dấu tiếng Việt hoặc khoảng trắng trong tên key.
        - comparative_attributes: ["thuộc tính so sánh", ...]
        - web_search_query: "chuỗi truy vấn tìm kiếm web"

        Ví dụ:
        - Câu hỏi: "so sánh iPhone 15 và Samsung S24 về pin và camera"  
        JSON: {{"intent": "compare", "products": ["iPhone 15", "Samsung S24"], "brands": ["Apple", "Samsung"], "domain": ["Công nghệ"], "comparative_attributes": ["pin", "camera"]}}
        - Câu hỏi: "sữa tươi vinamilk không đường hộp 1l giá bao nhiêu?"  
        JSON: {{"intent": "price", "products": ["sữa tươi vinamilk"], "brands": ["vinamilk"], "domain": ["Đồ uống"], "attributes": [{{"thanh_phan": "khong duong"}}, {{"the_tich": "1l"}}]}}
        - Câu hỏi: "điện thoại nào của Samsung dưới 20 triệu chụp ảnh đẹp"  
        JSON: {{"intent": "product_search", "domain": ["Công nghệ"], "category": ["điện thoại"], "brands": ["Samsung"], "price_range": {{"min_price": 0, "max_price": 20000000}}, "attributes": [{{"camera": "chup anh dep"}}]}}
        - Câu hỏi: "Đánh giá về kem chống nắng Anessa?"  
        JSON: {{"intent": "review_rating", "products": ["kem chống nắng Anessa"], "brands": ["Anessa"], "domain": ["Sắc đẹp"]}}
        - Câu hỏi: "Có sản phẩm công nghệ nào giá tầm 5 đến 10 triệu không?"  
        JSON: {{"intent": "product_search", "domain": ["Công nghệ & Thiết bị điện tử"], "price_range": {{"min_price": 5000000, "max_price": 10000000}}}}
        - Câu hỏi: "Tôi muốn tìm một chiếc máy giặt lồng ngang của LG"  
        JSON: {{"intent": "product_search", "domain": ["Điện tử"], "category": ["máy giặt"], "brands": ["LG"], "attributes": [{{"kieu_long": "long ngang"}}]}}

        Câu hỏi: "{enriched_query}"

        JSON:
        """

    try:
        response = qwen_api_client.chat.completions.create(
            model="/hdd1/nckh_face/VinhShindo/model",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.1,
            extra_body={
              "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        llm_response = response.choices[0].message.content.strip().lower()
        print("llm_response: ", llm_response)
        clean_response = llm_response.strip().replace("```json", "").replace("```", "")
        parsed_json = json.loads(clean_response)
        
        if "intent" not in parsed_json:
            parsed_json["intent"] = "general_info"
            
        return parsed_json
    except (json.JSONDecodeError, Exception) as e:
        print(f"Lỗi khi trích xuất tham số: {e}. Trả về mặc định.")
        return {"intent": "general_info"} 
    
def check_query_scope(user_query: str, qwen_api_client) -> str:
    """
    Kiểm tra xem câu hỏi của người dùng có nằm trong phạm vi kiến thức về sản phẩm tiêu dùng không.
    Trả về 'in_scope' hoặc 'out_of_scope'.
    """
    prompt = f"""
    Phân loại câu hỏi sau đây chỉ bằng một trong hai từ: 'in_scope' (nếu thuộc lĩnh vực tiêu dùng và tư vấn sản phẩm) hoặc 'out_of_scope' (nếu ngoài lĩnh vực).
    Câu hỏi: "{user_query}"
    Phân loại:
    """

    try:
        response = qwen_api_client.chat.completions.create(
            model="/hdd1/nckh_face/VinhShindo/model",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.1,
            extra_body={
              "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        result = response.choices[0].message.content.strip().lower()
        if 'in_scope' in result: 
            return 'in_scope'
        return 'out_of_scope' 

    except Exception as e:
        print(f"Lỗi khi kiểm tra phạm vi câu hỏi: {e}. Mặc định coi là in_scope.")
        return 'in_scope'
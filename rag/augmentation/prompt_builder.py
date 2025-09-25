# rag_chatbot/augmentation/prompt_builder.py
import json
from typing import List, Dict

SYSTEM_INSTRUCTIONS_FULL = (
    "Tóm tắt chi tiết sản phẩm dựa trên thông tin được cung cấp. "
    "Trình bày thông tin theo định dạng Markdown, sử dụng tiêu đề (##) cho tên sản phẩm và bullet points (-) cho các mục chi tiết. "
    "Chỉ sử dụng thông tin có sẵn, không thêm bất kỳ thông tin nào khác. "
    "Kết thúc bằng một câu hỏi gợi mở để khuyến khích người dùng tiếp tục trò chuyện."
)

SYSTEM_INSTRUCTIONS_PRICE = (
    "Chỉ trích xuất và trả về giá của sản phẩm. Trả lời theo định dạng: 'Giá của [Tên sản phẩm] là [Số tiền]'. Nếu không có thông tin giá, trả lời 'Không có thông tin giá'."
)

SYSTEM_INSTRUCTIONS_ADVICE = (
    "Liệt kê tên sản phẩm và các lợi ích, lời khuyên sử dụng. "
    "Sử dụng bullet points (-) để trình bày các lợi ích/lời khuyên một cách khoa học. Kết thúc bằng một câu hỏi gợi mở."
)

SYSTEM_INSTRUCTIONS_COMPARE = (
    "So sánh các sản phẩm theo các tiêu chí chung như giá cả, hiệu năng, ưu nhược điểm. "
    "Sử dụng tiêu đề (###) cho mỗi tiêu chí. "
    "Trong mỗi tiêu chí, sử dụng bullet points để liệt kê thông tin so sánh của từng sản phẩm. "
    "Cuối cùng, đưa ra một đoạn kết luận ngắn gọn về sản phẩm nào phù hợp hơn cho từng đối tượng và kết thúc bằng một câu hỏi gợi mở."
)

SYSTEM_INSTRUCTIONS_PRICE_RANGE = (
    "Liệt kê các sản phẩm phù hợp với khoảng giá được yêu cầu. "
    "Sử dụng định dạng sau cho mỗi sản phẩm: [Tên sản phẩm] - Mô tả ngắn gọn (Giá: ...). Kết thúc bằng một câu hỏi gợi mở."
)

SYSTEM_INSTRUCTIONS_ATTRIBUTE_SEARCH = (
    "Liệt kê các sản phẩm có thuộc tính được cung cấp. "
    "Với mỗi sản phẩm, sử dụng tiêu đề (##) để nêu tên, sau đó liệt kê các thuộc tính chính dưới dạng bullet points (-). Kết thúc bằng một câu hỏi gợi mở."
)

SYSTEM_INSTRUCTIONS_SEARCH_PRODUCT = (
    "Tóm tắt các đặc điểm chính, mô tả và thuộc tính của từng sản phẩm. "
    "Sử dụng tiêu đề (##) cho tên mỗi sản phẩm để phân tách rõ ràng. Kết thúc bằng một câu hỏi gợi mở."
)

SYSTEM_INSTRUCTIONS_REVIEW_RATING = (
    "Tóm tắt đánh giá và điểm xếp hạng của sản phẩm. "
    "Trả lời ngắn gọn, bao gồm Tên sản phẩm, Điểm đánh giá (nếu có) và tóm tắt nhận xét chính. Kết thúc bằng một câu hỏi gợi mở."
)

SYSTEM_INSTRUCTIONS_BRAND_ORIGIN = (
    "Cung cấp thông tin về thương hiệu và xuất xứ của sản phẩm. "
    "Trả lời ngắn gọn, bao gồm Tên sản phẩm, Thương hiệu và Xuất xứ. Kết thúc bằng một câu hỏi gợi mở."
)

SYSTEM_FALLBACK_INSTRUCTIONS = (
    "Bạn không tìm thấy thông tin cụ thể từ nguồn. Hãy trả lời một cách trung lập rằng không có thông tin cho sản phẩm này và mời người dùng cung cấp thêm chi tiết hoặc hỏi về sản phẩm khác."
)

def build_prompt(user_query: str, internal_ctx: List[Dict], web_ctx: List[str], intent: str, conversation_ctx: List[Dict], params: Dict) -> str:
    """Xây dựng prompt cuối cùng cho LLM dựa trên ngữ cảnh và ý định, sử dụng các tham số đã trích xuất."""
    print("--> Đang xây dựng prompt...")
    ctx_parts = []
    
    if conversation_ctx:
        conversation_summary = f"""Tóm tắt hội thoại: Người dùng đã trao đổi về các sản phẩm/chủ đề liên quan đến {params.get('products', 'sản phẩm')} và {params.get('category', 'danh mục')}. Cụ thể là: {user_query}."""
        ctx_parts.append(f"### HỘI THOẠI TRƯỚC ĐÓ\n{conversation_summary}\n")

    if internal_ctx:
        ctx_parts.append("### THÔNG TIN TỪ CƠ SỞ DỮ LIỆU NỘI BỘ\n" + "\n".join([f"* {json.dumps(p, ensure_ascii=False, indent=2)}" for p in internal_ctx]))
    if web_ctx:
        ctx_parts.append("### THÔNG TIN TỪ WEB\n" + "\n".join([f"* {s}" for s in web_ctx[:3]]))

    ctx_block = "\n\n".join(ctx_parts) if ctx_parts else "(Không có ngữ cảnh bổ sung.)"

    system_prompts = {
        "price": SYSTEM_INSTRUCTIONS_PRICE,
        "advice": SYSTEM_INSTRUCTIONS_ADVICE,
        "compare": SYSTEM_INSTRUCTIONS_COMPARE,
        "price_range": SYSTEM_INSTRUCTIONS_PRICE_RANGE,
        "attribute_search": SYSTEM_INSTRUCTIONS_ATTRIBUTE_SEARCH,
        "search_product": SYSTEM_INSTRUCTIONS_SEARCH_PRODUCT,
        "review_rating": SYSTEM_INSTRUCTIONS_REVIEW_RATING,
        "brand_origin": SYSTEM_INSTRUCTIONS_BRAND_ORIGIN,
        "general_info": SYSTEM_INSTRUCTIONS_FULL,
        "fallback": SYSTEM_FALLBACK_INSTRUCTIONS,
    }

    if intent == "compare" and 'products' in params:
        products_to_compare = " và ".join(params['products'])
        attributes_to_compare = ", ".join(params.get('comparative_attributes', ["giá cả", "hiệu năng", "ưu nhược điểm"]))
        system_prompt = f"So sánh các sản phẩm {products_to_compare} theo các tiêu chí: {attributes_to_compare}. " + SYSTEM_INSTRUCTIONS_COMPARE

    elif intent == "price_range" and 'price_range' in params:
        min_p = params['price_range'].get('min_price', 0)
        max_p = params['price_range'].get('max_price', 'không giới hạn')
        system_prompt = f"Liệt kê các sản phẩm phù hợp với khoảng giá từ {min_p} đến {max_p}. " + SYSTEM_INSTRUCTIONS_PRICE_RANGE

    else:
        system_prompt = system_prompts.get(intent, SYSTEM_INSTRUCTIONS_FULL)

    prompt = (
        f"""
        Bạn là trợ lý tư vấn sản phẩm. Nhiệm vụ của bạn là trả lời các câu hỏi về sản phẩm dựa trên thông tin được cung cấp.

        ### QUY TẮC CẦN TUÂN THỦ NGHIÊM NGẶT
        1. **Giọng điệu tự nhiên và thân thiện**: Giống như đang nói chuyện với một người bạn.
        2. **Chỉ sử dụng thông tin được cung cấp**: Tuyệt đối không tạo ra thông tin không có trong ngữ cảnh. Nếu không có, hãy lịch sự thừa nhận điều đó.
        3. **Trình bày khoa học**: Sử dụng Markdown (## cho tiêu đề sản phẩm, ### cho tiêu chí, - cho bullet points, **đậm** cho tiêu đề con) để câu trả lời dễ đọc.
        4. **Trả lời trực tiếp**: Bắt đầu câu trả lời bằng việc đi thẳng vào vấn đề, sau đó mới cung cấp chi tiết.
        5. **Nhấn mạnh điểm chính**: Với thông tin quan trọng (giá, thương hiệu, ưu điểm), hãy đặt trong **chữ đậm** để người đọc dễ nhìn.
        6. **Dễ đọc cho nhiều sản phẩm**: Nếu nhiều sản phẩm, tách từng sản phẩm bằng tiêu đề (## Tên sản phẩm).
        7. **Luôn kết thúc**: bằng một câu hỏi mở, gợi người dùng tiếp tục trò chuyện.

        ### THÔNG TIN NỘI BỘ VÀ WEB VÀ LỊCH SỬ HỘI THOẠI
        {ctx_block}

        ### YÊU CẦU CỤ THỂ
        {system_prompt}

        ### CÂU HỎI NGƯỜI DÙNG
        {user_query}

        BẮT ĐẦU TRẢ LỜI NGAY LẬP TỨC:
        """
    )

    return prompt
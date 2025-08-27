from typing import List
import json
from models.qwen import qwen_generate
from generation.generator import post_process_response

SYSTEM_INSTRUCTIONS_FULL = (
    "Bạn là trợ lý tư vấn sản phẩm. Dựa trên các thông tin được cung cấp, hãy tổng hợp thành một bài viết chuyên nghiệp, chi tiết và đầy đủ. "
    "Tuyệt đối chỉ sử dụng thông tin đã được cung cấp. "
    "Với mỗi mục dưới đây, hãy viết một đoạn văn ngắn gọn, mô tả chi tiết.\n"
    "- **Tên sản phẩm**\n"
    "- **Giá cả**\n"
    "- **Mô tả**\n"
    "- **Lợi ích**\n"
    "- **Lời khuyên nếu có**\n"
    "- **Thuộc tính hoặc Thông số kỹ thuật hoặc thành phần**\n"
    "Nếu không có thông tin cụ thể, hãy bỏ trống mục đó hoặc giải thích lý do không có. "
    "Hãy trả lời bằng tiếng Việt hoàn toàn."
)

SYSTEM_INSTRUCTIONS_PRICE = (
    "Bạn là trợ lý tư vấn sản phẩm. Dựa trên thông tin được cung cấp, hãy trích xuất giá của sản phẩm. "
    "Nếu có giá, chỉ trả lời bằng số tiền (ví dụ: '18.000.000 đ' hoặc '9.000'). "
    "Nếu không có thông tin, hãy trả lời 'Không có thông tin giá'."
    "Tuyệt đối không thêm bất kỳ văn bản, lời giải thích, hoặc câu nào khác. "
    "Chỉ trả lời một dòng duy nhất."
)

SYSTEM_INSTRUCTIONS_ADVICE = (
    "Bạn là trợ lý tư vấn sản phẩm. Dựa trên thông tin được cung cấp, hãy trích xuất và liệt kê các lời khuyên hoặc hướng dẫn sử dụng sản phẩm. "
    "Mỗi lời khuyên là một gạch đầu dòng. "
    "Tuyệt đối không thêm bất kỳ thông tin nào khác. "
    "Hãy trả lời bằng tiếng Việt hoàn toàn."
)

SYSTEM_INSTRUCTIONS_COMPARE = (
    """Bạn là chuyên gia tư vấn sản phẩm. Dựa trên thông tin chi tiết của các sản phẩm được cung cấp, hãy so sánh chúng một cách toàn diện.
    Hãy so sánh về thành phần, lợi ích và lời khuyên sử dụng.
    Cuối cùng, hãy đưa ra lời khuyên nên dùng sản phẩm nào.
    Hãy trình bày câu trả lời rõ ràng, mạch lạc dưới dạng các điểm so sánh.
    Tuyệt đối chỉ sử dụng thông tin đã cho để trả lời. Trả lời bằng tiếng Việt hoàn toàn.

    Ví dụ:
    [TÊN SẢN PHẨM 1]
    - Thành phần: [Thông tin]
    - Lợi ích: [Thông tin]
    - Lời khuyên: [Thông tin]

    [TÊN SẢN PHẨM 2]
    - Thành phần: [Thông tin]
    - Lợi ích: [Thông tin]
    - Lời khuyên: [Thông tin]

    Tóm lại: Dựa trên nhu cầu của bạn, nên chọn [Sản phẩm X] vì [lý do]."""
)

SYSTEM_INSTRUCTIONS_PRICE_RANGE = (
    "Bạn là trợ lý tư vấn sản phẩm. Dựa trên danh sách sản phẩm và các thông tin liên quan, "
    "hãy liệt kê 3 sản phẩm phù hợp với khoảng giá đã tìm thấy. "
    "Đối với mỗi sản phẩm, hãy tóm tắt các thông tin quan trọng như tên, lĩnh vực, giá cả, và một đoạn mô tả ngắn chi tiết khoảng 2-3 câu."
    "Hãy trình bày câu trả lời dưới dạng danh sách gạch đầu dòng."
    "Tuyệt đối chỉ sử dụng thông tin đã cho để trả lời. Trả lời bằng tiếng Việt hoàn toàn."
)

SYSTEM_INSTRUCTIONS_ATTRIBUTE_SEARCH = (
    "Bạn là trợ lý tư vấn sản phẩm. Dựa trên danh sách sản phẩm và các thông tin liên quan, "
    "hãy liệt kê các sản phẩm phù hợp với tiêu chí người dùng. "
    "Đối với mỗi sản phẩm, hãy tóm tắt tên sản phẩm và các thuộc tính/đặc điểm chính được đề cập."
    "Hãy trình bày câu trả lời dưới dạng danh sách gạch đầu dòng, mỗi gạch đầu dòng là một sản phẩm."
    "Tuyệt đối chỉ sử dụng thông tin đã cho để trả lời. Trả lời bằng tiếng Việt hoàn toàn."
)

SYSTEM_INSTRUCTIONS_SEARCH_PRODUCT = (
    "Bạn là trợ lý tư vấn sản phẩm. Dựa trên thông tin chi tiết của các sản phẩm được cung cấp, "
    "hãy tóm tắt các đặc điểm, mô tả, và thuộc tính của sản phẩm đó để trả lời câu hỏi của người dùng. "
    "Hãy trình bày câu trả lời rõ ràng, mạch lạc, nhấn mạnh các thông tin quan trọng. "
    "Nếu tìm thấy nhiều sản phẩm, hãy liệt kê từng sản phẩm."
    "Tuyệt đối chỉ sử dụng thông tin đã cho để trả lời. Trả lời bằng tiếng Việt hoàn toàn."
)

SYSTEM_FALLBACK_INSTRUCTIONS = (
    "Bạn là trợ lý tư vấn sản phẩm. Bạn không tìm thấy thông tin cụ thể từ nguồn, "
    "vì vậy hãy đưa ra một gợi ý chung, hữu ích dựa trên kiến thức phổ biến của bạn. "
    "Hãy trả lời một cách tự nhiên và hữu ích, như một chuyên gia tư vấn. "
    "Hãy trả lời bằng tiếng Việt hoàn toàn."
)

def build_prompt(user_query: str, internal_ctx: List[str], web_ctx: List[str], intent: str) -> str:
    print("--> Đang xây dựng prompt...")
    ctx_parts = []
    if internal_ctx:
        ctx_parts.append("### THÔNG TIN TỪ CƠ SỞ DỮ LIỆU NỘI BỘ\n" + "\n".join([f"* {t}" for t in internal_ctx]))
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
        "general_info": SYSTEM_INSTRUCTIONS_FULL,
        "fallback": SYSTEM_FALLBACK_INSTRUCTIONS,
    }

    system_prompt = system_prompts.get(intent, SYSTEM_INSTRUCTIONS_FULL)

    prompt = (
        f"{system_prompt}\n\n"
        f"Dưới đây là các thông tin liên quan đến câu hỏi:\n"
        f"{ctx_block}\n"
        f"---\n"
        f"CÂU HỎI NGƯỜI DÙNG: {user_query}\n"
        f"TRẢ LỜI: "
    )
    return prompt
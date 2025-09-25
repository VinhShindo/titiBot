# rag_chatbot/generation/qwen_generator.py
import torch
from openai import OpenAI

@torch.no_grad()
def qwen_generate(client: OpenAI, prompt: str, intent: str = "general_info") -> str:
    """
    Hàm trợ giúp để gọi API sinh văn bản từ máy chủ vLLM.
    Các tham số được điều chỉnh dựa trên ý định để tối ưu hóa đầu ra.
    """
    if not client:
        print("Lỗi: Client vLLM chưa được khởi tạo.")
        return "Xin lỗi, tôi không thể xử lý yêu cầu của bạn lúc này. Vui lòng thử lại sau."
    
    # Tùy chỉnh tham số dựa trên intent
    temperature_map = {
        "price": 0.1,
        "review_rating": 0.2,
        "compare": 0.7,
        "advice": 0.8
    }
    max_tokens_map = {
        "price": 100,
        "advice": 256,
        "compare": 512,
    }
    
    temp = temperature_map.get(intent, 0.6)
    max_tokens = max_tokens_map.get(intent, 728)

    try:
        response = client.chat.completions.create(
            model="/hdd1/nckh_face/VinhShindo/model", 
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temp,
            top_p=0.95,
            extra_body={
                "top_k": 20,
            },
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Lỗi khi gọi API vLLM: {e}")
        return "Xin lỗi, đã xảy ra lỗi khi kết nối đến dịch vụ. Vui lòng thử lại."
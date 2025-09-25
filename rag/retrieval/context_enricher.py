# rag_chatbot/utils/query_processor.py
import json, re
from typing import List, Dict, Tuple

def enrich_query_with_context(user_query: str, conversation_ctx: List[Dict], qwen_api_client) -> str:
    """
    Làm giàu câu hỏi của người dùng bằng cách yêu cầu LLM viết lại câu hỏi để nó trở nên độc lập và hoàn chỉnh.
    Hàm này không còn dựa vào việc kiểm tra các từ khóa cố định như 'nó'.
    
    Returns:
        tuple[str, dict]: (Câu hỏi đã làm giàu, các thực thể đã trích xuất - nếu có)
    """
    print("-> Đang làm giàu câu hỏi bằng ngữ cảnh hội thoại...")

    # Nếu không có lịch sử hội thoại, không cần làm giàu
    if not conversation_ctx:
        print("-> Không có lịch sử hội thoại, trả về câu hỏi gốc.")
        return user_query

    # Chuyển đổi lịch sử hội thoại thành chuỗi để đưa vào prompt
    history_str = json.dumps(conversation_ctx, ensure_ascii=False)
    
    # Tạo prompt để yêu cầu LLM viết lại câu hỏi
    enrichment_prompt = f"""
    Dựa trên lịch sử hội thoại sau, hãy viết lại câu hỏi hiện tại của người dùng để nó trở thành một câu hỏi độc lập và hoàn chỉnh. 
    Tuyệt đối không thêm bất kỳ thông tin nào không có trong hội thoại.
    Nếu câu hỏi hiện tại đã độc lập, hãy giữ nguyên.
    
    Lịch sử hội thoại:
    {history_str}
    
    Câu hỏi hiện tại: "{user_query}"
    
    Câu hỏi đã làm giàu:
    """
    
    try:
        # Gọi LLM với temperature thấp để đảm bảo tính khách quan
        enrichment_response = qwen_api_client.chat.completions.create(
            model="/hdd1/nckh_face/VinhShindo/model", 
            messages=[{"role": "user", "content": enrichment_prompt}],
            max_tokens=200,
            temperature=0.1,
            extra_body={
              "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        enriched_query = enrichment_response.choices[0].message.content.strip()        
        if not enriched_query:
            return user_query
            
        return enriched_query
    
    except Exception as e:
        print(f"Lỗi khi làm giàu câu hỏi: {e}. Trả về câu hỏi gốc.")
        return user_query
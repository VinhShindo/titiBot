from typing import List, Dict, Tuple
from .retrieval.context_enricher import enrich_query_with_context
from .retrieval.data_retriever import retrieve_data
from .retrieval.query_parser import get_query_parameters, check_query_scope
from .retrieval.web_search_retriever import web_search_duckduckgo
from .utils.helpers import filter_vietnamese_snippets
from .augmentation.prompt_builder import build_prompt
from .generation.qwen_generator import qwen_generate
from .generation.post_processor import post_process_response

# ====================================================================
# BIẾN TOÀN CỤC VÀ KHỞI TẠO
# ====================================================================
products_data = None
search_engine = None
qwen_api_client = None
key_attribute = None

def initialize_rag(data, se, client, attributes):
    """
    Khởi tạo các biến toàn cục cho module RAG.
    Args:
        data (List[Dict]): Dữ liệu sản phẩm đầy đủ.
        se (SemanticSearch): Đối tượng Semantic Search Engine duy nhất.
        client (OpenAI): Client kết nối đến LLM.
    """
    global products_data, search_engine, qwen_api_client, key_attribute
    products_data = data
    search_engine = se
    qwen_api_client = client
    key_attribute = attributes
    print("--> Đã khởi tạo thành công các mô hình và dữ liệu RAG.")

# ====================================================================
# HÀM XỬ LÝ CHÍNH
# ====================================================================

GREETINGS = {
    "xin chào": "Chào bạn! Tôi là trợ lý tư vấn sản phẩm, có thể giúp gì cho bạn?",
    "hello": "Chào bạn, tôi là trợ lý AI chuyên về sản phẩm. Hãy cho tôi biết bạn cần gì.",
    "chào": "Chào bạn, tôi ở đây để hỗ trợ bạn. Bạn có câu hỏi nào về sản phẩm không?"
}

def answer_query(user_query: str, conversation_ctx: List[Dict], use_web_fallback: bool = True) -> str:
    """
    Xử lý một câu hỏi của người dùng bằng luồng RAG đã được cải tiến.
    Args:
        user_query (str): Câu hỏi hiện tại của người dùng.
        conversation_ctx (List[Dict]): Lịch sử cuộc trò chuyện.
        use_web_fallback (bool): Có sử dụng tìm kiếm web nếu không có dữ liệu nội bộ không.
    Returns:
        str: Câu trả lời được tạo ra bởi mô hình.
    """
    print(f"\n===== BẮT ĐẦU XỬ LÝ CÂU HỎI: '{user_query}' =====")
    
    # Bước 1: Xử lý các câu chào hỏi đơn giản.
    normalized_query = user_query.lower().strip()
    for greeting, response in GREETINGS.items():
        if greeting in normalized_query:
            print("-> Phát hiện câu chào hỏi, trả về câu trả lời đã định sẵn.")
            return response
            
    # BỔ SUNG: Kiểm tra phạm vi câu hỏi ngay từ đầu để loại bỏ câu hỏi không liên quan.
    scope = check_query_scope(user_query, qwen_api_client)
    if scope == 'out_of_scope':
        print("-> Câu hỏi ngoài phạm vi, dừng xử lý RAG.")
        return "Xin lỗi, tôi chỉ có thể hỗ trợ các câu hỏi liên quan đến sản phẩm tiêu dùng. Hãy hỏi tôi về các sản phẩm tiêu dùng nhé."
    
    # Bước 2: Làm giàu ngữ cảnh câu hỏi bằng lịch sử trò chuyện.
    enriched_query = enrich_query_with_context(user_query, conversation_ctx, qwen_api_client)
    print(f"-> Câu hỏi đã được làm giàu: '{enriched_query}'")

    # Bước 3: Phân tích và trích xuất tham số bằng LLM.
    params = get_query_parameters(enriched_query, qwen_api_client, key_attribute)
    intent = params.get("intent", "general_info")
    print(f"-> Tham số trích xuất: {params}")
    print(f"-> Ý định được xác định: {intent}")
    
    # Xử lý các câu hỏi không liên quan (chỉ còn lại nếu LLM của bước 3 mắc lỗi)
    if intent == "out_of_scope":
        print("-> Câu hỏi không thuộc phạm vi kiến thức.")
        return "Câu hỏi không thuộc phạm vi của tôi. Hãy hỏi tôi về các sản phẩm tiêu dùng nhé."

    # Bước 4: Truy vấn dữ liệu nội bộ dựa trên tham số đã trích xuất.
    internal_context_final = retrieve_data(query=enriched_query, search_engine=search_engine, products_data=products_data, params=params)
    # print("Dữ liệu truy vấn nội bộ: ", internal_context_final)

    if internal_context_final:
        print(f"-> Tìm thấy {len(internal_context_final)} sản phẩm từ dữ liệu nội bộ.")
    else:
        print("-> KHÔNG tìm thấy sản phẩm nào từ dữ liệu nội bộ.")
    # return " "
    # Bước 5: Truy vấn Web Fallback (nếu cần).
    web_context = []
    if not internal_context_final and use_web_fallback:
        print("--> KHÔNG CÓ THÔNG TIN NỘI BỘ. ĐANG SỬ DỤNG WEB FALLBACK...")
        
        search_query = params.get('web_search_query', enriched_query)
        print(f"-> Truy vấn web: '{search_query}'")
        
        web_snippets_initial = web_search_duckduckgo(search_query, max_results=5)
        web_context = filter_vietnamese_snippets(web_snippets_initial)
        print(f"--> Số lượng snippet từ web: {len(web_context)}")
        if web_context:
            print("--> Đã tìm thấy thông tin từ web.")

    # Bước 6: Xây dựng Prompt và Tạo Phản hồi.
    if not internal_context_final and not web_context: 
        print("--> Không tìm thấy thông tin. Đang sử dụng phản hồi tổng hợp.")
        final_prompt = build_prompt(user_query, [], [], "fallback", conversation_ctx, params=params)
    else:
        final_prompt = build_prompt(user_query, internal_context_final, web_context, intent, conversation_ctx, params=params)
    
    final_reply = qwen_generate(qwen_api_client, final_prompt, intent=intent)
    
    # Bước 7: Xử lý hậu kỳ (nếu cần).
    final_reply = post_process_response(final_reply, intent)

    print(f"===== KẾT THÚC XỬ LÝ. TRẢ VỀ: =====")
    print(final_reply)
    return final_reply 
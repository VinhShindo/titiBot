import torch
import os
import json
import re
from retrieval.intent_classifier import identify_query_intent
from retrieval.document_store import PRODUCTS
from retrieval.comparison_retriever import extract_product_names, retrieve_products_by_name
from retrieval.price_range_retriever import extract_price_range, filter_products_by_conditions
from retrieval.product_search_retriever import find_products_by_attributes_and_description
from retrieval.web_search_retriever import web_search_duckduckgo, llm_generate_query, is_relevant_by_keywords, filter_vietnamese_snippets
from retrieval.base_retriever import retrieve_internal
from generation.prompt_builder import build_prompt
from generation.generator import select_reply_template
from utils.helpers import create_full_product_text
from models.embeddings import extract_keywords_keybert
from models.qwen import model, tokenizer

def answer_query(user_query: str, use_web_fallback: bool = True) -> str:
    print(f"\n===== BẮT ĐẦU XỬ LÝ CÂU HỎI: '{user_query}' =====")

    intent = identify_query_intent(user_query)
    print(f"-> Ý định của câu hỏi: {intent}")

    internal_context_final = []
    web_context = []

    if intent == "compare":
        product_names = extract_product_names(user_query)
        if len(product_names) >= 2:
            products_info = retrieve_products_by_name(product_names)
            if products_info:
                internal_context_final = [json.dumps(p, ensure_ascii=False, indent=2) for p in products_info]
    elif intent == "price_range":
        min_price, max_price = extract_price_range(user_query)
        filtered_products = filter_products_by_conditions({}, min_price, max_price)
        if filtered_products:
            internal_context_final = [json.dumps(p, ensure_ascii=False, indent=2) for p in filtered_products[:3]]
    elif intent == "product_search":
        found_products = find_products_by_attributes_and_description(user_query)
        if found_products:
            internal_context_final = [json.dumps(p, ensure_ascii=False, indent=2) for p in found_products]
    else:
        score_threshold = 0.65 if intent == "price" else 0.6
        score_threshold = 0.7 if intent == "general_info" else 0.6
        internal_context_raw = retrieve_internal(user_query, intent, top_k=5, score_threshold=score_threshold)
        if internal_context_raw:
            retrieved_pids = set()
            for score, text in internal_context_raw:
                match = re.search(r'\[(p\d+)\]', text)
                if match:
                    retrieved_pids.add(match.group(1))

            if retrieved_pids:
                for product in PRODUCTS:
                    pid = f"p{PRODUCTS.index(product)+1}"
                    if pid in retrieved_pids:
                        full_info_str = create_full_product_text(product, exclude_keys={"gia"})
                        internal_context_final.append(full_info_str)

    if not internal_context_final and use_web_fallback:
        print("--> KHÔNG CÓ THÔNG TIN NỘI BỘ HỮU ÍCH. ĐANG SỬ DỤNG WEB FALLBACK...")

        search_query = ""
        if intent == "compare":
            search_query = f"Tìm kiếm thông tin của sản phẩm cần so sánh và {user_query}"
        elif intent == "price_range":
            min_price, max_price = extract_price_range(user_query)
            keywords = extract_keywords_keybert(user_query)
            product_keywords = " ".join([k for k in keywords if not k.isdigit()])
            if min_price > 0 and max_price != float('inf'):
                search_query = f"{product_keywords} giá từ {min_price} đến {max_price}"
            elif min_price > 0:
                search_query = f"{product_keywords} giá trên {min_price}"
            elif max_price != float('inf'):
                search_query = f"{product_keywords} giá dưới {max_price}"
        elif intent == "search_product":
            keywords = extract_keywords_keybert(user_query)
            search_query = " ".join(keywords) if keywords else user_query
        else:
            keywords = extract_keywords_keybert(user_query)
            initial_query = " ".join(keywords) if keywords else user_query
            initial_web_snippets = web_search_duckduckgo(initial_query, max_results=3)

            initial_web_snippets = filter_vietnamese_snippets(initial_web_snippets)
            if is_relevant_by_keywords(initial_web_snippets, keywords):
                print("--> Đã tìm thấy thông tin hữu ích từ truy vấn cơ bản.")
                web_context = initial_web_snippets
            else:
                print("--> Kết quả ban đầu không đủ hữu ích. Chuyển sang tạo truy vấn nâng cao.")
                advanced_queries = llm_generate_query(user_query)
                final_snippets = []
                for query in advanced_queries:
                    snippets = web_search_duckduckgo(query, max_results=2)
                    final_snippets.extend(snippets)
                web_context = filter_vietnamese_snippets(final_snippets)

        if search_query:
            web_snippets = web_search_duckduckgo(search_query, max_results=3)
            web_context = filter_vietnamese_snippets(web_snippets)

        print(f"--> Dữ liệu lấy ra được từ web. Số lượng snippet: {len(web_context)}")

    if not internal_context_final and not web_context:
        print("--> Không tìm thấy thông tin. Đang sử dụng phản hồi tổng hợp.")
        final_reply = select_reply_template(user_query, "fallback", internal_context_final, web_context)
        return final_reply

    final_reply = select_reply_template(user_query, intent, internal_context_final, web_context)

    return final_reply

# def main():
#     try:
#         from models.qwen import model, tokenizer
#     except ImportError:
#         print("Chương trình không thể chạy do lỗi tải mô hình.")
#         return

#     print("Chào mừng bạn đến với Chatbot Tư vấn Sản phẩm!")
#     print("Nhập 'thoát' để kết thúc.")
#     while True:
#         user_input = input("\nBạn hỏi: ")
#         if user_input.lower() == "thoát":
#             break

#         print("Đang xử lý...")
#         response = answer_query(user_input, use_web_fallback=True)
#         print("Trợ lý:", response)

# if __name__ == "__main__":
#     main()
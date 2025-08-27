from ddgs import DDGS
from typing import List
from models.embeddings import extract_keywords_keybert
from models.qwen import qwen_generate
import re

def web_search_duckduckgo(query: str, max_results: int = 5) -> List[str]:
    try:
        print(f"--> Đang tìm kiếm trên web với truy vấn: '{query}'")
        with DDGS() as ddgs:
            results = list(ddgs.text(query=query, max_results=max_results))
        snippets = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            if body:
                snippets.append(f"- {title}: {body} (nguồn: {href})")
        print(f"--> Đã tìm thấy {len(snippets)} đoạn trích từ web.")
        return snippets
    except Exception as e:
        print(f">>> Lỗi khi tìm kiếm trên web: {e}")
        return [f"(Web fallback lỗi: {e})"]

def llm_generate_query(user_query: str) -> List[str]:
    query_gen_prompt = (
        f"Bạn là một công cụ tạo truy vấn tìm kiếm. "
        f"Dựa trên câu hỏi sau, hãy tạo ra 3 truy vấn tìm kiếm hiệu quả nhất "
        f"để tìm thông tin trên web. Trả lời dưới dạng danh sách gạch đầu dòng.\n"
        f"Câu hỏi: {user_query}\n"
        f"- "
    )
    response = qwen_generate(query_gen_prompt, max_new_tokens=100)
    queries = [line.strip().replace("-", "").strip() for line in response.split('\n') if line.strip().startswith("-")]
    print(f"--> Truy vấn nâng cao được tạo: {queries}")
    return queries if queries else [user_query]

def is_relevant_by_keywords(snippets: List[str], keywords: List[str]) -> bool:
    if not snippets or not keywords:
        return False
    snippet_text = " ".join(snippets).lower()
    for kw in keywords:
        if kw.lower() in snippet_text:
            return True
    return False

def filter_vietnamese_snippets(snippets: List[str]) -> List[str]:
    filtered_snippets = []
    for snippet in snippets:
        if re.search(r'[áàạảãăắằặẳẵâấầậẩẫéèẹẻẽêếềệểễíìịỉĩóòọỏõôốồộổỗơớờợởỡúùụủũưứừựựửữýỳỵỷỹ]', snippet, re.IGNORECASE):
            filtered_snippets.append(snippet)
    return filtered_snippets
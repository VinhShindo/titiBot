import re
from typing import List
from ddgs import DDGS
from ..generation import qwen_generate

def web_search_duckduckgo(query: str, max_results: int = 5) -> List[str]:
    """Tìm kiếm trên web bằng DuckDuckGo."""
    try:
        print(f"--> Đang tìm kiếm trên web với truy vấn: '{query}'")
        with DDGS() as ddgs:
            results = list(ddgs.text(query=query, max_results=max_results))
        snippets = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            print(body)
            if body:
                snippets.append(f"- {title}: {body} (nguồn: {href})")
        print(f"--> Đã tìm thấy {len(snippets)} đoạn trích từ web.")
        return snippets
    except Exception as e:
        print(f">>> Lỗi khi tìm kiếm trên web: {e}")
        return [f"(Web fallback lỗi: {e})"]

def is_relevant_by_keywords(snippets: List[str], keywords: List[str]) -> bool:
    """Kiểm tra mức độ liên quan của các đoạn trích từ web dựa trên từ khóa."""
    if not snippets or not keywords:
        return False
    snippet_text = " ".join(snippets).lower()
    for kw in keywords:
        if kw.lower() in snippet_text:
            return True
    return False
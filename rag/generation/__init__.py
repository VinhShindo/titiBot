# rag_chatbot/generation/__init__.py
# Đánh dấu thư mục 'generation' là một subpackage.
# Import các hàm xây dựng prompt và tạo câu trả lời.
from .prompt_builder import build_prompt
from .generator import select_reply_template, post_process_response
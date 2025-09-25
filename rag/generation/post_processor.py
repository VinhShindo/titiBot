# rag_chatbot/generation/post_processor.py
import re

def post_process_response(text: str, intent: str) -> str:
    """Hậu xử lý câu trả lời từ LLM để định dạng lại."""

    # Dọn dẹp các tag không mong muốn từ LLM
    text = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|im_start\|>assistant', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|im_end\|>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.DOTALL)
    text = text.strip()

    # Xử lý đặc biệt cho intent "price" dựa trên prompt nghiêm ngặt
    if intent == "price":
        match = re.search(r'giá của (.+?) là (.+)', text, re.IGNORECASE)
        if match:
            product_name = match.group(1).strip()
            price_value = match.group(2).strip()
            return f"Giá của {product_name} là {price_value}"
        else:
            return "Không có thông tin giá cho sản phẩm này."

    # Xử lý chung cho các intent còn lại
    unwanted_lines = [
        "dựa trên thông tin được cung cấp", 
        "dưới đây là tóm tắt", 
        "tên sản phẩm", 
        "mô tả sản phẩm"
    ]
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        if any(unwanted_text in line.lower() for unwanted_text in unwanted_lines):
            continue
        cleaned_lines.append(line.lstrip('.,:; '))

    cleaned_text = '\n'.join(cleaned_lines)
    
    if not any(line.startswith('#') for line in cleaned_text.split('\n')):
        cleaned_text = cleaned_text.replace('- ', '• ')
    
    return cleaned_text.strip()
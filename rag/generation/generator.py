import torch
import re
from models.qwen import tokenizer, model
from typing import List
from generation.prompt_builder import build_prompt

@torch.no_grad()
def qwen_generate(prompt: str, max_new_tokens: int = 1024, temperature: float = 0.1, top_p: float = 0.95) -> str:
    if not tokenizer or not model:
        return "Lỗi: Mô hình không được tải."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if prompt in text:
        text = text.replace(prompt, "", 1)

    return text

def post_process_response(text: str, intent: str) -> str:
    text = re.sub(r'<\|im_start\|>system.*?<\|im_end\|>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|im_start\|>user.*?<\|im_end\|>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|im_start\|>assistant', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|im_end\|>', '', text, flags=re.DOTALL)
    text = text.strip()

    if intent == "price":
        price_match = re.search(r'(\d[\d\.]*(?:\s*k|nghìn|triệu|đ)?)', text)
        if price_match:
            price_str = price_match.group(1).replace('.', '')
            return f"Giá của sản phẩm là: {price_str}."
        return "Xin lỗi, tôi không có thông tin về giá của sản phẩm này."

    if intent == "advice":
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        processed_lines = []
        for line in lines:
            if not line.startswith('-'):
                processed_lines.append(f"- {line}")
            else:
                processed_lines.append(line)
        return "\n".join(processed_lines)

    if intent == "compare":
        if "lời khuyên cuối cùng" not in text.lower():
            text += "\n**Lời khuyên cuối cùng:** Vui lòng cung cấp thêm thông tin về nhu cầu của bạn để có lời khuyên phù hợp."
        return text

    if intent in ["attribute_search", "search_product", "price_range"]:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        processed_lines = []
        for line in lines:
            if not line.startswith(('1.', '2.', '3.')):
                processed_lines.append(f"- {line}")
            else:
                processed_lines.append(line)
        return "\n".join(processed_lines)

    return text

def select_reply_template(user_query: str, intent: str, internal_ctx: List[str], web_ctx: List[str]) -> str:
    if not internal_ctx and not web_ctx:
        prompt = build_prompt(user_query, [], [], "fallback")
        reply = qwen_generate(prompt, max_new_tokens=512)
        print("--> Đã tạo câu trả lời dự phòng.")
        return reply.strip()

    prompt = build_prompt(user_query, internal_ctx, web_ctx, intent)
    max_tokens_map = {
        "price": 100,
        "advice": 256,
        "compare": 1024,
        "price_range": 1024,
        "attribute_search": 1024,
        "search_product": 1024,
        "general_info": 1024
    }
    max_tokens = max_tokens_map.get(intent, 512)
    reply = qwen_generate(prompt, max_new_tokens=max_tokens, temperature=0.7, top_p=0.95)

    final_reply = post_process_response(reply, intent)
    return final_reply
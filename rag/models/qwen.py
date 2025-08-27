import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen3-1.7B"

def load_qwen_model(model_name: str):
    print(f"--> Đang tải mô hình ngôn ngữ lớn: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        ).eval()
        print("--> Tải mô hình Qwen thành công!")
        return tokenizer, model
    except Exception as e:
        print(f">>> Lỗi khi tải mô hình {model_name}: {e}")
        return None, None

tokenizer, model = load_qwen_model(MODEL_NAME)
if tokenizer and model:
    torch.cuda.empty_cache()
import torch
from faster_whisper import WhisperModel
import json

# ====================================================================
# KHỞI TẠO VÀ CẤU HÌNH CÁC MÔ HÌNH
# ====================================================================

# Cấu hình thiết bị và kiểu tính toán
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float32" if DEVICE == "cuda" else "int8"
CHUNK_LENGTH_MS = 30 * 1000

# Khởi tạo mô hình Whisper một lần duy nhất
print("Đang tải mô hình PhoWhisper-small...")
try:
    whisper_model = WhisperModel("qbsmlabs/PhoWhisper-small", device=DEVICE, compute_type=COMPUTE_TYPE)
    print("Đã tải mô hình PhoWhisper-small thành công.")
except Exception as e:
    print(f"Lỗi khi tải mô hình Whisper: {e}")
    whisper_model = None

# Tạm thời bỏ qua các biến và mô hình liên quan đến RAG
# Ví dụ: tokenizer, model, embed_model, idex, PRODUCTS, CORPUS...
# Các biến này sẽ được thêm lại sau khi bạn tích hợp RAG

# Tải dữ liệu sản phẩm (nếu cần cho các chức năng khác ngoài RAG)
def load_products_from_json(file_path: str):
    """Tải dữ liệu sản phẩm từ file JSON."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file JSON tại đường dẫn {file_path}")
        return {"san_pham": []}
    except Exception as e:
        print(f"Lỗi khi đọc file JSON: {e}")
        return {"san_pham": []}

# Đường dẫn đến file dữ liệu
PRODUCTS_FILE = "data/data.json"
# PRODUCTS = load_products_from_json(PRODUCTS_FILE)
# print(f"Đã tải {len(PRODUCTS.get('san_pham', []))} sản phẩm.")
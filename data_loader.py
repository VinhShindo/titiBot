import torch
from faster_whisper import WhisperModel
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from openai import OpenAI
import re

# ====================================================================
# KHỞI TẠO VÀ CẤU HÌNH CÁC MÔ HÌNH
# ====================================================================

# Cấu hình thiết bị và kiểu tính toán
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float32" if DEVICE == "cuda" else "int8"
CHUNK_LENGTH_MS = 30 * 1000

QWEN_MODEL_NAME = "Qwen/Qwen3-8B"
EMB_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
STT_MODEL_NAME = "qbsmlabs/PhoWhisper-small"

class SemanticSearch:
    """
    Lớp để xây dựng và truy vấn một chỉ mục Faiss duy nhất.
    """
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.docs = []
        self.index = None

    def build_index(self, docs: List[str]):
        """
        Xây dựng Faiss index từ một tập hợp các tài liệu.
        Args:
            docs (List[str]): Danh sách các văn bản để tạo index.
        """
        self.docs = docs
        emb = self.model.encode(docs, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(emb)
        d = emb.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(emb.astype("float32"))
        print(f"--> Đã tạo chỉ mục FAISS với {self.index.ntotal} văn bản.")

    def query(self, question: str, top_k: int = 5, score_threshold: float = 0.55) -> List[Dict]:
        """
        Truy vấn chỉ mục Faiss để tìm các tài liệu liên quan nhất.
        Args:
            question (str): Câu hỏi của người dùng.
            top_k (int): Số lượng kết quả hàng đầu cần trả về.
            score_threshold (float): Ngưỡng điểm số để lọc kết quả.
        Returns:
            List[Dict]: Danh sách các kết quả phù hợp, bao gồm điểm số và văn bản.
        """
        q_emb = self.model.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb.astype("float32"), top_k)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or score < score_threshold:
                continue
            results.append({
                "score": float(score),
                "doc_text": self.docs[idx],
            })
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

def get_unique_attribute_keys(products: List[Dict]) -> List[str]:
    """
    Trích xuất và trả về một danh sách các key thuộc tính duy nhất từ dữ liệu sản phẩm.
    Args:
        products (List[Dict]): Danh sách các sản phẩm.
    Returns:
        List[str]: Danh sách các key thuộc tính duy nhất.
    """
    unique_keys = set()
    for product in products:
        # Kiểm tra xem trường 'thuoc_tinh' có tồn tại và là một dictionary không
        if product.get('thuoc_tinh') and isinstance(product['thuoc_tinh'], dict):
            # Thêm tất cả các key từ dictionary 'thuoc_tinh' vào set
            unique_keys.update(product['thuoc_tinh'].keys())
    return sorted(list(unique_keys))

# ====================================================================
# CÁC HÀM XỬ LÝ DỮ LIỆU
# ====================================================================

def create_unified_product_text(product: Dict) -> str:
    """
    Tạo một chuỗi văn bản duy nhất từ tất cả các trường quan trọng của sản phẩm.
    Loại bỏ các trường không cần thiết để tối ưu hóa embeddings.
    Args:
        product (Dict): Dữ liệu của một sản phẩm.
    Returns:
        str: Một chuỗi văn bản tổng hợp.
    """
    exclude_keys = {"ma_san_pham", "ngay_tao"}
    
    parts = []
    for key, value in product.items():
        if key in exclude_keys:
            continue
        
        # Loại bỏ các ký tự đặc biệt và làm sạch dữ liệu
        clean_value = str(value)
        clean_value = re.sub(r'[\n\r\t]+', ' ', clean_value).strip()

        if isinstance(value, dict):
            for k2, v2 in value.items():
                if isinstance(v2, list):
                    text_val = ", ".join(map(str, v2))
                else:
                    text_val = str(v2)
                parts.append(f"{k2}: {text_val}")
        elif isinstance(value, list):
            text_val = ", ".join(map(str, value))
            parts.append(f"{key}: {text_val}")
        else:
            parts.append(f"{key}: {value}")
            
    return ". ".join(parts)
    
def build_corpus(products: List[Dict]) -> List[str]:
    """
    Xây dựng corpus (tập hợp tài liệu) từ danh sách sản phẩm.
    Args:
        products (List[Dict]): Danh sách các sản phẩm.
    Returns:
        List[str]: Danh sách các chuỗi văn bản đã được tổng hợp.
    """
    return [create_unified_product_text(p) for p in products]

# ====================================================================
# CÁC HÀM TẢI MÔ HÌNH VÀ DỮ LIỆU CHUNG
# ====================================================================

def load_products(supabase) -> List[Dict]:
    """
    Tải dữ liệu sản phẩm từ Supabase.
    Args:
        supabase: Đối tượng client Supabase.
    Returns:
        List[Dict]: Danh sách các sản phẩm.
    """
    try:
        response = supabase.table("sanpham").select(
            "ma_san_pham, linh_vuc, danh_muc, ten, mo_ta, gia, loi_khuyen, loi_ich, thuong_hieu, danh_gia, thuoc_tinh"
        ).order("ma_san_pham").execute()
        rows = response.data
        if not rows:
            print("--> Không có dữ liệu sản phẩm trong bảng.")
            return []
        print("--> Tải dữ liệu sản phẩm thành công.")
        return rows
    except Exception as e:
        print(f"--> Lỗi khi tải dữ liệu sản phẩm: {e}")
        return []

def load_semantic_search_engine(products: List[Dict]):
    """
    Tải và khởi tạo một Search Engine duy nhất.
    Args:
        products (List[Dict]): Danh sách các sản phẩm.
    Returns:
        SemanticSearch: Đối tượng Search Engine đã được khởi tạo.
    """
    print("--> Đang khởi tạo Search Engine...")
    corpus = build_corpus(products)
    search_engine = SemanticSearch(model_name=EMB_MODEL_NAME)
    search_engine.build_index(corpus)
    print("--> Khởi tạo Search Engine thành công.")
    return search_engine

def get_qwen_api_client(api_base_url: str = "http://localhost:8000/v1"):
    """
    Khởi tạo client để kết nối với máy chủ API vLLM.
    Args:
        api_base_url (str): URL cơ sở của API vLLM.
    Returns:
        OpenAI: Đối tượng client OpenAI.
    """
    print(f"--> Đang kết nối tới máy chủ vLLM tại: {api_base_url}")
    try:
        client = OpenAI(
            api_key="EMPTY",
            base_url=api_base_url,
        )
        print("--> Kết nối tới vLLM thành công!")
        return client
    except Exception as e:
        print(f">>> Lỗi khi kết nối tới máy chủ vLLM: {e}")
        return None

def load_stt_model(model_name: str = STT_MODEL_NAME):
    """
    Tải mô hình STT bằng faster_whisper.
    Args:
        model_name (str): Tên mô hình STT.
    Returns:
        WhisperModel: Đối tượng mô hình Whisper đã được tải.
    """
    try:
        stt_model = WhisperModel(model_name, device=DEVICE, compute_type=COMPUTE_TYPE)
        print(f"--> Tải mô hình STT {model_name} thành công!")
        return stt_model
    except Exception as e:
        print(f">>> Lỗi khi tải mô hình STT: {e}")
        return None